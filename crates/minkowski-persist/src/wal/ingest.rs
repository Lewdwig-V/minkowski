use super::*;
use crate::{Follower, FollowerError};

const JOURNAL_MAGIC: &[u8; 4] = b"MKI1";
const JOURNAL_HEADER_SIZE: u64 = 24; // magic + history + CRC32

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("source history does not match the ingest journal; rejoin required")]
    HistoryMismatch,
    #[error(transparent)]
    Wal(#[from] WalError),
    #[error(transparent)]
    Follower(#[from] FollowerError),
}

impl From<io::Error> for IngestError {
    fn from(error: io::Error) -> Self {
        Self::Wal(error.into())
    }
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct JournalEntry {
    history: [u8; 16],
    range: WalFrameRange,
}

/// Durable raw-range ingestion for one authoritative source history.
///
/// Owns an initially empty world and its codecs. Restart reconstructs both
/// world state and progress by replaying the entire retained journal. No
/// nonempty baseline installation or journal compaction is supported yet.
/// Change `history` when replacing source history; ingestion checks it on
/// every call. An exclusive file lock prevents simultaneous journal owners.
/// World access goes through `read_at`; local mutation is not exposed.
pub struct JournaledFollower {
    file: File,
    history: [u8; 16],
    limits: WalRangeLimits,
    follower: Follower,
    view: u64,
    world: World,
    codecs: CodecRegistry,
}

impl JournaledFollower {
    /// Create and synchronize a new journal, starting from sequence zero.
    pub fn create(
        dir: &Path,
        history: [u8; 16],
        codecs: CodecRegistry,
        limits: WalRangeLimits,
    ) -> Result<Self, IngestError> {
        limits.validate()?;
        std::fs::create_dir_all(dir)?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .append(true)
            .open(dir.join("ingest.log"))?;
        file.try_lock().map_err(io::Error::from)?;
        let mut header = Vec::from(JOURNAL_MAGIC.as_slice());
        header.extend_from_slice(&history);
        header.extend_from_slice(&crc32fast::hash(&header).to_le_bytes());
        file.write_all(&header)?;
        file.sync_all()?;
        sync_directory_ancestry(dir)?;
        Ok(Self::empty(file, history, codecs, limits))
    }

    /// Restore an empty baseline plus every complete journal entry. An
    /// incomplete final frame is truncated; complete corrupt frames refuse.
    /// No follower/world escapes if journal validation or replay fails.
    pub fn open(
        dir: &Path,
        history: [u8; 16],
        codecs: CodecRegistry,
        limits: WalRangeLimits,
    ) -> Result<Self, IngestError> {
        limits.validate()?;
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .open(dir.join("ingest.log"))?;
        file.try_lock().map_err(io::Error::from)?;
        let mut header = [0; JOURNAL_HEADER_SIZE as usize];
        read_exact_at(&file, 0, &mut header)?;
        if &header[..4] != JOURNAL_MAGIC
            || crc32fast::hash(&header[..20])
                != u32::from_le_bytes(header[20..].try_into().unwrap())
        {
            return Err(WalError::Format("invalid ingest journal header".into()).into());
        }
        if header[4..20] != history {
            return Err(IngestError::HistoryMismatch);
        }
        // A process crash may leave complete but previously unsynced entries
        // in page cache. Synchronize before treating them as replayable input.
        file.sync_all()?;
        sync_directory_ancestry(dir)?;
        let mut this = Self::empty(file, history, codecs, limits);
        // ponytail: replay the full retained journal; verified state-transfer
        // baselines are required before adding journal prefix compaction.
        let mut pos = JOURNAL_HEADER_SIZE;
        let len = this.file.metadata()?.len();
        while pos < len {
            if len - pos >= FRAME_HEADER_SIZE {
                let mut header = [0; FRAME_HEADER_SIZE as usize];
                read_exact_at(&this.file, pos, &mut header)?;
                let payload_len = u32::from_le_bytes(header[..4].try_into().unwrap());
                let length_guard = u64::from_le_bytes(header[8..].try_into().unwrap());
                if length_guard != !u64::from(payload_len) {
                    return Err(
                        WalError::Format("corrupt ingest journal length header".into()).into(),
                    );
                }
            }
            let Some(frame) = RawFrame::read(&this.file, pos)? else {
                this.file.set_len(pos)?;
                this.file.sync_all()?;
                break;
            };
            frame.verify()?;
            let entry = rkyv::from_bytes::<JournalEntry, rkyv::rancor::Error>(&frame.payload)
                .map_err(|e| WalError::Format(format!("invalid ingest journal entry: {e}")))?;
            this.apply_range(
                entry.history,
                &entry.range,
                None::<fn(&File) -> io::Result<()>>,
            )?;
            pos = frame.next_offset();
        }
        Ok(this)
    }

    fn empty(file: File, history: [u8; 16], codecs: CodecRegistry, limits: WalRangeLimits) -> Self {
        let mut world = World::new();
        for id in codecs.registered_ids() {
            codecs.register_one(id, &mut world);
        }
        Self {
            file,
            history,
            limits,
            follower: Follower::new(),
            view: 0,
            world,
            codecs,
        }
    }

    /// Validate, journal, fsync, then apply a range. Only this successful return
    /// supplies a new pull position. Any error poisons this handle and its reads.
    pub fn ingest_frames(
        &mut self,
        history: [u8; 16],
        range: &WalFrameRange,
    ) -> Result<u64, IngestError> {
        self.ingest_with_sync(history, range, File::sync_all)
    }

    fn ingest_with_sync(
        &mut self,
        history: [u8; 16],
        range: &WalFrameRange,
        sync: impl FnOnce(&File) -> io::Result<()>,
    ) -> Result<u64, IngestError> {
        let result = self.apply_range(history, range, Some(sync));
        if result.is_err() {
            self.follower.poison();
        }
        result
    }

    // Both live ingestion and restart use this interpreter. `sync` is absent
    // only for entries read from the already-synchronized journal on open.
    fn apply_range(
        &mut self,
        history: [u8; 16],
        range: &WalFrameRange,
        sync: Option<impl FnOnce(&File) -> io::Result<()>>,
    ) -> Result<u64, IngestError> {
        if self.follower.is_poisoned() {
            return Err(FollowerError::Poisoned.into());
        }
        if history != self.history {
            return Err(IngestError::HistoryMismatch);
        }
        let applied = self.follower.applied_seq();
        if range.from_seq > applied {
            return Err(FollowerError::Gap {
                expected: applied,
                got: range.from_seq,
            }
            .into());
        }
        let plan = ValidatedRange::from_frames_after(
            range,
            self.view,
            applied,
            self.limits,
            &mut self.world,
            &self.codecs,
        )?;
        if range.next_seq <= applied && plan.view() == self.view {
            return Ok(applied);
        }
        if let Some(sync) = sync {
            let entry = JournalEntry {
                history,
                range: range.clone(),
            };
            let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry)
                .map_err(|e| WalError::Format(e.to_string()))?;
            if payload.len() > MAX_FRAME_SIZE {
                return Err(
                    WalError::Format("ingest envelope exceeds maximum frame size".into()).into(),
                );
            }
            write_frame(
                &mut BufWriter::new(&self.file),
                &payload,
                !(payload.len() as u64),
            )?;
            sync(&self.file)?;
        }
        self.view = plan.execute_following(&self.follower)?;
        Ok(self.follower.applied_seq())
    }

    /// Exclusive successfully applied prefix, not merely received bytes.
    pub fn applied_seq(&self) -> u64 {
        self.follower.applied_seq()
    }

    pub(crate) fn range_limits(&self) -> WalRangeLimits {
        self.limits
    }

    pub(crate) fn source_history(&self) -> [u8; 16] {
        self.history
    }

    pub fn view(&self) -> u64 {
        self.view
    }

    pub fn is_poisoned(&self) -> bool {
        self.follower.is_poisoned()
    }

    pub fn read_at<R>(&self, seq: u64, read: impl FnOnce(&World) -> R) -> Result<R, FollowerError> {
        self.follower.read_at(seq, &self.world, read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HISTORY: [u8; 16] = [7; 16];

    fn limits() -> WalRangeLimits {
        WalRangeLimits {
            max_records: 100,
            max_bytes: 1024 * 1024,
            max_control_frames: 100,
        }
    }

    fn codecs(world: &mut World) -> CodecRegistry {
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<u32>("value", world).unwrap();
        codecs.register_as::<String>("name", world).unwrap();
        codecs
    }

    fn target_codecs() -> CodecRegistry {
        let mut world = World::builder().memory_budget(1024 * 1024).build().unwrap();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<String>("name", &mut world).unwrap();
        codecs.register_as::<u32>("value", &mut world).unwrap();
        codecs
    }

    fn source() -> (tempfile::TempDir, Wal, World, CodecRegistry, Entity) {
        let dir = tempfile::tempdir().unwrap();
        let mut world = World::builder().memory_budget(1024 * 1024).build().unwrap();
        let codecs = codecs(&mut world);
        let mut wal =
            Wal::create(&dir.path().join("source"), &codecs, WalConfig::default()).unwrap();
        let entity = world.alloc_entity();
        let mut changes = EnumChangeSet::new();
        changes
            .spawn_bundle(&mut world, entity, (1u32, String::from("initial")))
            .unwrap();
        wal.append(&changes, &codecs, world.current_tick()).unwrap();
        changes.apply(&mut world).unwrap();
        wal.views.bump();
        wal.acknowledge_flush(1).unwrap();
        let mut changes = EnumChangeSet::new();
        changes.insert(&mut world, entity, String::from("updated"));
        wal.append(&changes, &codecs, world.current_tick()).unwrap();
        changes.apply(&mut world).unwrap();
        (dir, wal, world, codecs, entity)
    }

    #[test]
    fn journaled_follower_round_trip_and_restart() {
        let (dir, wal, world, source_codecs, entity) = source();
        let path = dir.path().join("follower");
        let range = wal.records_from(0, limits()).unwrap();
        let mut follower =
            JournaledFollower::create(&path, HISTORY, target_codecs(), limits()).unwrap();
        assert_eq!(follower.applied_seq(), 0);
        assert!(follower.read_at(0, |_| ()).is_err());
        assert_eq!(follower.ingest_frames(HISTORY, &range).unwrap(), 2);
        let expected = crate::world_fingerprint(&world, &source_codecs).unwrap();
        assert_eq!(
            follower
                .read_at(1, |world| crate::world_fingerprint(world, &source_codecs)
                    .unwrap())
                .unwrap(),
            expected
        );
        assert_eq!(
            follower.read_at(1, World::current_tick).unwrap(),
            world.current_tick()
        );
        drop(follower);
        let follower = JournaledFollower::open(&path, HISTORY, target_codecs(), limits()).unwrap();
        assert_eq!(follower.applied_seq(), 2);
        assert_eq!(follower.view(), 1);
        assert_eq!(
            follower
                .read_at(1, |world| world.get::<String>(entity).unwrap().clone())
                .unwrap(),
            "updated"
        );
        assert_eq!(
            follower
                .read_at(1, |world| crate::world_fingerprint(world, &source_codecs)
                    .unwrap())
                .unwrap(),
            expected
        );
    }

    #[test]
    fn journaled_follower_deduplicates_overlap_and_controls() {
        let (dir, mut wal, world, source_codecs, entity) = source();
        let path = dir.path().join("follower");
        let first = wal
            .records_from(
                0,
                WalRangeLimits {
                    max_records: 1,
                    ..limits()
                },
            )
            .unwrap();
        let mut follower =
            JournaledFollower::create(&path, HISTORY, target_codecs(), limits()).unwrap();
        assert_eq!(follower.ingest_frames(HISTORY, &first).unwrap(), 1);
        let len = follower.file.metadata().unwrap().len();
        assert_eq!(follower.ingest_frames(HISTORY, &first).unwrap(), 1);
        assert_eq!(follower.file.metadata().unwrap().len(), len);
        let overlap = wal.records_from(0, limits()).unwrap();
        assert_eq!(follower.ingest_frames(HISTORY, &overlap).unwrap(), 2);
        // Replaying old view-zero bytes after applying view one is a duplicate,
        // not a stale new mutation, and must not regress the restored fence.
        assert_eq!(follower.ingest_frames(HISTORY, &first).unwrap(), 2);
        assert_eq!(follower.view(), 1);
        wal.views.bump();
        wal.acknowledge_flush(2).unwrap();
        let tail = wal.records_from(2, limits()).unwrap();
        let len = follower.file.metadata().unwrap().len();
        assert_eq!(follower.ingest_frames(HISTORY, &tail).unwrap(), 2);
        assert!(follower.file.metadata().unwrap().len() > len);
        assert_eq!(follower.view(), 2);
        assert_eq!(follower.world.current_tick(), world.current_tick());
        drop(follower);
        let follower = JournaledFollower::open(&path, HISTORY, target_codecs(), limits()).unwrap();
        assert_eq!(follower.applied_seq(), 2);
        assert_eq!(follower.view(), 2);
        assert_eq!(follower.world.get::<String>(entity).unwrap(), "updated");
        assert_eq!(
            crate::world_fingerprint(&follower.world, &source_codecs).unwrap(),
            crate::world_fingerprint(&world, &source_codecs).unwrap()
        );
    }

    #[test]
    fn journaled_follower_refuses_invalid_input() {
        let (dir, wal, _, _, _) = source();
        for case in ["history", "crc", "gap", "schema", "bounds"] {
            let path = dir.path().join(case);
            let mut follower =
                JournaledFollower::create(&path, HISTORY, target_codecs(), limits()).unwrap();
            let len = follower.file.metadata().unwrap().len();
            let tick = follower.world.current_tick();
            let mut range = wal.records_from(0, limits()).unwrap();
            let mut history = HISTORY;
            match case {
                "history" => history[0] ^= 1,
                "crc" => {
                    *range.runs[0].frames.last_mut().unwrap() ^= 1;
                }
                "gap" => range = wal.records_from(1, limits()).unwrap(),
                "schema" => range.runs[0].schema_frame.clear(),
                "bounds" => range.next_seq += 1,
                _ => unreachable!(),
            }
            assert!(follower.ingest_frames(history, &range).is_err(), "{case}");
            assert!(follower.is_poisoned());
            assert_eq!(follower.applied_seq(), 0);
            assert_eq!(follower.world.current_tick(), tick);
            assert!(follower.world.entity_allocator_state().0.is_empty());
            assert_eq!(follower.file.metadata().unwrap().len(), len);
            assert!(matches!(
                follower.read_at(0, |_| ()),
                Err(FollowerError::Poisoned)
            ));
            assert!(
                follower
                    .ingest_frames(HISTORY, &wal.records_from(0, limits()).unwrap())
                    .is_err()
            );
        }
    }

    #[test]
    fn journaled_follower_checks_component_bytes_despite_valid_crc() {
        let dir = tempfile::tempdir().unwrap();
        let mut world = World::new();
        let mut codecs = CodecRegistry::new();
        codecs
            .register_raw_copy_as::<bool>("flag", &mut world)
            .unwrap();
        let id = world.component_id::<bool>().unwrap();
        let mut wal =
            Wal::create(&dir.path().join("source"), &codecs, WalConfig::default()).unwrap();
        wal.append(&EnumChangeSet::new(), &codecs, 1).unwrap();
        let mut range = wal.records_from(0, limits()).unwrap();
        range.runs[0].frames.clear();
        let entry = WalEntry::Mutations(crate::WalRecord {
            seq: 0,
            mutations: vec![SerializedMutation::Spawn {
                entity: Entity::from_bits(0).to_bits(),
                components: vec![(id, vec![2])],
            }],
            tick_after: 1,
        });
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry).unwrap();
        write_frame(&mut range.runs[0].frames, &payload, 0).unwrap();
        let path = dir.path().join("follower");
        let mut follower = JournaledFollower::create(&path, HISTORY, codecs, limits()).unwrap();
        assert!(matches!(
            follower.ingest_frames(HISTORY, &range),
            Err(IngestError::Follower(FollowerError::Apply(
                WalError::Codec(CodecError::Deserialize(_))
            )))
        ));
        assert_eq!(follower.applied_seq(), 0);
        assert!(follower.world.entity_allocator_state().0.is_empty());
        assert!(matches!(
            follower.read_at(0, |_| ()),
            Err(FollowerError::Poisoned)
        ));
        drop(follower);
        let mut codecs = CodecRegistry::new();
        codecs
            .register_raw_copy_as::<bool>("flag", &mut world)
            .unwrap();
        assert!(JournaledFollower::open(&path, HISTORY, codecs, limits()).is_err());
    }

    #[test]
    fn journaled_follower_failure_boundaries() {
        let (dir, wal, _, _, _) = source();
        let range = wal.records_from(0, limits()).unwrap();
        for durable in [false, true] {
            let path = dir
                .path()
                .join(if durable { "after-sync" } else { "before-sync" });
            let mut follower =
                JournaledFollower::create(&path, HISTORY, target_codecs(), limits()).unwrap();
            let old_len = follower.file.metadata().unwrap().len();
            assert!(
                follower
                    .ingest_with_sync(HISTORY, &range, |file| {
                        if durable {
                            file.sync_all()?;
                        }
                        Err(io::Error::other("interrupted before application"))
                    })
                    .is_err()
            );
            assert_eq!(follower.applied_seq(), 0);
            assert!(follower.world.entity_allocator_state().0.is_empty());
            assert!(follower.is_poisoned());
            assert!(matches!(
                follower.read_at(0, |_| ()),
                Err(FollowerError::Poisoned)
            ));
            if !durable {
                // Explicitly model discarded volatile tail bytes; merely
                // reopening would retain them in the operating system cache.
                follower.file.set_len(old_len).unwrap();
                follower.file.sync_all().unwrap();
            }
            drop(follower);
            let restored =
                JournaledFollower::open(&path, HISTORY, target_codecs(), limits()).unwrap();
            assert_eq!(restored.applied_seq(), if durable { 2 } else { 0 });
        }

        let path = dir.path().join("write-failure");
        let mut follower =
            JournaledFollower::create(&path, HISTORY, target_codecs(), limits()).unwrap();
        let writable = std::mem::replace(
            &mut follower.file,
            File::open(path.join("ingest.log")).unwrap(),
        );
        // Ensure all writes fit in the buffer: only explicit flushing can
        // report this descriptor's write error before sync/application.
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&JournalEntry {
            history: HISTORY,
            range: range.clone(),
        })
        .unwrap();
        assert!(
            FRAME_HEADER_SIZE as usize + payload.len() < BufWriter::new(&follower.file).capacity()
        );
        let tick = follower.world.current_tick();
        let mut sync_called = false;
        let result = follower.ingest_with_sync(HISTORY, &range, |_| {
            sync_called = true;
            Ok(()) // A successful sync must not hide an earlier flush error.
        });
        assert!(!sync_called);
        assert!(matches!(result, Err(IngestError::Wal(WalError::Io(_)))));
        follower.file = writable;
        assert_eq!(follower.file.metadata().unwrap().len(), JOURNAL_HEADER_SIZE);
        assert_eq!(follower.applied_seq(), 0);
        assert_eq!(follower.world.current_tick(), tick);
        assert!(follower.world.entity_allocator_state().0.is_empty());
        assert!(follower.is_poisoned());
        assert!(matches!(
            follower.read_at(0, |_| ()),
            Err(FollowerError::Poisoned)
        ));

        // A valid later record can still fail on world state after the first
        // slot applied. The complete journal is a replayable failure capture.
        let path = dir.path().join("apply-failure");
        let mut follower =
            JournaledFollower::create(&path, HISTORY, target_codecs(), limits()).unwrap();
        let first = wal
            .records_from(
                0,
                WalRangeLimits {
                    max_records: 1,
                    ..limits()
                },
            )
            .unwrap();
        let mut bad = first.clone();
        bad.next_seq = 2;
        let entry = WalEntry::Mutations(crate::WalRecord {
            seq: 1,
            mutations: vec![SerializedMutation::Despawn {
                entity: Entity::from_bits(999).to_bits(),
            }],
            tick_after: 100,
        });
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry).unwrap();
        write_frame(&mut bad.runs[0].frames, &payload, 0).unwrap();
        assert!(follower.ingest_frames(HISTORY, &bad).is_err());
        assert_eq!(follower.applied_seq(), 1);
        assert!(matches!(
            follower.read_at(0, |_| ()),
            Err(FollowerError::Poisoned)
        ));
        drop(follower);
        let capture = dir.path().join("failure-capture");
        std::fs::create_dir(&capture).unwrap();
        std::fs::copy(path.join("ingest.log"), capture.join("ingest.log")).unwrap();
        assert!(matches!(
            JournaledFollower::open(&capture, HISTORY, target_codecs(), limits()),
            Err(IngestError::Follower(FollowerError::Apply(_)))
        ));
    }

    #[test]
    fn journaled_follower_restart_reconstructs_progress() {
        let (dir, wal, world, _, _) = source();
        let range = wal.records_from(0, limits()).unwrap();
        for partial_header in [false, true] {
            let path = dir.path().join(if partial_header {
                "header-tail"
            } else {
                "payload-tail"
            });
            let mut follower =
                JournaledFollower::create(&path, HISTORY, target_codecs(), limits()).unwrap();
            follower.ingest_frames(HISTORY, &range).unwrap();
            let len = follower.file.metadata().unwrap().len();
            if partial_header {
                (&follower.file).write_all(&[1, 2, 3]).unwrap();
            } else {
                // Valid length guard, incomplete payload: an interrupted append.
                let mut bytes = Vec::new();
                write_frame(&mut bytes, &[0; 100], !100u64).unwrap();
                (&follower.file).write_all(&bytes[..20]).unwrap();
            }
            drop(follower);
            let mut follower =
                JournaledFollower::open(&path, HISTORY, target_codecs(), limits()).unwrap();
            assert_eq!(follower.applied_seq(), 2);
            assert_eq!(follower.world.current_tick(), world.current_tick());
            assert_eq!(follower.file.metadata().unwrap().len(), len);
            // Lost progress report: resending the whole durable range is safe.
            assert_eq!(follower.ingest_frames(HISTORY, &range).unwrap(), 2);
            assert_eq!(follower.file.metadata().unwrap().len(), len);
        }
    }

    #[test]
    fn journaled_follower_rejects_corrupt_history() {
        let (dir, wal, _, _, _) = source();
        let path = dir.path().join("original");
        let mut follower =
            JournaledFollower::create(&path, HISTORY, target_codecs(), limits()).unwrap();
        assert!(JournaledFollower::open(&path, HISTORY, target_codecs(), limits()).is_err());
        follower
            .ingest_frames(HISTORY, &wal.records_from(0, limits()).unwrap())
            .unwrap();
        drop(follower);
        assert!(matches!(
            JournaledFollower::open(&path, [0; 16], target_codecs(), limits()),
            Err(IngestError::HistoryMismatch)
        ));
        let original = std::fs::read(path.join("ingest.log")).unwrap();
        for offset in [0, 4, JOURNAL_HEADER_SIZE as usize, original.len() - 1] {
            let path = dir.path().join(format!("corrupt-{offset}"));
            std::fs::create_dir(&path).unwrap();
            let mut bytes = original.clone();
            bytes[offset] ^= 1;
            std::fs::write(path.join("ingest.log"), &bytes).unwrap();
            assert!(JournaledFollower::open(&path, HISTORY, target_codecs(), limits()).is_err());
            assert_eq!(std::fs::read(path.join("ingest.log")).unwrap(), bytes);
        }
        // A correctly checksummed envelope from another history also refuses.
        let path = dir.path().join("spliced-history");
        let follower =
            JournaledFollower::create(&path, HISTORY, target_codecs(), limits()).unwrap();
        let entry = JournalEntry {
            history: [9; 16],
            range: wal.records_from(0, limits()).unwrap(),
        };
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry).unwrap();
        write_frame(
            &mut BufWriter::new(&follower.file),
            &payload,
            !(payload.len() as u64),
        )
        .unwrap();
        drop(follower);
        assert!(matches!(
            JournaledFollower::open(&path, HISTORY, target_codecs(), limits()),
            Err(IngestError::HistoryMismatch)
        ));
    }
}
