use super::*;

/// Independent bounds on one raw response. Every limit must be positive.
#[derive(Clone, Copy, Debug)]
pub struct WalRangeLimits {
    /// Maximum mutation slots in the exclusive sequence interval.
    pub max_records: usize,
    /// Maximum returned frame bytes, including schema headers and payloads.
    pub max_bytes: usize,
    /// Maximum returned schema and checkpoint frames combined.
    pub max_control_frames: usize,
}

impl WalRangeLimits {
    pub(super) fn validate(self) -> Result<(), WalError> {
        if self.max_records == 0 || self.max_bytes == 0 || self.max_control_frames == 0 {
            return Err(WalError::InvalidRangeLimits);
        }
        Ok(())
    }
}

/// Detached original frames covering exactly `[from_seq, next_seq)`.
/// This is a local read result, not a transport envelope or an application ack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalFrameRange {
    pub from_seq: u64,
    pub next_seq: u64,
    /// Maximum durable view before the first returned mutation. For an empty
    /// tail response, includes all durable controls at that position.
    pub seed_view: u64,
    pub runs: Vec<WalSegmentRun>,
}

/// One segment's exact schema frame followed by original mutation/checkpoint
/// frames in source order. Buffers include frame headers, but no segment magic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WalSegmentRun {
    pub segment_start_seq: u64,
    pub schema_frame: Vec<u8>,
    pub frames: Vec<u8>,
}

impl Wal {
    /// Copy a bounded, contiguous range of synchronized WAL frames.
    ///
    /// Append and deletion require an exclusive borrow; the returned buffers
    /// can be sent after releasing this shared borrow. Sequence and byte bounds
    /// come from the same post-fsync publication, including control-only writes.
    ///
    /// Stale slots require authoritative dispositions, which are not implemented
    /// yet. Missing prefix fence context after retention also requires rejoin.
    /// Neither case is silently skipped. Limits may shorten a response; if the
    /// first mutation and its schema cannot fit, returns `RangeLimitTooSmall`.
    pub fn records_from(
        &self,
        from_seq: u64,
        limits: WalRangeLimits,
    ) -> Result<WalFrameRange, WalError> {
        limits.validate()?;
        if from_seq > self.durable_next_seq {
            return Err(WalError::RangeAhead {
                requested: from_seq,
                durable_tail: self.durable_next_seq,
            });
        }
        let (&oldest, _) = self
            .durable_ends
            .first_key_value()
            .ok_or_else(|| WalError::Format("no published WAL segment".into()))?;
        if from_seq < oldest {
            return Err(WalError::CursorBehind {
                requested: from_seq,
                oldest,
            });
        }
        if oldest != 0 {
            return Err(WalError::MissingFenceContext { oldest });
        }
        let slots = (self.durable_next_seq - from_seq)
            .min(u64::try_from(limits.max_records).unwrap_or(u64::MAX));
        let end_seq = from_seq + slots;
        let mut range = WalFrameRange {
            from_seq,
            next_seq: from_seq,
            seed_view: 0,
            runs: Vec::new(),
        };
        let mut expected_seq = 0;
        let mut max_view = 0;
        let mut bytes_left = limits.max_bytes;
        let mut controls_left = limits.max_control_frames;

        // ponytail: scan the retained prefix to reconstruct its fence; add a
        // durable seek/fence index if repeated catch-up reads become costly.
        for (&start, &end) in &self.durable_ends {
            if start != expected_seq {
                return Err(WalError::UnresolvedHistory {
                    seq: expected_seq,
                    reason: "segment boundary does not match the sequence prefix",
                });
            }
            let mut reader = File::open(self.dir.join(segment_filename(start)))?.take(end);
            let mut magic = [0; SEGMENT_MAGIC_SIZE as usize];
            reader.read_exact(&mut magic)?;
            if magic != SEGMENT_MAGIC {
                return Err(WalError::Format("invalid raw range segment magic".into()));
            }
            let mut pos = SEGMENT_MAGIC_SIZE;
            let mut schema = None;
            let mut run_started = false;
            while pos < end {
                let frame = RawFrame::read_from(&mut reader, pos)?.ok_or_else(|| {
                    WalError::Format(format!("truncated published frame at offset {pos}"))
                })?;
                pos = frame.next_offset();
                let (entry, _, view) = frame.decode()?;
                let prior_view = max_view;
                let stale = observe_view(&mut max_view, view);
                if stale
                    && (start == self.active_start_seq || matches!(entry, WalEntry::Mutations(_)))
                {
                    return Err(WalError::UnresolvedHistory {
                        seq: expected_seq,
                        reason: "stale frame has no durable terminal disposition",
                    });
                }
                if let WalEntry::Schema(_) = entry {
                    if schema.is_some() || frame.offset != SEGMENT_MAGIC_SIZE {
                        return Err(WalError::Format("misplaced range schema".into()));
                    }
                    schema = Some(frame);
                    continue;
                }
                let schema = schema.as_ref().ok_or_else(|| {
                    WalError::Format("raw range segment lacks a schema preamble".into())
                })?;
                let is_mutation = matches!(entry, WalEntry::Mutations(_));
                if let WalEntry::Mutations(record) = entry {
                    if record.seq != expected_seq || expected_seq >= self.durable_next_seq {
                        return Err(WalError::UnresolvedHistory {
                            seq: expected_seq,
                            reason: "mutation sequence gap, duplicate, or unpublished slot",
                        });
                    }
                    expected_seq += 1;
                    if record.seq < from_seq {
                        continue;
                    }
                    if range.next_seq == from_seq {
                        range.seed_view = prior_view;
                    }
                } else if range.next_seq == from_seq {
                    // Skipped prefix controls are represented by seed_view.
                    continue;
                }

                let schema_bytes = if run_started {
                    0
                } else {
                    FRAME_HEADER_SIZE as usize + schema.payload.len()
                };
                let frame_bytes = FRAME_HEADER_SIZE as usize + frame.payload.len();
                let controls = usize::from(!run_started) + usize::from(!is_mutation);
                if schema_bytes > bytes_left
                    || frame_bytes > bytes_left - schema_bytes
                    || controls > controls_left
                {
                    return if range.next_seq == from_seq {
                        Err(WalError::RangeLimitTooSmall)
                    } else {
                        Ok(range)
                    };
                }
                bytes_left -= schema_bytes + frame_bytes;
                controls_left -= controls;
                if !run_started {
                    let mut schema_frame = Vec::with_capacity(schema_bytes);
                    schema.append_to(&mut schema_frame);
                    range.runs.push(WalSegmentRun {
                        segment_start_seq: start,
                        schema_frame,
                        frames: Vec::new(),
                    });
                    run_started = true;
                }
                frame.append_to(&mut range.runs.last_mut().unwrap().frames);
                if is_mutation {
                    range.next_seq += 1;
                    if range.next_seq == end_seq {
                        return Ok(range);
                    }
                }
            }
            if schema.is_none() {
                return Err(WalError::Format(
                    "raw range segment lacks a schema preamble".into(),
                ));
            }
        }
        if expected_seq != self.durable_next_seq {
            return Err(WalError::UnresolvedHistory {
                seq: expected_seq,
                reason: "published sequence tail has missing mutation frames",
            });
        }
        range.seed_view = max_view;
        Ok(range)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    fn limits() -> WalRangeLimits {
        WalRangeLimits {
            max_records: 100,
            max_bytes: 1024 * 1024,
            max_control_frames: 100,
        }
    }

    #[test]
    fn records_from_mid_segment_includes_schema_context() {
        let dir = tempfile::tempdir().unwrap();
        let mut world = World::builder().memory_budget(1024 * 1024).build().unwrap();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<u32>("counter", &mut world).unwrap();
        let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();
        for tick in 0..3 {
            wal.append(&EnumChangeSet::new(), &codecs, tick).unwrap();
        }
        let bytes = std::fs::read(dir.path().join(segment_filename(0))).unwrap();
        let schema = RawFrame::read(&wal.active_file, SEGMENT_MAGIC_SIZE)
            .unwrap()
            .unwrap();
        let first = RawFrame::read(&wal.active_file, schema.next_offset())
            .unwrap()
            .unwrap();
        let range = wal.records_from(1, limits()).unwrap();
        assert_eq!((range.from_seq, range.next_seq, range.seed_view), (1, 3, 0));
        assert_eq!(range.runs.len(), 1);
        assert_eq!(range.runs[0].segment_start_seq, 0);
        assert_eq!(
            range.runs[0].schema_frame,
            bytes[4..schema.next_offset() as usize]
        );
        assert_eq!(range.runs[0].frames, bytes[first.next_offset() as usize..]);
    }

    // Stop at the real write/flush boundary, before the production fsync helper.
    fn write_pending(wal: &mut Wal, entry: WalEntry, view: u64) {
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(&entry).unwrap();
        let bytes = write_frame(&mut BufWriter::new(&wal.active_file), &payload, view).unwrap();
        wal.active_bytes += bytes;
        if let WalEntry::Mutations(record) = entry {
            wal.next_seq = record.seq + 1;
        }
    }

    fn mutation(seq: u64) -> WalEntry {
        WalEntry::Mutations(crate::record::WalRecord {
            seq,
            tick_after: seq,
            mutations: Vec::new(),
        })
    }

    #[test]
    fn records_from_resume_preserves_fence_context() {
        let dir = tempfile::tempdir().unwrap();
        let codecs = CodecRegistry::new();
        let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();
        write_pending(&mut wal, mutation(0), 2);
        write_pending(&mut wal, WalEntry::Checkpoint { flush_seq: 1 }, 4);
        write_pending(&mut wal, mutation(1), 5);
        write_pending(&mut wal, mutation(2), 9);
        wal.sync_active_prefix().unwrap();
        let all = wal.records_from(0, limits()).unwrap();
        assert_eq!(all.seed_view, 0); // The later view 9 cannot fence seq 0.
        let resumed = wal.records_from(1, limits()).unwrap();
        assert_eq!(resumed.seed_view, 4);
        assert_eq!(resumed.next_seq, 3);
        let mut reader = io::Cursor::new(&all.runs[0].frames);
        let mut checkpoints = 0;
        while let Some(frame) = RawFrame::read_from(&mut reader, 0).unwrap() {
            checkpoints += usize::from(matches!(
                frame.decode().unwrap().0,
                WalEntry::Checkpoint { .. }
            ));
        }
        assert_eq!(checkpoints, 1);
        let tail = wal.records_from(3, limits()).unwrap();
        assert!(tail.runs.is_empty());
        assert_eq!((tail.next_seq, tail.seed_view), (3, 9));
        // A control-only fsync advances context without allocating a sequence.
        wal.views = Views::with_current(12);
        wal.acknowledge_flush(3).unwrap();
        assert_eq!(wal.records_from(3, limits()).unwrap().seed_view, 12);
        assert_eq!(wal.records_from(1, limits()).unwrap().seed_view, 4);

        // A stale schema remains usable in a sealed run, but an active stale
        // schema is a recoverable/truncatable suffix and cannot be published.
        let dir = tempfile::tempdir().unwrap();
        let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();
        write_pending(&mut wal, mutation(0), 2);
        wal.sync_active_prefix().unwrap();
        wal.roll_segment().unwrap(); // Schema is still stamped view 0.
        write_pending(&mut wal, mutation(1), 2);
        wal.sync_active_prefix().unwrap();
        assert!(matches!(
            wal.records_from(1, limits()),
            Err(WalError::UnresolvedHistory { .. })
        ));
        wal.views = Views::with_current(2);
        wal.roll_segment().unwrap();
        let resumed = wal.records_from(1, limits()).unwrap();
        assert_eq!(resumed.seed_view, 2);
        assert_eq!(resumed.runs.len(), 1);
        assert_eq!(&resumed.runs[0].schema_frame[8..16], &0u64.to_le_bytes());
    }

    #[test]
    fn records_from_excludes_unsynced_frames() {
        let dir = tempfile::tempdir().unwrap();
        let codecs = CodecRegistry::new();
        let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();
        write_pending(&mut wal, mutation(0), 1);
        assert_eq!(wal.next_seq(), 1);
        assert_eq!(wal.records_from(0, limits()).unwrap().next_seq, 0);
        assert!(matches!(
            wal.records_from(1, limits()),
            Err(WalError::RangeAhead {
                durable_tail: 0,
                ..
            })
        ));
        // Linux fsync(/dev/null) fails with EINVAL. Verify publication is kept
        // at the old endpoint on a real sync error, not merely a pending write.
        #[cfg(target_os = "linux")]
        {
            let file = std::mem::replace(&mut wal.active_file, File::open("/dev/null").unwrap());
            assert!(matches!(wal.sync_active_prefix(), Err(WalError::Io(_))));
            wal.active_file = file;
            assert_eq!(wal.records_from(0, limits()).unwrap().next_seq, 0);
        }
        wal.sync_active_prefix().unwrap();
        assert_eq!(wal.records_from(0, limits()).unwrap().next_seq, 1);
        write_pending(&mut wal, WalEntry::Checkpoint { flush_seq: 1 }, 7);
        assert_eq!(wal.records_from(1, limits()).unwrap().seed_view, 1);
        wal.sync_active_prefix().unwrap();
        assert_eq!(wal.records_from(1, limits()).unwrap().seed_view, 7);
    }

    #[test]
    fn records_from_survives_reopen_and_rollover() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/wal");
        let codecs = CodecRegistry::new();
        let config = WalConfig {
            max_segment_bytes: 1,
            ..WalConfig::default()
        };
        let mut wal = Wal::create(&path, &codecs, config.clone()).unwrap();
        for tick in 0..3 {
            wal.views.bump();
            wal.append(&EnumChangeSet::new(), &codecs, tick).unwrap();
        }
        let range = wal.records_from(0, limits()).unwrap();
        assert_eq!(range.runs.len(), 3);
        assert_eq!(
            wal.records_from(
                0,
                WalRangeLimits {
                    max_control_frames: 1,
                    ..limits()
                }
            )
            .unwrap()
            .next_seq,
            1
        );
        for (seq, run) in range.runs.iter().enumerate() {
            assert_eq!(run.segment_start_seq, seq as u64);
            let bytes = std::fs::read(path.join(segment_filename(seq as u64))).unwrap();
            assert_eq!(
                [run.schema_frame.as_slice(), run.frames.as_slice()].concat(),
                bytes[4..]
            );
        }
        assert_eq!(wal.records_from(2, limits()).unwrap().seed_view, 2);
        let tail = wal.records_from(3, limits()).unwrap();
        assert_eq!((tail.next_seq, tail.seed_view), (3, 3));
        // Detached results survive deletion of the borrowed handle.
        drop(wal);
        let wal = Wal::open(&path, &codecs, config).unwrap();
        assert_eq!(wal.records_from(0, limits()).unwrap(), range);
        assert_eq!(wal.records_from(3, limits()).unwrap(), tail);
    }

    #[test]
    fn records_from_refuses_unresolved_history() {
        // Gaps, duplicate slots, stale mutations, and a stale active control.
        for entries in [
            vec![(mutation(1), 0)],
            vec![(mutation(0), 0), (mutation(0), 0)],
            vec![(mutation(0), 2), (mutation(1), 1)],
            vec![(mutation(0), 2), (WalEntry::Checkpoint { flush_seq: 1 }, 1)],
        ] {
            let dir = tempfile::tempdir().unwrap();
            let codecs = CodecRegistry::new();
            let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();
            for (entry, view) in entries {
                write_pending(&mut wal, entry, view);
            }
            wal.sync_active_prefix().unwrap();
            // Even a tail request must validate the skipped prefix.
            assert!(matches!(
                wal.records_from(wal.next_seq(), limits()),
                Err(WalError::UnresolvedHistory { .. })
            ));
        }

        let dir = tempfile::tempdir().unwrap();
        let codecs = CodecRegistry::new();
        let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();
        write_pending(&mut wal, mutation(0), 2);
        write_pending(&mut wal, mutation(1), 1);
        wal.sync_active_prefix().unwrap();
        wal.views = Views::with_current(2);
        wal.roll_segment().unwrap();
        assert!(matches!(
            wal.records_from(0, limits()),
            Err(WalError::UnresolvedHistory { seq: 1, .. })
        ));
    }

    #[test]
    fn records_from_rejects_corrupt_published_frames() {
        for corruption in ["torn", "crc", "magic", "missing_schema"] {
            let dir = tempfile::tempdir().unwrap();
            let codecs = CodecRegistry::new();
            let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();
            write_pending(&mut wal, mutation(0), 0);
            wal.sync_active_prefix().unwrap();
            let path = dir.path().join(segment_filename(0));
            let mut bytes = std::fs::read(&path).unwrap();
            match corruption {
                "torn" => {
                    bytes.pop();
                }
                "crc" => {
                    *bytes.last_mut().unwrap() ^= 1;
                }
                "magic" => {
                    bytes[0] ^= 1;
                }
                "missing_schema" => {
                    let schema = RawFrame::read(&wal.active_file, 4).unwrap().unwrap();
                    bytes.drain(4..schema.next_offset() as usize);
                }
                _ => unreachable!(),
            }
            std::fs::write(path, bytes).unwrap();
            let error = wal.records_from(0, limits()).unwrap_err();
            if corruption == "crc" {
                assert!(matches!(error, WalError::ChecksumMismatch { .. }));
            } else {
                assert!(
                    matches!(error, WalError::Format(_)),
                    "{corruption}: {error}"
                );
            }
        }
    }

    #[test]
    fn records_from_range_boundary_edges() {
        let dir = tempfile::tempdir().unwrap();
        let codecs = CodecRegistry::new();
        let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();
        assert_eq!(wal.records_from(0, limits()).unwrap().next_seq, 0);
        for tick in 0..3 {
            wal.append(&EnumChangeSet::new(), &codecs, tick).unwrap();
            wal.acknowledge_flush(tick + 1).unwrap();
        }
        for bounds in [
            WalRangeLimits {
                max_records: 0,
                ..limits()
            },
            WalRangeLimits {
                max_bytes: 0,
                ..limits()
            },
            WalRangeLimits {
                max_control_frames: 0,
                ..limits()
            },
        ] {
            assert!(matches!(
                wal.records_from(0, bounds),
                Err(WalError::InvalidRangeLimits)
            ));
        }
        for seq in [4, u64::MAX] {
            assert!(matches!(
                wal.records_from(seq, limits()),
                Err(WalError::RangeAhead { .. })
            ));
        }
        assert_eq!(
            wal.records_from(
                0,
                WalRangeLimits {
                    max_records: usize::MAX,
                    ..limits()
                }
            )
            .unwrap()
            .next_seq,
            3
        );
        let one = wal
            .records_from(
                0,
                WalRangeLimits {
                    max_records: 1,
                    ..limits()
                },
            )
            .unwrap();
        assert_eq!(one.next_seq, 1);
        let bytes = one.runs[0].schema_frame.len() + one.runs[0].frames.len();
        assert_eq!(
            wal.records_from(
                0,
                WalRangeLimits {
                    max_bytes: bytes,
                    ..limits()
                }
            )
            .unwrap(),
            one
        );
        assert!(matches!(
            wal.records_from(
                0,
                WalRangeLimits {
                    max_bytes: bytes - 1,
                    ..limits()
                }
            ),
            Err(WalError::RangeLimitTooSmall)
        ));
        assert_eq!(
            wal.records_from(
                0,
                WalRangeLimits {
                    max_control_frames: 1,
                    ..limits()
                }
            )
            .unwrap(),
            one
        );
        let two = wal
            .records_from(
                0,
                WalRangeLimits {
                    max_control_frames: 2,
                    ..limits()
                },
            )
            .unwrap();
        assert_eq!(two.next_seq, 2);
        assert_eq!(
            wal.records_from(two.next_seq, limits()).unwrap().next_seq,
            3
        );
        assert!(wal.records_from(3, limits()).unwrap().runs.is_empty());
    }

    #[test]
    fn records_from_survives_failed_rollover_sync() {
        for fail_at in [1, 2] {
            let dir = tempfile::tempdir().unwrap();
            let codecs = CodecRegistry::new();
            let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();
            let changeset = EnumChangeSet::new();
            wal.append(&changeset, &codecs, 0).unwrap();
            let mut syncs = 0;
            assert!(
                wal.roll_segment_with_sync(|file| {
                    syncs += 1;
                    if syncs == fail_at {
                        Err(io::Error::other("injected rollover sync failure"))
                    } else {
                        file.sync_all()
                    }
                })
                .is_err()
            );
            assert_eq!(wal.active_start_seq, 0);
            assert!(!dir.path().join(segment_filename(1)).exists());
            // append treats rollover failure as nonfatal. Continue in the old
            // segment, then roll at a later sequence and reopen the directory.
            wal.config.max_segment_bytes = 1;
            wal.append(&changeset, &codecs, 1).unwrap();
            let expected = wal.records_from(0, limits()).unwrap();
            assert_eq!(expected.next_seq, 2);
            drop(wal);
            let wal = Wal::open(dir.path(), &codecs, WalConfig::default()).unwrap();
            assert_eq!(wal.records_from(0, limits()).unwrap(), expected);
        }
    }

    #[test]
    #[cfg(unix)]
    fn failed_rollover_cleanup_blocks_writes() {
        let dir = tempfile::tempdir().unwrap();
        let codecs = CodecRegistry::new();
        let mut wal = Wal::create(dir.path(), &codecs, WalConfig::default()).unwrap();
        let changeset = EnumChangeSet::new();
        wal.append(&changeset, &codecs, 0).unwrap();
        let pending = dir.path().join(segment_filename(1));
        assert!(
            wal.roll_segment_with_sync(|_| {
                // A directory in place of the failed file makes unlink fail even
                // when tests run as root. Unix permits unlinking the open file.
                std::fs::remove_file(&pending)?;
                std::fs::create_dir(&pending)?;
                Err(io::Error::other("injected rollover sync failure"))
            })
            .is_err()
        );
        let original = std::fs::read(dir.path().join(segment_filename(0))).unwrap();
        assert!(wal.append(&changeset, &codecs, 1).is_err());
        assert!(wal.acknowledge_flush(1).is_err());
        assert!(wal.delete_segments_before(1).is_err());
        assert_eq!(wal.next_seq(), 1);
        assert_eq!(
            std::fs::read(dir.path().join(segment_filename(0))).unwrap(),
            original
        );

        std::fs::remove_dir(&pending).unwrap();
        wal.config.max_segment_bytes = 1;
        wal.append(&changeset, &codecs, 1).unwrap();
        drop(wal);
        let wal = Wal::open(dir.path(), &codecs, WalConfig::default()).unwrap();
        assert_eq!(wal.records_from(0, limits()).unwrap().next_seq, 2);
    }

    #[test]
    fn records_from_requires_retained_fence_context() {
        let dir = tempfile::tempdir().unwrap();
        let codecs = CodecRegistry::new();
        let config = WalConfig {
            max_segment_bytes: 1,
            ..WalConfig::default()
        };
        let mut wal = Wal::create(dir.path(), &codecs, config.clone()).unwrap();
        for tick in 0..3 {
            wal.append(&EnumChangeSet::new(), &codecs, tick).unwrap();
        }
        assert_eq!(wal.delete_segments_before(1).unwrap(), 1);
        for wal in [wal, Wal::open(dir.path(), &codecs, config).unwrap()] {
            assert!(matches!(
                wal.records_from(0, limits()),
                Err(WalError::CursorBehind { oldest: 1, .. })
            ));
            for seq in [1, 3] {
                assert!(matches!(
                    wal.records_from(seq, limits()),
                    Err(WalError::MissingFenceContext { oldest: 1 })
                ));
            }
        }
    }
}
