use super::*;

/// Private replay plan. The exclusive borrow prevents mappings from
/// escaping to another world or surviving a change to its component registry.
/// This does not establish a finalized range for follower ingestion.
pub(super) struct ValidatedRange<'a> {
    world: &'a mut World,
    codecs: &'a CodecRegistry,
    records: Vec<(RawFrame, Arc<ApplyRemap>)>,
    last_seq: u64,
    max_view: u64,
}

impl<'a> ValidatedRange<'a> {
    /// Preflight one complete raw range for the private application gate.
    #[cfg(test)]
    pub(super) fn from_frames(
        range: &WalFrameRange,
        restored_view: u64,
        limits: WalRangeLimits,
        world: &'a mut World,
        codecs: &'a CodecRegistry,
    ) -> Result<Self, WalError> {
        Self::from_frames_after(range, restored_view, range.from_seq, limits, world, codecs)
    }

    pub(super) fn from_frames_after(
        range: &WalFrameRange,
        restored_view: u64,
        apply_from: u64,
        limits: WalRangeLimits,
        world: &'a mut World,
        codecs: &'a CodecRegistry,
    ) -> Result<Self, WalError> {
        limits.validate()?;
        if range.from_seq > apply_from {
            return Err(WalError::UnresolvedHistory {
                seq: apply_from,
                reason: "range starts beyond the applied prefix",
            });
        }
        let slots = range
            .next_seq
            .checked_sub(range.from_seq)
            .ok_or_else(|| WalError::Format("reversed raw range interval".into()))?;
        if slots > u64::try_from(limits.max_records).unwrap_or(u64::MAX) {
            return Err(WalError::Format("raw range exceeds mutation limit".into()));
        }
        if range.runs.len() > limits.max_control_frames {
            return Err(WalError::Format("raw range exceeds control limit".into()));
        }
        // Validate total response size before copying or decoding any frame.
        let mut bytes_left = limits.max_bytes;
        for run in &range.runs {
            for bytes in [&run.schema_frame, &run.frames] {
                bytes_left = bytes_left
                    .checked_sub(bytes.len())
                    .ok_or_else(|| WalError::Format("raw range exceeds byte limit".into()))?;
            }
        }
        let mut controls_left = limits.max_control_frames;
        let mut control = || -> Result<(), WalError> {
            controls_left = controls_left
                .checked_sub(1)
                .ok_or_else(|| WalError::Format("raw range exceeds control limit".into()))?;
            Ok(())
        };
        let mut expected_seq = range.from_seq;
        // Validate the source prefix on its own. A restored later fence must
        // not retroactively reject already-applied duplicate frames.
        let mut max_view = 0;
        observe_view(&mut max_view, range.seed_view);
        let mut expected_tick = world.current_tick();
        let mut records = Vec::new();
        let mut previous_start = None;
        for run in &range.runs {
            if match previous_start {
                None => run.segment_start_seq > range.from_seq,
                Some(start) => {
                    run.segment_start_seq <= start || run.segment_start_seq != expected_seq
                }
            } {
                return Err(WalError::Format(
                    "invalid raw range segment boundary".into(),
                ));
            }
            previous_start = Some(run.segment_start_seq);
            control()?;
            let mut bytes = run.schema_frame.as_slice();
            let schema_frame = RawFrame::read_bytes(&mut bytes, 0)?;
            let (entry, _, view) = schema_frame.decode()?;
            let WalEntry::Schema(schema) = entry else {
                return Err(WalError::Format("raw range lacks a schema preamble".into()));
            };
            if !bytes.is_empty() {
                return Err(WalError::Format(
                    "extra bytes after raw range schema".into(),
                ));
            }
            observe_view(&mut max_view, view);
            let remap = Arc::new(build_apply_remap(Some(&schema.components), world, codecs)?);
            let mut bytes = run.frames.as_slice();
            while !bytes.is_empty() {
                let offset = (run.frames.len() - bytes.len()) as u64;
                let frame = RawFrame::read_bytes(&mut bytes, offset)?;
                let (entry, _, view) = frame.decode()?;
                let stale = observe_view(&mut max_view, view);
                match entry {
                    WalEntry::Schema(_) => {
                        return Err(WalError::Format(
                            "schema inside raw range frame stream".into(),
                        ));
                    }
                    WalEntry::Checkpoint { .. } => control()?,
                    WalEntry::Mutations(record) => {
                        if record.seq != expected_seq || expected_seq >= range.next_seq {
                            return Err(WalError::UnresolvedHistory {
                                seq: expected_seq,
                                reason: "raw range mutation gap, duplicate, or out-of-bounds slot",
                            });
                        }
                        if stale || (record.seq >= apply_from && view < restored_view) {
                            return Err(WalError::UnresolvedHistory {
                                seq: expected_seq,
                                reason: "stale mutation has no durable terminal disposition",
                            });
                        }
                        validate_record_components(&record, &remap, codecs)?;
                        if record.seq >= apply_from {
                            expected_tick = validate_record_tick(&record, expected_tick)?;
                            records.push((frame, Arc::clone(&remap)));
                        }
                        expected_seq += 1;
                    }
                }
            }
        }
        if expected_seq != range.next_seq {
            return Err(WalError::UnresolvedHistory {
                seq: expected_seq,
                reason: "raw range is missing mutation slots",
            });
        }
        observe_view(&mut max_view, restored_view);
        Ok(Self {
            world,
            codecs,
            records,
            last_seq: range.next_seq.saturating_sub(1),
            max_view,
        })
    }

    pub(super) fn read(
        dir: &Path,
        from_seq: u64,
        world: &'a mut World,
        codecs: &'a CodecRegistry,
    ) -> Result<Self, WalError> {
        let mut records = Vec::new();
        let mut last_seq = from_seq.saturating_sub(1);
        let mut max_view_seen = 0;
        let mut expected_tick = world.current_tick();
        for (_, path) in list_segments(dir)? {
            let file = File::open(&path)?;
            validate_segment_magic(&file, &path)?;
            let mut pos = SEGMENT_MAGIC_SIZE;
            let mut remap = None;
            while let Some(frame) = RawFrame::read(&file, pos)? {
                let (entry, _proof, view) = frame.decode()?;
                pos = frame.next_offset();
                let is_stale = observe_view(&mut max_view_seen, view);
                match entry {
                    // A stale schema still binds current mutations in its run.
                    WalEntry::Schema(schema) => {
                        remap = Some(Arc::new(build_apply_remap(
                            Some(&schema.components),
                            world,
                            codecs,
                        )?));
                    }
                    WalEntry::Mutations(record) if !is_stale && record.seq >= from_seq => {
                        expected_tick = validate_record_tick(&record, expected_tick)?;
                        // Preserve legacy schema-less local recovery. The raw
                        // range constructor requires an explicit run schema.
                        let binding = match &remap {
                            Some(binding) => Arc::clone(binding),
                            None => {
                                let binding = Arc::new(build_apply_remap(None, world, codecs)?);
                                remap = Some(Arc::clone(&binding));
                                binding
                            }
                        };
                        validate_record_components(&record, &binding, codecs)?;
                        last_seq = record.seq;
                        records.push((frame, binding));
                    }
                    WalEntry::Mutations(_) | WalEntry::Checkpoint { .. } => {}
                }
            }
        }
        Ok(Self {
            world,
            codecs,
            records,
            last_seq,
            max_view: max_view_seen,
        })
    }

    pub(super) fn execute(self) -> Result<(u64, u64), WalError> {
        for (frame, remap) in self.records {
            // ponytail: decode the owned archive again; cache record descriptors
            // only if replay profiling shows this second decode matters.
            let (entry, proof, _) = frame.decode()?;
            let WalEntry::Mutations(record) = entry else {
                unreachable!("the private plan stores only mutation frames");
            };
            apply_record(&record, self.world, self.codecs, Some(&remap), Some(&proof))?;
        }
        Ok((self.last_seq, self.max_view))
    }

    pub(super) fn view(&self) -> u64 {
        self.max_view
    }

    pub(super) fn execute_following(
        self,
        follower: &crate::Follower,
    ) -> Result<u64, crate::FollowerError> {
        for (frame, remap) in self.records {
            let (entry, _, _) = frame.decode()?;
            let WalEntry::Mutations(record) = entry else {
                unreachable!("the private plan stores only mutation frames");
            };
            follower.apply_next(record.seq, || {
                // Received bytes can carry a valid CRC and invalid component
                // values (e.g. bool = 2). Require checked component decoding.
                apply_record(&record, self.world, self.codecs, Some(&remap), None)
            })?;
        }
        Ok(self.max_view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> WalRangeLimits {
        WalRangeLimits {
            max_records: 100,
            max_bytes: 1024 * 1024,
            max_control_frames: 100,
        }
    }

    fn world() -> World {
        World::builder().memory_budget(1024 * 1024).build().unwrap()
    }

    fn frame(entry: &WalEntry, view: u64) -> Vec<u8> {
        let payload = rkyv::to_bytes::<rkyv::rancor::Error>(entry).unwrap();
        let mut bytes = Vec::new();
        write_frame(&mut bytes, &payload, view).unwrap();
        bytes
    }

    #[test]
    fn raw_frame_buffer_checks_lengths_and_alignment() {
        let bytes = frame(&WalEntry::Checkpoint { flush_seq: 7 }, 3);
        let mut unaligned = vec![0];
        unaligned.extend_from_slice(&bytes);
        let mut input = &unaligned[1..];
        let frame = RawFrame::read_bytes(&mut input, 0).unwrap();
        assert!(input.is_empty());
        let (entry, _, view) = frame.decode().unwrap();
        assert!(matches!(entry, WalEntry::Checkpoint { flush_seq: 7 }));
        assert_eq!(view, 3);
        for end in 0..bytes.len() {
            assert!(RawFrame::read_bytes(&mut &bytes[..end], 0).is_err());
        }
        let mut oversized = [0; FRAME_HEADER_SIZE as usize];
        oversized[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(RawFrame::read_bytes(&mut oversized.as_slice(), 0).is_err());
    }

    fn insert(entity: Entity, component_id: usize, value: i32) -> SerializedMutation {
        SerializedMutation::Insert {
            entity: entity.to_bits(),
            component_id,
            data: rkyv::to_bytes::<rkyv::rancor::Error>(&value)
                .unwrap()
                .to_vec(),
        }
    }

    fn record(seq: u64, tick_after: u64, mutations: Vec<SerializedMutation>) -> WalEntry {
        WalEntry::Mutations(crate::record::WalRecord {
            seq,
            mutations,
            tick_after,
        })
    }

    fn fixture() -> (World, CodecRegistry, Entity, WalFrameRange) {
        let mut world = world();
        let mut codecs = CodecRegistry::new();
        codecs.register_as::<i32>("left", &mut world).unwrap();
        codecs.register_as::<u32>("right", &mut world).unwrap();
        let entity = world.spawn((1i32, 2u32));
        let schema = Wal::build_schema(&codecs);
        let mut swapped = schema.clone();
        for component in &mut swapped.components {
            component.id = 1 - component.id;
        }
        let range = WalFrameRange {
            from_seq: 0,
            next_seq: 2,
            seed_view: 2,
            runs: vec![
                WalSegmentRun {
                    segment_start_seq: 0,
                    schema_frame: frame(&WalEntry::Schema(schema), 0),
                    frames: frame(
                        &record(0, world.current_tick(), vec![insert(entity, 0, 10)]),
                        2,
                    ),
                },
                WalSegmentRun {
                    segment_start_seq: 1,
                    schema_frame: frame(&WalEntry::Schema(swapped), 1),
                    frames: [
                        frame(
                            &WalEntry::Checkpoint {
                                flush_seq: u64::MAX,
                            },
                            3,
                        ),
                        frame(
                            &record(1, world.current_tick() + 1, vec![insert(entity, 0, 20)]),
                            3,
                        ),
                        frame(
                            &WalEntry::Checkpoint {
                                flush_seq: u64::MAX,
                            },
                            4,
                        ),
                    ]
                    .concat(),
                },
            ],
        };
        (world, codecs, entity, range)
    }

    #[test]
    fn raw_plan_preserves_run_remaps_and_fences() {
        let (mut world, codecs, entity, range) = fixture();
        let tick = world.current_tick();
        let plan = ValidatedRange::from_frames(&range, 1, limits(), &mut world, &codecs).unwrap();
        assert_eq!(plan.execute().unwrap(), (1, 4));
        assert_eq!(world.get::<i32>(entity), Some(&10));
        assert_eq!(world.get::<u32>(entity), Some(&20)); // ID 0 means right in run 2.
        assert_eq!(world.current_tick(), tick + 2);
        let empty = WalFrameRange {
            from_seq: 2,
            next_seq: 2,
            seed_view: 5,
            runs: vec![],
        };
        let plan = ValidatedRange::from_frames(&empty, 9, limits(), &mut world, &codecs).unwrap();
        assert_eq!(plan.execute().unwrap(), (1, 9));
        assert_eq!(world.current_tick(), tick + 2);

        let (mut world, codecs, _, range) = fixture();
        assert!(matches!(
            ValidatedRange::from_frames(&range, 3, limits(), &mut world, &codecs),
            Err(WalError::UnresolvedHistory { seq: 0, .. })
        ));
    }

    #[test]
    fn raw_plan_rejects_invalid_range_before_effects() {
        for case in [
            "crc",
            "schema_crc",
            "no_schema",
            "wrong_schema",
            "extra_schema",
            "body_schema",
            "run_order",
            "run_gap",
            "first_run",
            "missing_slot",
            "short_end",
            "reversed",
            "gap",
            "duplicate",
            "stale",
            "component",
            "tick",
            "tick_overflow",
            "schema_name",
            "duplicate_id",
            "duplicate_name",
            "truncated",
        ] {
            let (mut world, codecs, entity, mut range) = fixture();
            let tick = world.current_tick();
            let fingerprint = crate::world_fingerprint(&world, &codecs).unwrap();
            let generations = world.entity_allocator_state().0.to_vec();
            let free = world.entity_allocator_state().1.to_vec();
            let mut schema = Wal::build_schema(&codecs);
            match case {
                "crc" => {
                    *range.runs[1].frames.last_mut().unwrap() ^= 1;
                }
                "schema_crc" => {
                    *range.runs[1].schema_frame.last_mut().unwrap() ^= 1;
                }
                "no_schema" => range.runs[1].schema_frame.clear(),
                "wrong_schema" => {
                    range.runs[1].schema_frame = frame(&WalEntry::Checkpoint { flush_seq: 0 }, 0);
                }
                "extra_schema" => range.runs[1].schema_frame.push(0),
                "body_schema" => range.runs[1].frames = frame(&WalEntry::Schema(schema), 3),
                "run_order" => range.runs[1].segment_start_seq = 0,
                "run_gap" => range.runs[1].segment_start_seq = 2,
                "first_run" => range.runs[0].segment_start_seq = 1,
                "missing_slot" => {
                    range.runs.pop();
                }
                "short_end" => range.next_seq = 1,
                "reversed" => range.from_seq = 3,
                "gap" => range.runs[1].frames = frame(&record(2, tick + 1, vec![]), 3),
                "duplicate" => range.runs[1].frames = frame(&record(0, tick + 1, vec![]), 3),
                "stale" => range.runs[1].frames = frame(&record(1, tick + 1, vec![]), 1),
                "component" => {
                    range.runs[1].frames =
                        frame(&record(1, tick + 1, vec![insert(entity, usize::MAX, 5)]), 3);
                }
                "tick" => range.runs[1].frames = frame(&record(1, tick, vec![]), 3),
                "tick_overflow" => range.runs[1].frames = frame(&record(1, u64::MAX, vec![]), 3),
                "schema_name" | "duplicate_id" | "duplicate_name" => {
                    match case {
                        "schema_name" => schema.components[0].name = "absent".into(),
                        "duplicate_id" => schema.components[1].id = schema.components[0].id,
                        _ => schema.components[1].name = schema.components[0].name.clone(),
                    }
                    range.runs[1].schema_frame = frame(&WalEntry::Schema(schema), 1);
                }
                "truncated" => {
                    range.runs[1].frames.pop();
                }
                _ => unreachable!(),
            }
            let error = ValidatedRange::from_frames(&range, 0, limits(), &mut world, &codecs)
                .err()
                .unwrap_or_else(|| panic!("accepted {case}"));
            if matches!(case, "crc" | "schema_crc") {
                assert!(
                    matches!(error, WalError::ChecksumMismatch { .. }),
                    "{case}: {error}"
                );
            }
            assert_eq!(world.current_tick(), tick, "{case}");
            assert_eq!(
                crate::world_fingerprint(&world, &codecs).unwrap(),
                fingerprint,
                "{case}"
            );
            assert_eq!(world.entity_allocator_state().0, generations, "{case}");
            assert_eq!(world.entity_allocator_state().1, free, "{case}");
        }
    }

    #[test]
    fn raw_plan_enforces_limits() {
        let (mut world, codecs, _, range) = fixture();
        let bytes = range
            .runs
            .iter()
            .map(|run| run.schema_frame.len() + run.frames.len())
            .sum();
        let exact = WalRangeLimits {
            max_records: 2,
            max_bytes: bytes,
            max_control_frames: 4,
        };
        assert!(ValidatedRange::from_frames(&range, 0, exact, &mut world, &codecs).is_ok());
        for limit in [
            WalRangeLimits {
                max_records: 1,
                ..exact
            },
            WalRangeLimits {
                max_bytes: bytes - 1,
                ..exact
            },
            WalRangeLimits {
                max_control_frames: 3,
                ..exact
            },
            WalRangeLimits {
                max_records: 0,
                ..exact
            },
            WalRangeLimits {
                max_bytes: 0,
                ..exact
            },
            WalRangeLimits {
                max_control_frames: 0,
                ..exact
            },
        ] {
            assert!(ValidatedRange::from_frames(&range, 0, limit, &mut world, &codecs).is_err());
        }
        let mut oversized = range;
        oversized.runs[1].frames = vec![0; FRAME_HEADER_SIZE as usize];
        oversized.runs[1].frames[..4].copy_from_slice(&(MAX_FRAME_SIZE as u32).to_le_bytes());
        assert!(ValidatedRange::from_frames(&oversized, 0, limits(), &mut world, &codecs).is_err());
    }

    #[test]
    fn raw_plan_propagates_partial_apply_failure() {
        let (mut world, codecs, entity, mut range) = fixture();
        let dead = Entity::from_bits(999);
        range.runs[1].frames = frame(
            &record(
                1,
                world.current_tick() + 1,
                vec![insert(entity, 0, 20), insert(dead, 0, 30)],
            ),
            3,
        );
        let plan = ValidatedRange::from_frames(&range, 0, limits(), &mut world, &codecs).unwrap();
        assert!(matches!(
            plan.execute(),
            Err(WalError::Apply(minkowski::ApplyError::PartialApply { .. }))
        ));
        assert_eq!(world.get::<i32>(entity), Some(&10));
        assert_eq!(world.get::<u32>(entity), Some(&20));
    }

    #[test]
    fn wal_frames_round_trip_divergent_live_follower() {
        let dir = tempfile::tempdir().unwrap();
        let mut leader = world();
        let mut source = CodecRegistry::new();
        source.register_as::<u32>("value", &mut leader).unwrap();
        source.register_as::<u64>("health", &mut leader).unwrap();
        source.register_as::<String>("name", &mut leader).unwrap();
        let baseline = leader.spawn((1u32, String::from("baseline")));
        let a = leader.spawn((1u64,));
        let b = leader.spawn((2u64,));
        assert!(leader.despawn(b));
        assert!(leader.despawn(a));

        let mut codec_world = world();
        let mut target = CodecRegistry::new();
        target
            .register_as::<String>("name", &mut codec_world)
            .unwrap();
        target
            .register_as::<u32>("value", &mut codec_world)
            .unwrap();
        target
            .register_as::<u64>("health", &mut codec_world)
            .unwrap();
        let mut replica = world();
        replica.register_component::<u64>();
        replica.register_component::<String>();
        replica.register_component::<u32>();
        assert_eq!(replica.spawn((1u32, String::from("baseline"))), baseline);
        assert_eq!(replica.spawn((1u64,)), a);
        assert_eq!(replica.spawn((2u64,)), b);
        assert!(replica.despawn(a));
        assert!(replica.despawn(b));
        assert_ne!(
            leader.entity_allocator_state().1,
            replica.entity_allocator_state().1
        );
        assert_ne!(leader.component_id::<u32>(), replica.component_id::<u32>());
        assert_ne!(
            codec_world.component_id::<u32>(),
            replica.component_id::<u32>()
        );

        let mut wal = Wal::create(dir.path(), &source, WalConfig::default()).unwrap();
        // Seq 0 belongs to the equivalent, nonempty baseline. Fetch starts in
        // the middle of this segment and crosses into the next one.
        let changes = EnumChangeSet::new();
        wal.append(&changes, &source, leader.current_tick())
            .unwrap();
        changes.apply(&mut leader).unwrap();
        EnumChangeSet::new().apply(&mut replica).unwrap();
        assert_eq!(
            crate::world_fingerprint(&leader, &source).unwrap(),
            crate::world_fingerprint(&replica, &target).unwrap()
        );
        assert_eq!(leader.current_tick(), replica.current_tick());
        wal.views.bump();
        wal.acknowledge_flush(1).unwrap();

        let entity = leader.alloc_entity();
        assert_eq!(entity.index(), a.index());
        let mut changes = EnumChangeSet::new();
        changes
            .spawn_bundle(&mut leader, entity, (3u32, String::from("spawned")))
            .unwrap();
        changes.insert(&mut leader, entity, 42u64);
        changes.insert(&mut leader, baseline, String::from("updated"));
        wal.append(&changes, &source, leader.current_tick())
            .unwrap();
        changes.apply(&mut leader).unwrap();
        wal.roll_segment().unwrap();
        wal.views.bump();
        wal.acknowledge_flush(2).unwrap();
        let mut changes = EnumChangeSet::new();
        changes.insert(&mut leader, entity, 9u32);
        changes.remove::<String>(&mut leader, baseline);
        wal.append(&changes, &source, leader.current_tick())
            .unwrap();
        changes.apply(&mut leader).unwrap();

        let range = wal.records_from(1, limits()).unwrap();
        assert_eq!((range.from_seq, range.next_seq, range.seed_view), (1, 3, 1));
        assert_eq!(range.runs.len(), 2);
        let plan = ValidatedRange::from_frames(&range, 0, limits(), &mut replica, &target).unwrap();
        let (last_seq, fence) = plan.execute().unwrap();
        assert_eq!((last_seq + 1, fence), (3, 2));
        assert_eq!(
            crate::world_fingerprint(&leader, &source).unwrap(),
            crate::world_fingerprint(&replica, &target).unwrap()
        );
        assert_eq!(replica.current_tick(), leader.current_tick());
        assert_eq!(replica.get::<String>(entity).unwrap(), "spawned");
        assert_eq!(replica.alloc_entity(), leader.alloc_entity());
    }
}
