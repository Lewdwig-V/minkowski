use std::sync::mpsc;
use std::time::Duration;

use minkowski::{Access, Entity, EnumChangeSet, Optimistic, Transact, World};
use minkowski_persist::{
    CodecRegistry, Durable, Fetch, FetchResponse, FollowerError, JournaledFollower, LoopbackFetch,
    PumpError, RecordingFetch, ReplicationPump, TransportError, Wal, WalConfig, WalError,
    WalRangeLimits, world_fingerprint,
};

const HISTORY: [u8; 16] = [7; 16];

fn limits() -> WalRangeLimits {
    WalRangeLimits {
        max_records: 1,
        max_bytes: 64 * 1024,
        max_control_frames: 64,
    }
}

fn world() -> World {
    World::builder().memory_budget(1024 * 1024).build().unwrap()
}

fn target_codecs() -> CodecRegistry {
    let mut world = world();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<String>("name", &mut world).unwrap();
    codecs.register_as::<u32>("value", &mut world).unwrap();
    codecs
}

fn source() -> (tempfile::TempDir, Durable<Optimistic>, World, Entity) {
    let dir = tempfile::tempdir().unwrap();
    let mut world = world();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<u32>("value", &mut world).unwrap();
    codecs.register_as::<String>("name", &mut world).unwrap();
    let mut wal = Wal::create(
        &dir.path().join("source"),
        &codecs,
        WalConfig {
            max_segment_bytes: 512,
            ..WalConfig::default()
        },
    )
    .unwrap();
    let entity = world.alloc_entity();
    for value in 0..4u32 {
        let mut changes = EnumChangeSet::new();
        if value == 0 {
            changes
                .spawn_bundle(&mut world, entity, (value, String::from("first")))
                .unwrap();
        } else {
            changes.insert(&mut world, entity, value);
            changes.insert(&mut world, entity, format!("value-{value}"));
        }
        wal.append(&changes, &codecs, world.current_tick()).unwrap();
        changes.apply(&mut world).unwrap();
    }
    // The final empty fetch must ingest this fence without claiming more slots.
    wal.views.bump();
    wal.acknowledge_flush(4).unwrap();
    let strategy = Optimistic::new(&world);
    (dir, Durable::new(strategy, wal, codecs), world, entity)
}

#[test]
fn pull_pump_reports_only_ingested_progress() {
    let (dir, source, world, entity) = source();
    let follower = JournaledFollower::create(
        &dir.path().join("follower"),
        HISTORY,
        target_codecs(),
        limits(),
    )
    .unwrap();
    let fetch = RecordingFetch::new(LoopbackFetch::new(&source, HISTORY));
    let mut pump = ReplicationPump::new(follower, fetch);
    for next in 1..=4 {
        assert_eq!(pump.pump_once().unwrap(), next);
        assert_eq!(pump.fetch().requests().last().unwrap().from_seq, next - 1);
        assert_eq!(pump.follower().applied_seq(), next);
    }
    assert_eq!(pump.follower().view(), 0);
    assert_eq!(pump.pump_once().unwrap(), 4);
    assert_eq!(pump.follower().view(), 1);
    assert_eq!(
        pump.fetch()
            .requests()
            .iter()
            .map(|r| r.from_seq)
            .collect::<Vec<_>>(),
        [0, 1, 2, 3, 4]
    );
    for request in pump.fetch().requests() {
        assert_eq!(request.limits.max_records, limits().max_records);
        assert_eq!(request.limits.max_bytes, limits().max_bytes);
        assert_eq!(
            request.limits.max_control_frames,
            limits().max_control_frames
        );
    }
    let codecs = target_codecs();
    assert_eq!(
        pump.follower()
            .read_at(3, |w| world_fingerprint(w, &codecs).unwrap())
            .unwrap(),
        world_fingerprint(&world, &codecs).unwrap()
    );
    assert_eq!(
        pump.follower().read_at(3, World::current_tick).unwrap(),
        world.current_tick()
    );
    assert_eq!(
        pump.follower()
            .read_at(3, |w| *w.get::<u32>(entity).unwrap())
            .unwrap(),
        3
    );
    let (_, mut fetch) = pump.into_parts();
    assert_eq!(fetch.take_requests().len(), 5);
    assert!(fetch.requests().is_empty());
}

#[test]
fn pull_pump_restart_reports_reconstructed_progress() {
    let (dir, source, world, _) = source();
    for applied in [1, 4] {
        let path = dir.path().join(format!("restart-{applied}"));
        let follower =
            JournaledFollower::create(&path, HISTORY, target_codecs(), limits()).unwrap();
        let mut pump = ReplicationPump::new(follower, LoopbackFetch::new(&source, HISTORY));
        for _ in 0..applied {
            pump.pump_once().unwrap();
        }
        // Crash after durable application, before the next request reports it.
        drop(pump);
        let follower = JournaledFollower::open(&path, HISTORY, target_codecs(), limits()).unwrap();
        let mut pump = ReplicationPump::new(
            follower,
            RecordingFetch::new(LoopbackFetch::new(&source, HISTORY)),
        );
        assert_eq!(pump.pump_once().unwrap(), (applied + 1).min(4));
        assert_eq!(pump.fetch().requests()[0].from_seq, applied);
        while pump.follower().applied_seq() < 4 {
            pump.pump_once().unwrap();
        }
        assert_eq!(pump.pump_once().unwrap(), 4);
        assert_eq!(pump.follower().view(), 1);
        assert_eq!(
            pump.follower().read_at(3, World::current_tick).unwrap(),
            world.current_tick()
        );
    }
}

#[test]
fn pull_pump_stops_on_terminal_errors() {
    let (dir, source, _, _) = source();
    for case in ["history", "corrupt", "gap", "source", "rejoin"] {
        let follower =
            JournaledFollower::create(&dir.path().join(case), HISTORY, target_codecs(), limits())
                .unwrap();
        let fetch = RecordingFetch::new(|seq, limits| {
            if case == "source" {
                return Err(TransportError::Source(WalError::MissingFenceContext {
                    oldest: 1,
                }));
            }
            if case == "rejoin" {
                return Err(TransportError::RejoinRequired);
            }
            let mut response = LoopbackFetch::new(&source, HISTORY).fetch(seq, limits)?;
            match case {
                "history" => response.history[0] ^= 1,
                "corrupt" => *response.range.runs[0].frames.last_mut().unwrap() ^= 1,
                "gap" => response.range = source.records_from(seq + 1, limits)?,
                _ => unreachable!(),
            }
            Ok(response)
        });
        let mut pump = ReplicationPump::new(follower, fetch);
        let error = pump.pump_once().unwrap_err();
        match case {
            "source" | "rejoin" => assert!(matches!(error, PumpError::Transport(_))),
            _ => {
                assert!(matches!(error, PumpError::Ingest(_)));
                assert!(matches!(
                    pump.follower().read_at(0, |_| ()),
                    Err(FollowerError::Poisoned)
                ));
            }
        }
        assert_eq!(pump.follower().applied_seq(), 0);
        assert!(matches!(pump.pump_once(), Err(PumpError::Stopped)));
        assert_eq!(pump.fetch().requests().len(), 1);
        if pump.follower().is_poisoned() {
            let (follower, fetch) = pump.into_parts();
            let mut replacement = ReplicationPump::new(follower, fetch);
            assert!(matches!(replacement.pump_once(), Err(PumpError::Stopped)));
            assert_eq!(replacement.fetch().requests().len(), 1);
        }
    }
}

#[test]
fn pull_pump_never_reports_partial_apply() {
    let (dir, source, world, _) = source();
    drop(source);
    let codecs = target_codecs();
    let mut wal = Wal::open(&dir.path().join("source"), &codecs, WalConfig::default()).unwrap();
    let mut bad = EnumChangeSet::new();
    bad.record_despawn(Entity::from_bits(999));
    wal.append(&bad, &codecs, world.current_tick()).unwrap();
    let source = Durable::new(Optimistic::new(&world), wal, codecs);
    let follower = JournaledFollower::create(
        &dir.path().join("partial"),
        HISTORY,
        target_codecs(),
        WalRangeLimits {
            max_records: 10,
            ..limits()
        },
    )
    .unwrap();
    let mut pump = ReplicationPump::new(
        follower,
        RecordingFetch::new(LoopbackFetch::new(&source, HISTORY)),
    );
    assert!(matches!(pump.pump_once(), Err(PumpError::Ingest(_))));
    assert_eq!(pump.follower().applied_seq(), 4);
    assert!(matches!(
        pump.follower().read_at(3, |_| ()),
        Err(FollowerError::Poisoned)
    ));
    assert!(matches!(pump.pump_once(), Err(PumpError::Stopped)));
    assert_eq!(pump.fetch().requests().len(), 1);
    assert_eq!(pump.fetch().requests()[0].from_seq, 0);
}

#[test]
fn pull_fetch_chaos_converges_pinned_seeds() {
    let (dir, source, world, _) = source();
    let codecs = target_codecs();
    for seed in [1u64, 7, 42] {
        let follower = JournaledFollower::create(
            &dir.path().join(format!("chaos-{seed}")),
            HISTORY,
            target_codecs(),
            limits(),
        )
        .unwrap();
        let mut rng = seed;
        let mut calls = 0;
        let mut cached: Option<FetchResponse> = None;
        let mut faults = [0; 6];
        let mut loopback = LoopbackFetch::new(&source, HISTORY);
        let fetch = RecordingFetch::new(|seq, limits| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let action = if calls < 64 {
                ((rng >> 32) % 6) as usize
            } else {
                5
            };
            calls += 1;
            faults[action] += 1;
            match action {
                0 => Err(TransportError::Down),
                1 => Err(TransportError::Lost), // request lost
                2 => {
                    loopback.fetch(seq, limits)?;
                    Err(TransportError::Lost) // response lost after source read
                }
                3 if cached.is_some() => Ok(cached.as_ref().unwrap().clone()), // delayed duplicate
                4 => {
                    loopback.fetch(seq, limits)?; // duplicate request
                    loopback.fetch(seq, limits)
                }
                _ => {
                    let response = loopback.fetch(seq, limits)?;
                    cached = Some(response.clone());
                    Ok(response)
                }
            }
        });
        let mut pump = ReplicationPump::new(follower, fetch);
        // Fixed fault prefix followed by a repaired link, without wall-clock sleeps.
        for _ in 0..70 {
            let before = pump.follower().applied_seq();
            match pump.pump_once() {
                Ok(after) => assert!(after >= before && after <= 4),
                Err(PumpError::Transport(TransportError::Lost | TransportError::Down)) => {
                    assert_eq!(pump.follower().applied_seq(), before);
                    assert!(!pump.follower().is_poisoned());
                }
                other => panic!("seed {seed}: {other:?}"),
            }
            assert_eq!(pump.fetch().requests().last().unwrap().from_seq, before);
        }
        assert_eq!(pump.follower().applied_seq(), 4);
        assert_eq!(pump.fetch().requests().last().unwrap().from_seq, 4);
        assert_eq!(pump.follower().view(), 1);
        assert_eq!(
            pump.follower()
                .read_at(3, |w| world_fingerprint(w, &codecs).unwrap())
                .unwrap(),
            world_fingerprint(&world, &codecs).unwrap()
        );
        assert_eq!(
            pump.follower().read_at(3, World::current_tick).unwrap(),
            world.current_tick()
        );
        drop(pump);
        assert!(
            faults.iter().all(|count| *count > 0),
            "seed {seed}: {faults:?}"
        );
    }
}

#[test]
fn pull_blocked_delivery_does_not_hold_wal_lock() {
    let (dir, source, mut world, entity) = source();
    let follower = JournaledFollower::create(
        &dir.path().join("blocked"),
        HISTORY,
        target_codecs(),
        limits(),
    )
    .unwrap();
    let access = Access::of::<(&mut u32,)>(&mut world);
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (commit_tx, commit_rx) = mpsc::channel();
    std::thread::scope(|scope| {
        let source = &source;
        let fetch = move |seq, limits| {
            let response = LoopbackFetch::new(source, HISTORY).fetch(seq, limits)?;
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(response)
        };
        let worker = scope.spawn(move || {
            let mut pump = ReplicationPump::new(follower, fetch);
            assert_eq!(pump.pump_once().unwrap(), 1);
            pump
        });
        ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        scope.spawn(|| {
            source
                .transact(&mut world, &access, |tx, world| {
                    tx.write(world, entity, 99u32);
                })
                .unwrap();
            commit_tx.send(()).unwrap();
        });
        let completed = commit_rx.recv_timeout(Duration::from_secs(10));
        // Release even on timeout so a failed assertion cannot strand a thread.
        release_tx.send(()).unwrap();
        let pump = worker.join().unwrap();
        assert!(completed.is_ok(), "delivery held the source WAL lock");
        assert_eq!(source.wal_seq(), 5);
        assert_eq!(
            pump.follower()
                .read_at(0, |w| *w.get::<u32>(entity).unwrap())
                .unwrap(),
            0
        );
    });
}
