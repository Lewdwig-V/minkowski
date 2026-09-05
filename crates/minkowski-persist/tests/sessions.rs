use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use minkowski::{Access, Optimistic, Transact, World};
use minkowski_persist::{
    CheckpointHandler, CodecRegistry, Durable, Fetch, JournaledFollower, LoopbackFetch, PumpError,
    Replicated, ReplicationPump, SessionError, TransportError, Wal, WalConfig, WalError,
    WalRangeLimits,
};

const HISTORY: [u8; 16] = [7; 16];

fn world() -> World {
    World::builder().memory_budget(1024 * 1024).build().unwrap()
}

fn codecs() -> CodecRegistry {
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<u32>("value", &mut world()).unwrap();
    codecs
}

fn limits() -> WalRangeLimits {
    WalRangeLimits {
        max_records: 1,
        max_bytes: 65536,
        max_control_frames: 64,
    }
}

fn durable(dir: &Path, records: usize) -> (Durable<Optimistic>, World) {
    let mut world = world();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<u32>("value", &mut world).unwrap();
    let wal = Wal::create(
        dir,
        &codecs,
        WalConfig {
            max_segment_bytes: 128,
            ..WalConfig::default()
        },
    )
    .unwrap();
    let durable = Durable::new(Optimistic::new(&world), wal, codecs);
    let access = Access::of::<(&mut u32,)>(&mut world);
    for value in 0..records as u32 {
        durable
            .transact(&mut world, &access, |tx, world| {
                tx.spawn(world, (value,));
            })
            .unwrap();
    }
    (durable, world)
}

fn follower(dir: &Path) -> JournaledFollower {
    JournaledFollower::create(dir, HISTORY, codecs(), limits()).unwrap()
}

#[test]
fn session_requests_report_only_applied_progress() {
    let dir = tempfile::tempdir().unwrap();
    let (durable, _) = durable(&dir.path().join("source"), 4);
    let source = Replicated::new(durable, HISTORY, ["a".to_owned()]).unwrap();
    let mut pump = source.join("a", follower(&dir.path().join("a"))).unwrap();
    assert_eq!(pump.pump_once().unwrap(), 1);
    assert_eq!(source.member_progress("a"), Some(0));
    let (follower, mut fetch) = pump.into_parts();
    // The source records the preceding applied prefix even if the response is lost.
    drop(fetch.fetch(1, limits()).unwrap());
    assert_eq!(follower.applied_seq(), 1);
    assert_eq!(source.member_progress("a"), Some(1));
    drop(fetch.fetch(0, limits()).unwrap()); // delayed duplicate request
    assert_eq!(source.member_progress("a"), Some(1));
    assert!(matches!(
        fetch.fetch(5, limits()),
        Err(TransportError::Source(WalError::RangeAhead { .. }))
    ));
    assert!(
        fetch
            .fetch(
                4,
                WalRangeLimits {
                    max_bytes: 0,
                    ..limits()
                }
            )
            .is_err()
    );
    assert_eq!(source.member_progress("a"), Some(1));
    let mut pump = ReplicationPump::new(follower, fetch);
    for next in 2..=4 {
        assert_eq!(pump.pump_once().unwrap(), next);
        assert_eq!(source.member_progress("a"), Some(next - 1));
    }
    assert_eq!(pump.pump_once().unwrap(), 4); // empty fetch reports final progress
    assert_eq!(source.member_progress("a"), Some(4));
}

#[test]
fn session_join_validates_follower_and_revokes_old_client() {
    let dir = tempfile::tempdir().unwrap();
    let (durable_source, _) = durable(&dir.path().join("source"), 4);
    let source = Replicated::new(durable_source, HISTORY, ["a".to_owned()]).unwrap();
    let mut old = source.join("a", follower(&dir.path().join("old"))).unwrap();
    old.pump_once().unwrap();
    old.pump_once().unwrap();
    assert_eq!(source.member_progress("a"), Some(1));

    let wrong_history =
        JournaledFollower::create(&dir.path().join("wrong"), [9; 16], codecs(), limits()).unwrap();
    assert!(matches!(
        source.join("a", wrong_history),
        Err(SessionError::HistoryMismatch)
    ));
    assert!(matches!(
        source.join("missing", follower(&dir.path().join("unknown"))),
        Err(SessionError::UnknownMember(_))
    ));
    let mut poisoned = follower(&dir.path().join("poisoned"));
    let bad = minkowski_persist::WalFrameRange {
        from_seq: 1,
        next_seq: 1,
        seed_view: 0,
        runs: vec![],
    };
    assert!(poisoned.ingest_frames(HISTORY, &bad).is_err());
    assert!(matches!(
        source.join("a", poisoned),
        Err(SessionError::PoisonedFollower)
    ));

    let (longer, _) = durable(&dir.path().join("longer"), 5);
    let mut ahead = follower(&dir.path().join("ahead"));
    for seq in 0..5 {
        let range = LoopbackFetch::new(&longer, HISTORY)
            .fetch(seq, limits())
            .unwrap();
        ahead.ingest_frames(HISTORY, &range.range).unwrap();
    }
    assert!(matches!(
        source.join("a", ahead),
        Err(SessionError::Ahead {
            applied: 5,
            published: 4
        })
    ));
    // Failed joins did not replace the existing capability or its progress.
    assert_eq!(source.member_progress("a"), Some(1));
    assert_eq!(old.pump_once().unwrap(), 3);

    let mut new = source.join("a", follower(&dir.path().join("new"))).unwrap();
    assert_eq!(source.member_progress("a"), Some(0));
    assert!(matches!(
        old.pump_once(),
        Err(PumpError::Transport(TransportError::RejoinRequired))
    ));
    assert_eq!(source.member_progress("a"), Some(0));
    new.pump_once().unwrap();
    source.remove_member("a").unwrap();
    source.add_member("a".to_owned()).unwrap();
    assert!(matches!(
        new.pump_once(),
        Err(PumpError::Transport(TransportError::RejoinRequired))
    ));
    assert_eq!(source.member_progress("a"), Some(0));
    assert!(matches!(
        source.add_member("a".to_owned()),
        Err(SessionError::DuplicateMember(_))
    ));
    assert!(matches!(
        source.remove_member("missing"),
        Err(SessionError::UnknownMember(_))
    ));
}

#[test]
fn retention_plan_respects_members_and_recovery_floor() {
    let dir = tempfile::tempdir().unwrap();
    let (durable, _) = durable(&dir.path().join("source"), 4);
    let source = Replicated::new(durable, HISTORY, ["a".to_owned(), "b".to_owned()]).unwrap();
    let mut a = source.join("a", follower(&dir.path().join("a"))).unwrap();
    for _ in 0..5 {
        a.pump_once().unwrap();
    }
    assert_eq!(source.retention_plan(4).candidate_cutoff, 0); // b has not joined
    let mut b = source.join("b", follower(&dir.path().join("b"))).unwrap();
    b.pump_once().unwrap();
    b.pump_once().unwrap();
    drop(b); // disconnect is not removal
    assert_eq!(source.retention_plan(4).follower_floor, Some(1));
    assert_eq!(source.retention_plan(4).candidate_cutoff, 1);
    assert_eq!(source.retention_plan(0).candidate_cutoff, 0); // no recovery baseline
    source.remove_member("b").unwrap();
    assert_eq!(source.retention_plan(2).candidate_cutoff, 2);
    assert_eq!(source.retention_plan(4).candidate_cutoff, 4);
    source.remove_member("a").unwrap();
    assert_eq!(source.retention_plan(2).follower_floor, None);
    assert_eq!(source.retention_plan(2).candidate_cutoff, 2);
    assert_eq!(source.retention_plan(u64::MAX).candidate_cutoff, 4);
    // The proposal never authorizes erasing required fence context.
    assert_eq!(source.retention_plan(u64::MAX).deletion_cutoff, 0);
}

struct DeletingCheckpoint(Arc<AtomicUsize>);

impl CheckpointHandler for DeletingCheckpoint {
    fn on_checkpoint_needed(
        &mut self,
        _: &mut World,
        wal: &mut Wal,
        _: &CodecRegistry,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        assert_eq!(wal.delete_segments_before(u64::MAX)?, 0);
        wal.acknowledge_flush(wal.next_seq())?;
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn replicated_checkpoint_cannot_delete_required_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let mut world = world();
    let mut codecs = CodecRegistry::new();
    codecs.register_as::<u32>("value", &mut world).unwrap();
    let wal = Wal::create(
        &dir.path().join("source"),
        &codecs,
        WalConfig {
            max_segment_bytes: 128,
            max_bytes_between_checkpoints: Some(1),
        },
    )
    .unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let durable = Durable::with_checkpoint(
        Optimistic::new(&world),
        wal,
        codecs,
        DeletingCheckpoint(Arc::clone(&calls)),
    );
    let source = Replicated::new(durable, HISTORY, ["a".to_owned()]).unwrap();
    let access = Access::of::<(&mut u32,)>(&mut world);
    for value in 0..4u32 {
        source
            .transact(&mut world, &access, |tx, world| {
                tx.spawn(world, (value,));
            })
            .unwrap();
    }
    assert_eq!(calls.load(Ordering::Relaxed), 4);
    // Multiple rollovers and checkpoint deletion attempts still permit a seq-zero join.
    assert!(
        std::fs::read_dir(dir.path().join("source"))
            .unwrap()
            .count()
            > 1
    );
    let mut pump = source.join("a", follower(&dir.path().join("a"))).unwrap();
    for _ in 0..5 {
        pump.pump_once().unwrap();
    }
    assert_eq!(pump.follower().applied_seq(), 4);
    assert_eq!(source.retention_plan(0).candidate_cutoff, 0); // checkpoint is not a baseline
    assert_eq!(source.retention_plan(4).deletion_cutoff, 0);
}

#[test]
fn session_blocked_delivery_does_not_block_commit_or_rejoin() {
    let dir = tempfile::tempdir().unwrap();
    let (durable, mut world) = durable(&dir.path().join("source"), 4);
    let source = Replicated::new(durable, HISTORY, ["a".to_owned()]).unwrap();
    let initial = source.join("a", follower(&dir.path().join("old"))).unwrap();
    let (follower, mut client) = initial.into_parts();
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let (done_tx, done_rx) = mpsc::channel();
    let replacement = self::follower(&dir.path().join("new"));
    let access = Access::of::<(&mut u32,)>(&mut world);
    std::thread::scope(|scope| {
        let worker = scope.spawn(move || {
            let mut pump = ReplicationPump::new(follower, move |seq, limits| {
                let response = client.fetch(seq, limits)?;
                ready_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok(response)
            });
            assert_eq!(pump.pump_once().unwrap(), 1);
            assert!(matches!(
                pump.pump_once(),
                Err(PumpError::Transport(TransportError::RejoinRequired))
            ));
        });
        ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        scope.spawn(|| {
            let mut replacement = source.join("a", replacement).unwrap();
            source
                .transact(&mut world, &access, |tx, world| {
                    tx.spawn(world, (99u32,));
                })
                .unwrap();
            replacement.pump_once().unwrap();
            done_tx.send(()).unwrap();
        });
        let completed = done_rx.recv_timeout(Duration::from_secs(10));
        release_tx.send(()).unwrap(); // release before any assertion can unwind
        worker.join().unwrap();
        assert!(
            completed.is_ok(),
            "delivery retained a membership or WAL lock"
        );
        assert_eq!(source.member_progress("a"), Some(0));
    });
}
