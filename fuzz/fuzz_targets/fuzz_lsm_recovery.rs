#![no_main]

use libfuzzer_sys::fuzz_target;
use minkowski::World;
use minkowski_lsm::codec::CodecRegistry;
use minkowski_lsm::manifest_log::ManifestLog;
use minkowski_lsm::manifest_ops::flush_and_record;
use minkowski_lsm::recovery::LsmRecovery;
use minkowski_lsm::types::{SeqNo, SeqRange};
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
struct FuzzPos {
    x: f32,
    y: f32,
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(_) => return,
    };
    let log_path = dir.path().join("manifest.log");

    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    if codecs.register::<FuzzPos>(&mut world).is_err() {
        return;
    }

    let count = (data[0] as usize % 20) + 1;
    let mut spawned = 0usize;
    for (i, chunk) in data[1..].chunks(4).take(count).enumerate() {
        if chunk.len() < 4 {
            break;
        }
        let bits = u32::from_le_bytes(chunk.try_into().unwrap());
        world.spawn((FuzzPos {
            x: f32::from_bits(bits),
            y: i as f32,
        },));
        spawned += 1;
    }

    // If no entities were spawned, flush returns None (no dirty pages) and
    // recovery has no baseline — flush_seq is 0, not seq_hi. There is nothing
    // to round-trip, so skip the assertions rather than false-positive.
    if spawned == 0 {
        return;
    }

    let seq_hi = (data.len() as u64).saturating_add(1);
    let (mut manifest, mut log) = match ManifestLog::recover::<4>(&log_path) {
        Ok(v) => v,
        Err(_) => return,
    };
    let flush_wrote = match flush_and_record(
        &world,
        SeqRange::new(SeqNo::from(0u64), SeqNo::from(seq_hi)).unwrap(),
        &mut manifest,
        &mut log,
        dir.path(),
        &codecs,
    ) {
        Ok(Some(_)) => true,
        Ok(None) => false,
        Err(_) => return,
    };

    // Capture the source set (bit patterns: exact + NaN-safe) before recovery.
    let mut expected: Vec<u32> = world
        .query::<(&FuzzPos,)>()
        .map(|(p,)| p.x.to_bits())
        .collect();
    expected.sort_unstable();

    let (mut result, _, _) = match LsmRecovery::recover::<4>(dir.path(), &log_path, &codecs) {
        Ok(v) => v,
        Err(_) => return,
    };

    // flush_seq matches seq_hi only when a run was actually written. A
    // spawn-heavy world with no dirty pages (impossible here since we
    // `spawned > 0` and spawn marks pages dirty) would return None; guard
    // the assertion for the case where flush returned None for any reason.
    if flush_wrote {
        assert_eq!(result.flush_seq, seq_hi);
    }

    // Recovery must reproduce the source world exactly — no dropped entities,
    // no corrupted column bytes.
    let mut got: Vec<u32> = result
        .world
        .query::<(&FuzzPos,)>()
        .map(|(p,)| p.x.to_bits())
        .collect();
    got.sort_unstable();
    assert_eq!(got, expected, "recovered FuzzPos set must equal source");
});
