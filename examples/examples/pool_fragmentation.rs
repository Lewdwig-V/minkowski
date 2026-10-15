//! Pool fragmentation: how much of a fixed memory budget is usable before
//! `PoolExhausted`, under column-growth patterns that stress the allocator.
//!
//! Each scenario builds a World with a fixed budget, then spawns entities
//! (round-robin across archetypes) until every archetype refuses to grow.
//! Reported per scenario:
//!   - `first_fail`: pool utilization at the first `PoolExhausted`
//!   - `final`:      pool utilization once every archetype has failed
//!   - `entities`:   total entities placed
//!
//! A final section measures resident memory (RSS) right after `World::new()`
//! for the default demand-paged pool, which shows how much of the pool the
//! allocator touches before the application allocates anything.
//!
//! Run: cargo run -p minkowski-examples --example pool_fragmentation --release

use minkowski::{HugePages, World};

const BUDGET: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Tag<const I: usize>;

#[derive(Clone, Copy)]
struct Payload<const N: usize>([u8; N]);

/// Spawn one entity into archetype `arch` (0..64): 16 tags × 4 payload sizes.
fn spawn_into(world: &mut World, arch: usize) -> bool {
    macro_rules! by_tag {
        ($p:expr) => {
            match arch % 16 {
                0 => world.try_spawn((Tag::<0>, $p)).is_ok(),
                1 => world.try_spawn((Tag::<1>, $p)).is_ok(),
                2 => world.try_spawn((Tag::<2>, $p)).is_ok(),
                3 => world.try_spawn((Tag::<3>, $p)).is_ok(),
                4 => world.try_spawn((Tag::<4>, $p)).is_ok(),
                5 => world.try_spawn((Tag::<5>, $p)).is_ok(),
                6 => world.try_spawn((Tag::<6>, $p)).is_ok(),
                7 => world.try_spawn((Tag::<7>, $p)).is_ok(),
                8 => world.try_spawn((Tag::<8>, $p)).is_ok(),
                9 => world.try_spawn((Tag::<9>, $p)).is_ok(),
                10 => world.try_spawn((Tag::<10>, $p)).is_ok(),
                11 => world.try_spawn((Tag::<11>, $p)).is_ok(),
                12 => world.try_spawn((Tag::<12>, $p)).is_ok(),
                13 => world.try_spawn((Tag::<13>, $p)).is_ok(),
                14 => world.try_spawn((Tag::<14>, $p)).is_ok(),
                _ => world.try_spawn((Tag::<15>, $p)).is_ok(),
            }
        };
    }
    match arch / 16 {
        0 => by_tag!(Payload::<8>([0; 8])),
        1 => by_tag!(Payload::<16>([0; 16])),
        2 => by_tag!(Payload::<64>([0; 64])),
        _ => by_tag!(Payload::<256>([0; 256])),
    }
}

struct Outcome {
    first_fail: f64,
    final_util: f64,
    entities: u64,
}

/// Round-robin across `archetypes`, `weight(arch)` spawns per turn, until
/// every archetype has hit `PoolExhausted` once.
fn run(archetypes: &[usize], weight: impl Fn(usize) -> usize) -> Outcome {
    let mut world = World::builder()
        .memory_budget(BUDGET)
        .hugepages(HugePages::Off)
        .build()
        .expect("failed to allocate memory pool");
    let mut alive = vec![true; archetypes.len()];
    let mut first_fail = None;
    let mut entities = 0u64;
    while alive.iter().any(|a| *a) {
        for (slot, &arch) in archetypes.iter().enumerate() {
            if !alive[slot] {
                continue;
            }
            for _ in 0..weight(arch) {
                if spawn_into(&mut world, arch) {
                    entities += 1;
                } else {
                    alive[slot] = false;
                    if first_fail.is_none() {
                        world.flush_pool_caches();
                        let s = world.stats();
                        first_fail = Some(s.pool_used as f64 / s.pool_capacity as f64);
                    }
                    break;
                }
            }
        }
    }
    world.flush_pool_caches();
    let s = world.stats();
    Outcome {
        first_fail: first_fail.unwrap_or(1.0),
        final_util: s.pool_used as f64 / s.pool_capacity as f64,
        entities,
    }
}

fn report(name: &str, o: &Outcome) {
    println!(
        "{name:<28} first_fail {:>5.1}%   final {:>5.1}%   entities {:>9}",
        o.first_fail * 100.0,
        o.final_util * 100.0,
        o.entities
    );
}

#[cfg(target_os = "linux")]
fn rss_bytes() -> Option<usize> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(resident * 4096)
}

#[cfg(not(target_os = "linux"))]
fn rss_bytes() -> Option<usize> {
    None
}

fn main() {
    println!(
        "--- Usable budget before PoolExhausted ({} MiB pool) ---",
        BUDGET >> 20
    );

    // Many small archetypes with small payloads: demand concentrates in the
    // small and medium size classes.
    let small: Vec<usize> = (0..32).collect();
    report("small columns (32 arch)", &run(&small, |_| 1));

    // A single wide column: one archetype whose column must grow past 1 MiB.
    report("one big column (256 B)", &run(&[48], |_| 1));

    // Mixed: a few heavy archetypes plus many light ones.
    let mixed: Vec<usize> = (0..64).collect();
    report(
        "mixed (4 heavy, 60 light)",
        &run(&mixed, |a| if a % 16 == 0 { 64 } else { 1 }),
    );

    // All 64 archetypes with equal weight.
    report("uniform (64 arch)", &run(&mixed, |_| 1));

    println!("\n--- Resident memory after World::new() (256 MiB demand-paged pool) ---");
    match (rss_bytes(), {
        let w = World::new();
        let after = rss_bytes();
        std::hint::black_box(&w);
        after
    }) {
        (Some(before), Some(after)) => println!(
            "RSS delta: {:.1} MiB",
            after.saturating_sub(before) as f64 / 1_048_576.0
        ),
        _ => println!("RSS not available on this platform"),
    }
}
