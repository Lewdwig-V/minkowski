//! Memory footprint: pool (component storage) vs system heap (everything else).
//!
//! The pool budget covers component storage only, on the assumption that
//! component bytes dwarf the engine's heap-side bookkeeping. This harness
//! measures that assumption. A counting global allocator tracks exact live
//! and peak heap bytes; `WorldStats::pool_used` gives pool bytes. Each
//! scenario reports both and the heap/pool ratio.
//!
//! Heap figures exclude the world's own fixed costs at construction (the
//! pool side table, 1/64 of the budget), which are reported once up front.
//!
//! Run: cargo run -p minkowski-examples --example memory_footprint --release

// Byte counts are converted to f64/isize only for printing and signed deltas
// (well under 2^52 / isize::MAX in practice).
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use minkowski::{
    Access, BTreeIndex, EnumChangeSet, HashIndex, HugePages, JoinKind, Optimistic, QueryPlanner,
    SpatialIndex, Transact, World,
};

// ── Counting allocator ──────────────────────────────────────────────

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(n: usize) {
    let live = LIVE.fetch_add(n, Ordering::Relaxed) + n;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: forwards every call to `System` unchanged; only adds counters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            add(layout.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            add(layout.size());
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            add(new_size);
        }
        p
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn live() -> usize {
    LIVE.load(Ordering::Relaxed)
}

fn reset_peak() {
    PEAK.store(live(), Ordering::Relaxed);
}

fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}

// ── Components ──────────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Payload<const N: usize>([u8; N]);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Score(u32);

#[derive(Clone, Copy)]
struct Team(#[allow(dead_code)] u32);

#[derive(Clone, Copy)]
struct Pos(#[allow(dead_code)] [f32; 3]);

#[derive(Clone, Copy)]
struct SparseVal(#[allow(dead_code)] u64);

#[derive(Clone, Copy)]
struct Tag<const I: usize>;

// ── Reporting ───────────────────────────────────────────────────────

const BUDGET: usize = 1 << 30; // 1 GiB

/// Fresh world plus the heap baseline to measure against.
///
/// The thread-local pool cache holds an `Arc` to the last pool this thread
/// allocated from, so a dropped world's pool (and its heap side table) is
/// only released when this thread first allocates from the new pool. Do
/// that allocation here, before taking the baseline.
fn world() -> (World, isize) {
    let mut w = World::builder()
        .memory_budget(BUDGET)
        .hugepages(HugePages::Off)
        .build()
        .expect("pool");
    let e = w.spawn((Warm(0),));
    w.despawn(e);
    let base = live_i();
    (w, base)
}

fn live_i() -> isize {
    live() as isize
}

#[derive(Clone, Copy)]
struct Warm(#[allow(dead_code)] u8);

fn mib(b: usize) -> f64 {
    b as f64 / 1_048_576.0
}

fn row(name: &str, pool: usize, heap: isize) {
    let ratio = if pool == 0 {
        f64::INFINITY
    } else {
        heap as f64 / pool as f64
    };
    println!(
        "{name:<46} pool {:>9.2} MiB   heap {:>9.2} MiB   heap/pool {:>6.2}",
        mib(pool),
        heap as f64 / 1_048_576.0,
        ratio
    );
}

fn pool_used(w: &mut World) -> usize {
    w.flush_pool_caches();
    w.stats().pool_used
}

// ── Scenarios ───────────────────────────────────────────────────────

fn dense<const N: usize>(n: usize) {
    let (mut w, base) = world();
    w.spawn_batch((0..n).map(|i| (Payload::<N>([i as u8; N]),)));
    let pool = pool_used(&mut w);
    row(
        &format!("dense: {n} entities x {N} B"),
        pool,
        live_i() - base,
    );
}

fn archetypes(per_mask: usize) {
    let (mut w, base) = world();
    macro_rules! tag {
        ($e:expr, $bit:expr) => {
            match $bit {
                0 => w.insert($e, (Tag::<0>,)),
                1 => w.insert($e, (Tag::<1>,)),
                2 => w.insert($e, (Tag::<2>,)),
                3 => w.insert($e, (Tag::<3>,)),
                4 => w.insert($e, (Tag::<4>,)),
                5 => w.insert($e, (Tag::<5>,)),
                6 => w.insert($e, (Tag::<6>,)),
                7 => w.insert($e, (Tag::<7>,)),
                8 => w.insert($e, (Tag::<8>,)),
                9 => w.insert($e, (Tag::<9>,)),
                10 => w.insert($e, (Tag::<10>,)),
                _ => w.insert($e, (Tag::<11>,)),
            }
            .expect("live entity")
        };
    }
    for mask in 0u32..4096 {
        for _ in 0..per_mask {
            let e = w.spawn((Payload::<64>([0; 64]),));
            for bit in 0..12 {
                if mask & (1 << bit) != 0 {
                    tag!(e, bit);
                }
            }
        }
    }
    let pool = pool_used(&mut w);
    row(
        &format!("archetypes: 4096 x {per_mask} entities x 64 B"),
        pool,
        live_i() - base,
    );
}

fn sparse(n: usize) {
    let (mut w, base) = world();
    let es = w.spawn_batch((0..n).map(|_| (Payload::<8>([0; 8]),)));
    for e in es.iter().copied() {
        w.insert_sparse(e, SparseVal(e.to_bits()));
    }
    drop(es);
    let pool = pool_used(&mut w);
    row(
        &format!("sparse: {n} entities x 8 B + sparse u64"),
        pool,
        live_i() - base,
    );
}

fn indexes(n: usize) {
    let (mut w, base) = world();
    w.spawn_batch((0..n).map(|i| (Score(i as u32),)));
    let pool = pool_used(&mut w);
    let world_heap = live_i() - base;
    let before = live_i();
    let mut bt = BTreeIndex::<Score>::new();
    bt.rebuild(&mut w);
    let bt_heap = live_i() - before;
    let before = live_i();
    let mut hi = HashIndex::<Score>::new();
    hi.rebuild(&mut w);
    let hi_heap = live_i() - before;
    row(
        &format!("index world: {n} x Score(u32), no index"),
        pool,
        world_heap,
    );
    row("  + BTreeIndex<Score>", pool, bt_heap);
    row("  + HashIndex<Score>", pool, hi_heap);
    black_box((&bt, &hi));
}

fn join(n: usize) {
    let (mut w, base) = world();
    w.spawn_batch((0..n).map(|i| (Score(i as u32), Pos([0.0; 3]), Team(i as u32 % 8))));
    let pool = pool_used(&mut w);
    let world_heap = live_i() - base;
    let before = live_i();
    let mut plan = {
        let planner = QueryPlanner::new(&w);
        planner
            .scan::<(&Score, &Pos)>()
            .join::<(&Team,)>(JoinKind::Inner)
            .build()
    };
    let built = live_i() - before;
    reset_peak();
    let start_live = live();
    let len = plan.execute_collect(&mut w).expect("plan").len();
    let first_peak = (peak() - start_live) as isize;
    let retained = live_i() - start_live as isize + built;
    reset_peak();
    let second_start = live();
    let len2 = plan.execute_collect(&mut w).expect("plan").len();
    let second_peak = (peak() - second_start) as isize;
    assert_eq!(len, n);
    assert_eq!(len2, n);
    row(
        &format!("join world: {n} x (Score, Pos, Team)"),
        pool,
        world_heap,
    );
    row("  join scratch retained by plan", pool, retained);
    row("  join: peak extra heap, 1st execution", pool, first_peak);
    row("  join: peak extra heap, 2nd execution", pool, second_peak);
}

fn changesets(n: usize) {
    let (mut w, base) = world();
    let es = w.spawn_batch((0..n).map(|_| (Payload::<16>([0; 16]),)));
    let pool = pool_used(&mut w);
    let world_heap = live_i() - base;

    // Cost of constructing an empty changeset (Tx::new does this per tx).
    let before = live_i();
    let cs = EnumChangeSet::new();
    let empty_heap = live_i() - before;
    drop(cs);
    let iters = 200;
    let t = Instant::now();
    for _ in 0..iters {
        black_box(EnumChangeSet::new());
    }
    let per_new = t.elapsed() / iters;

    let mut cs = EnumChangeSet::new();
    let before = live_i();
    for e in es.iter().copied() {
        cs.insert::<Payload<16>>(&mut w, e, Payload([1; 16]));
    }
    let cs_heap = live_i() - before;
    row(&format!("changeset world: {n} x 16 B"), pool, world_heap);
    row(
        &format!("  EnumChangeSet, {n} inserts (bytes off-budget)"),
        n * 16,
        cs_heap,
    );
    println!(
        "  empty EnumChangeSet::new(): {:.2} MiB heap, {:?} each (own 256 MiB pool per call)",
        empty_heap as f64 / 1_048_576.0,
        per_new
    );
    black_box(cs);

    // An empty Optimistic transaction: begin -> Tx::new -> EnumChangeSet::new.
    let access = Access::of::<(&Payload<16>,)>(&mut w);
    let strategy = Optimistic::new(&w);
    let iters = 200;
    let t = Instant::now();
    for _ in 0..iters {
        strategy
            .transact(&mut w, &access, |_tx, _w| {})
            .expect("commit");
    }
    println!(
        "  empty Optimistic::transact: {:?} each",
        t.elapsed() / iters
    );
}

/// Resident set size from /proc/self/status (VmRSS), Linux only.
fn rss_mib() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb: f64 = s
        .lines()
        .find(|l| l.starts_with("VmRSS:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()?;
    Some(kb / 1024.0)
}

fn dropped_world_retention() {
    let r0 = rss_mib();
    let mut w = World::builder()
        .memory_budget(BUDGET)
        .hugepages(HugePages::Off)
        .build()
        .expect("pool");
    w.spawn((Payload::<8>([0; 8]),));
    let r1 = rss_mib();
    drop(w);
    let r2 = rss_mib();
    // First allocation from a different pool on this thread replaces the
    // thread-local cache, releasing the last Arc to the dropped pool.
    let mut other = World::new();
    other.spawn((Payload::<8>([0; 8]),));
    let r3 = rss_mib();
    if let (Some(r0), Some(r1), Some(r2), Some(r3)) = (r0, r1, r2, r3) {
        println!(
            "RSS: before {r0:.0} MiB, pre-faulted {} MiB world live {r1:.0} MiB, \
             after drop {r2:.0} MiB, after next pool use {r3:.0} MiB",
            BUDGET >> 20
        );
    }
}

fn main() {
    println!("--- Fixed per-world cost ---");
    let before = live();
    let (w, _) = world();
    println!(
        "World::builder().memory_budget({} MiB): {:.2} MiB heap at construction",
        BUDGET >> 20,
        mib(live() - before)
    );
    drop(w);
    let before = live();
    let w = World::new();
    println!(
        "World::new() (256 MiB default pool): {:.2} MiB heap at construction\n",
        mib(live() - before)
    );
    drop(w);

    println!("--- Dense component scaling ---");
    for n in [10_000, 100_000, 1_000_000] {
        dense::<8>(n);
        dense::<64>(n);
        dense::<256>(n);
    }
    println!("\n--- Archetype count ---");
    archetypes(1);
    archetypes(16);
    println!("\n--- Sparse components ---");
    sparse(1_000_000);
    println!("\n--- Secondary indexes ---");
    indexes(1_000_000);
    println!("\n--- Planner join (result size is unbounded) ---");
    for n in [100_000, 1_000_000, 4_000_000] {
        join(n);
    }
    println!("\n--- Changesets and transactions ---");
    changesets(100_000);
    println!("\n--- Dropped world: pool lifetime ---");
    dropped_world_retention();
}
