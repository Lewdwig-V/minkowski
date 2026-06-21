//! Deterministic parameterized workload builder for LSM benchmarks. Gated behind
//! the `bench-support` feature so it never ships in normal builds. Every world is
//! a pure function of `WorkloadParams` — seeded SplitMix64, no wall-clock.

use minkowski::World;
use rkyv::{Archive, Deserialize, Serialize};

use crate::codec::CodecRegistry;

#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct BenchPos {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct BenchVel {
    pub dx: f32,
    pub dy: f32,
}

#[derive(Clone, Archive, Serialize, Deserialize)]
pub struct BenchName {
    pub text: String,
}

/// Component payload shape: fixed-size POD (`BenchPos`/`BenchVel`) versus
/// heap-backed (`BenchName` with a `String`). Drives the raw-copy fast path
/// versus the rkyv serialized-column path in flush/compaction.
#[derive(Clone, Copy)]
pub enum Shape {
    Pod,
    Heap,
}

/// Archetype layout: every entity in one archetype (`Single`) versus a
/// deterministic spread across several archetypes (`Fragmented`), which
/// exercises multi-archetype flush/merge paths.
#[derive(Clone, Copy)]
pub enum Layout {
    Single,
    Fragmented,
}

/// Fully describes a deterministic workload. `build_world` is a pure function
/// of these fields plus the seeded PRNG state derived from `seed`.
#[derive(Clone, Copy)]
pub struct WorkloadParams {
    pub entities: usize,
    pub shape: Shape,
    pub layout: Layout,
    pub sparse: bool,
    pub seed: u64,
}

/// Seeded SplitMix64 step. Deterministic, no wall-clock — the sole source of
/// non-uniformity in the builder.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Spawn one entity numbered `i` with the shape/layout of `params`. `s` is the
/// running PRNG state used to pick the archetype bucket under `Layout::Fragmented`.
/// Shared by `build_world` (initial population) and `grow` (incremental growth).
fn spawn_one(world: &mut World, params: &WorkloadParams, i: usize, s: &mut u64) {
    let bucket = if matches!(params.layout, Layout::Fragmented) {
        (splitmix(s) % 3) as u8
    } else {
        0
    };
    match params.shape {
        Shape::Pod => match bucket {
            0 => {
                world.spawn((
                    BenchPos {
                        x: i as f32,
                        y: 0.0,
                    },
                    BenchVel { dx: 1.0, dy: 0.0 },
                ));
            }
            1 => {
                world.spawn((BenchPos {
                    x: i as f32,
                    y: 0.0,
                },));
            }
            _ => {
                world.spawn((BenchVel {
                    dx: 1.0,
                    dy: i as f32,
                },));
            }
        },
        Shape::Heap => match bucket {
            0 => {
                world.spawn((
                    BenchName {
                        text: format!("e{i}"),
                    },
                    BenchVel { dx: 1.0, dy: 0.0 },
                ));
            }
            1 => {
                world.spawn((BenchName {
                    text: format!("e{i}"),
                },));
            }
            _ => {
                world.spawn((BenchName {
                    text: format!("name-{i}-padded"),
                },));
            }
        },
    }
}

/// Build a deterministic World + matching CodecRegistry for `params`.
///
/// # Panics
/// Panics if codec registration fails (a programming error: the bench
/// components are statically known to be raw-copy / rkyv compatible).
#[must_use]
pub fn build_world(params: &WorkloadParams) -> (World, CodecRegistry) {
    let mut world = World::new();
    let mut codecs = CodecRegistry::new();
    codecs
        .register_as::<BenchPos>("bench_pos", &mut world)
        .unwrap();
    codecs
        .register_as::<BenchVel>("bench_vel", &mut world)
        .unwrap();
    codecs
        .register_as::<BenchName>("bench_name", &mut world)
        .unwrap();

    let mut s = params.seed;
    for i in 0..params.entities {
        spawn_one(&mut world, params, i, &mut s);
    }
    let _ = params.sparse; // placeholder until a sparse-enabled bench needs it
    (world, codecs)
}

/// Spawn `count` ADDITIONAL entities into `world`, numbered `start_index..`, so
/// successive flushes accumulate and the dataset grows — letting it cascade
/// through LSM levels (the only way the level-count `N` becomes observable; a
/// constant dataset just re-supersedes itself and never reaches deep levels).
/// The components are already registered (from `build_world`), so this only spawns.
pub fn grow(
    world: &mut World,
    params: &WorkloadParams,
    start_index: usize,
    count: usize,
    seed: u64,
) {
    let mut s = seed;
    for i in start_index..start_index + count {
        spawn_one(world, params, i, &mut s);
    }
}

/// Mutate `ratio` (0.0..=1.0) of `BenchVel` rows deterministically so a
/// subsequent flush is non-trivially dirty (drives dedup/merge + write-amp).
///
/// Iterates `BenchVel` columns in archetype order; the first `target` rows get a
/// seeded perturbation. The perturbation is a pure function of `seed`, so repeated
/// calls with the same `(world, ratio, seed)` produce identical mutations.
pub fn overwrite(world: &mut World, ratio: f64, seed: u64) {
    let total = world.query::<(&BenchVel,)>().count();
    let target = ((total as f64) * ratio).round() as usize;
    let mut hit = 0usize;
    let mut s = seed;
    world.query::<(&mut BenchVel,)>().for_each(|(v,)| {
        if hit < target {
            // Deterministic, non-zero perturbation derived from the seed.
            v.dx += 1.0 + ((splitmix(&mut s) & 0xFF) as f32);
            hit += 1;
        }
    });
}

/// Write amplification: output bytes ÷ input bytes over a compaction.
#[derive(Clone, Copy, Debug, Default)]
pub struct WriteAmp {
    pub input_bytes: u64,
    pub output_bytes: u64,
}

impl WriteAmp {
    #[must_use]
    pub fn ratio(&self) -> f64 {
        if self.input_bytes == 0 {
            0.0
        } else {
            self.output_bytes as f64 / self.input_bytes as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_requested_entity_count_and_shape() {
        let (mut world, _c) = build_world(&WorkloadParams {
            entities: 1000,
            shape: Shape::Pod,
            layout: Layout::Single,
            sparse: false,
            seed: 42,
        });
        assert_eq!(world.query::<(&BenchPos,)>().count(), 1000);
    }

    #[test]
    fn heap_shape_uses_string_component() {
        let (mut world, _c) = build_world(&WorkloadParams {
            entities: 16,
            shape: Shape::Heap,
            layout: Layout::Single,
            sparse: false,
            seed: 7,
        });
        assert_eq!(world.query::<(&BenchName,)>().count(), 16);
    }

    #[test]
    fn deterministic_same_seed_same_world() {
        let p = WorkloadParams {
            entities: 64,
            shape: Shape::Pod,
            layout: Layout::Fragmented,
            sparse: true,
            seed: 99,
        };
        let (a, _) = build_world(&p);
        let (b, _) = build_world(&p);
        assert_eq!(a.archetype_count(), b.archetype_count());
    }

    #[test]
    fn write_amp_ratio_is_output_over_input() {
        let wa = WriteAmp {
            input_bytes: 400,
            output_bytes: 100,
        };
        assert!((wa.ratio() - 0.25).abs() < 1e-9);
        assert_eq!(WriteAmp::default().ratio(), 0.0);
    }
}
