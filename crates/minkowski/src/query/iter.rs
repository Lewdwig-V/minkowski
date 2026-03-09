//! Query iterators — [`QueryIter`] for sequential, parallel
//! ([`par_for_each`](QueryIter::par_for_each)), and chunk-based
//! ([`for_each_chunk`](QueryIter::for_each_chunk)) iteration over matched archetypes.

use super::fetch::WorldQuery;
use crate::sync::{Arc, AtomicBool, Ordering};
use rayon::prelude::*;
use std::marker::PhantomData;

/// Iterator over entities matching a query.
///
/// When created by [`World::query`](crate::World::query), carries a shared
/// flag that is set on first iteration. The world uses this flag to decide
/// whether to commit a deferred read-tick advancement — if the iterator is
/// dropped without being iterated, the `Changed<T>` window is preserved.
pub struct QueryIter<'w, Q: WorldQuery> {
    fetches: Vec<(Q::Fetch<'w>, usize)>, // (fetch_state, archetype_len)
    current_arch: usize,
    current_row: usize,
    /// Whether we've already set the iterated flag (avoids repeated atomics).
    marked: bool,
    /// Shared flag set on first iteration. `None` for iterators created
    /// without tick tracking (e.g., via `query_raw`).
    iterated: Option<Arc<AtomicBool>>,
    _marker: PhantomData<&'w Q>,
}

impl<'w, Q: WorldQuery> QueryIter<'w, Q> {
    /// Create an iterator without tick tracking (used by `query_raw`).
    pub(crate) fn new(fetches: Vec<(Q::Fetch<'w>, usize)>) -> Self {
        Self {
            fetches,
            current_arch: 0,
            current_row: 0,
            iterated: None,
            marked: false,
            _marker: PhantomData,
        }
    }

    /// Create an iterator with a shared flag for lazy tick advancement.
    pub(crate) fn with_tick_flag(
        fetches: Vec<(Q::Fetch<'w>, usize)>,
        iterated: Arc<AtomicBool>,
    ) -> Self {
        Self {
            fetches,
            current_arch: 0,
            current_row: 0,
            iterated: Some(iterated),
            marked: false,
            _marker: PhantomData,
        }
    }

    /// Mark the iterator as having been iterated (sets the shared flag once).
    #[inline]
    fn mark_iterated(&mut self) {
        if !self.marked {
            if let Some(ref flag) = self.iterated {
                flag.store(true, Ordering::Relaxed);
            }
            self.marked = true;
        }
    }

    /// Iterate archetypes, yielding typed column slices per archetype.
    /// Each invocation of `f` receives all matched rows in one archetype
    /// as contiguous slices — enabling SIMD auto-vectorization.
    pub fn for_each_chunk<F>(mut self, mut f: F)
    where
        F: FnMut(Q::Slice<'w>),
    {
        self.mark_iterated();
        for (fetch, len) in &self.fetches {
            if *len > 0 {
                let slices = unsafe { Q::as_slice(fetch, *len) };
                f(slices);
            }
        }
    }

    /// Execute `f` for each matched entity in parallel using rayon.
    /// Parallelizes across rows within each archetype.
    pub fn par_for_each<F>(mut self, f: F)
    where
        F: Fn(Q::Item<'_>) + Send + Sync,
    {
        self.mark_iterated();
        for (fetch, len) in &self.fetches {
            (0..*len).into_par_iter().for_each(|row| {
                let item = unsafe { Q::fetch(fetch, row) };
                f(item);
            });
        }
    }
}

impl<'w, Q: WorldQuery> Iterator for QueryIter<'w, Q> {
    type Item = Q::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_arch >= self.fetches.len() {
                return None;
            }
            let (ref fetch, len) = self.fetches[self.current_arch];
            if self.current_row < len {
                let item = unsafe { Q::fetch(fetch, self.current_row) };
                self.current_row += 1;
                self.mark_iterated();
                return Some(item);
            }
            self.current_arch += 1;
            self.current_row = 0;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining: usize = self.fetches[self.current_arch..]
            .iter()
            .map(|(_, len)| *len)
            .sum::<usize>()
            .saturating_sub(self.current_row);
        (remaining, Some(remaining))
    }
}

#[cfg(test)]
mod tests {
    use crate::entity::Entity;
    use crate::world::World;

    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Pos {
        x: f32,
        y: f32,
    }
    #[derive(Debug, PartialEq, Clone, Copy)]
    struct Vel {
        dx: f32,
        dy: f32,
    }
    #[test]
    fn iterate_single_archetype() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 },));
        world.spawn((Pos { x: 3.0, y: 0.0 },));

        let positions: Vec<f32> = world.query::<&Pos>().map(|p| p.x).collect();
        assert_eq!(positions, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn iterate_multiple_archetypes() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 0.0, dy: 0.0 }));

        // Query for &Pos matches both archetypes
        let count = world.query::<&Pos>().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn iterate_filters_archetypes() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 0.0, dy: 0.0 }));

        // Query for (&Pos, &Vel) only matches the second archetype
        let count = world.query::<(&Pos, &Vel)>().count();
        assert_eq!(count, 1);
    }

    #[test]
    fn iterate_with_entity() {
        let mut world = World::new();
        let e1 = world.spawn((Pos { x: 1.0, y: 0.0 },));
        let e2 = world.spawn((Pos { x: 2.0, y: 0.0 },));

        let entities: Vec<Entity> = world.query::<(Entity, &Pos)>().map(|(e, _)| e).collect();
        assert_eq!(entities, vec![e1, e2]);
    }

    #[test]
    fn iterate_empty() {
        let mut world = World::new();
        let count = world.query::<&Pos>().count();
        assert_eq!(count, 0);
    }

    #[test]
    fn mutate_during_iteration() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 }, Vel { dx: 10.0, dy: 0.0 }));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 20.0, dy: 0.0 }));

        for (pos, vel) in world.query::<(&mut Pos, &Vel)>() {
            pos.x += vel.dx;
        }

        let xs: Vec<f32> = world.query::<&Pos>().map(|p| p.x).collect();
        assert_eq!(xs, vec![11.0, 22.0]);
    }

    #[test]
    fn for_each_chunk_yields_correct_data() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 },));
        world.spawn((Pos { x: 3.0, y: 0.0 },));

        let mut total = 0.0f32;
        world.query::<&Pos>().for_each_chunk(|positions| {
            for p in positions {
                total += p.x;
            }
        });
        assert_eq!(total, 6.0);
    }

    #[test]
    fn for_each_chunk_mutation() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 }, Vel { dx: 10.0, dy: 0.0 }));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 20.0, dy: 0.0 }));

        world
            .query::<(&mut Pos, &Vel)>()
            .for_each_chunk(|(positions, velocities)| {
                for i in 0..positions.len() {
                    positions[i].x += velocities[i].dx;
                }
            });

        let xs: Vec<f32> = world.query::<&Pos>().map(|p| p.x).collect();
        assert_eq!(xs, vec![11.0, 22.0]);
    }

    #[test]
    fn for_each_chunk_skips_empty() {
        let mut world = World::new();
        let e = world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.despawn(e);

        let mut chunk_count = 0;
        world.query::<&Pos>().for_each_chunk(|_| {
            chunk_count += 1;
        });
        assert_eq!(chunk_count, 0);
    }

    #[test]
    fn for_each_chunk_multiple_archetypes() {
        let mut world = World::new();
        world.spawn((Pos { x: 1.0, y: 0.0 },));
        world.spawn((Pos { x: 2.0, y: 0.0 }, Vel { dx: 0.0, dy: 0.0 }));

        let mut chunk_count = 0;
        let mut total = 0.0f32;
        world.query::<&Pos>().for_each_chunk(|positions| {
            chunk_count += 1;
            for p in positions {
                total += p.x;
            }
        });
        assert_eq!(chunk_count, 2);
        assert_eq!(total, 3.0);
    }

    #[test]
    fn par_for_each_updates_all() {
        use std::sync::atomic::{AtomicU32, Ordering};

        let mut world = World::new();
        for i in 0..1000u32 {
            world.spawn((Pos {
                x: i as f32,
                y: 0.0,
            },));
        }

        let sum = AtomicU32::new(0);
        world.query::<&Pos>().par_for_each(|pos| {
            sum.fetch_add(pos.x as u32, Ordering::Relaxed);
        });
        // Sum of 0..1000 = 999 * 1000 / 2 = 499500
        assert_eq!(sum.load(Ordering::SeqCst), 499500);
    }

    #[test]
    fn par_for_each_mutation() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn((
                Pos {
                    x: i as f32,
                    y: 0.0,
                },
                Vel { dx: 1.0, dy: 0.0 },
            ));
        }

        world
            .query::<(&mut Pos, &Vel)>()
            .par_for_each(|(pos, vel)| {
                pos.x += vel.dx;
            });

        let xs: Vec<f32> = world.query::<&Pos>().map(|p| p.x).collect();
        for (i, x) in xs.iter().enumerate() {
            assert_eq!(*x, i as f32 + 1.0);
        }
    }
}
