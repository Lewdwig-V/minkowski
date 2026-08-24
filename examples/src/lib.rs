// Shared components, math, and spatial-grid helpers for the examples.
// Previously copy-pasted ~5x across individual examples.

use std::ops::{Add, AddAssign, Div, Mul, Sub};

use minkowski::Entity;
use rkyv::{Archive, Deserialize, Serialize};

// ── Components ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Pos {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Vel {
    pub dx: f32,
    pub dy: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Health(pub u32);

// ── Vec2 ──────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };

    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    pub fn length_sq(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-8 { Self::ZERO } else { self / len }
    }

    pub fn clamped(self, max_len: f32) -> Self {
        if self.length_sq() > max_len * max_len {
            self.normalized() * max_len
        } else {
            self
        }
    }
}

impl Add for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vec2 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Div<f32> for Vec2 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

// ── Toroidal helpers ──────────────────────────────────────────────────────────

/// Wrap a coordinate into `[0, size)`. Correct for offsets in `(-size, 2*size)`,
/// which covers all movement in these examples.
pub fn wrap(mut v: f32, size: f32) -> f32 {
    if v >= size {
        v -= size;
    } else if v < 0.0 {
        v += size;
    }
    v
}

/// Minimum-image signed difference on a toroidal domain of size `world_size`.
pub fn wrapped_diff(a: f32, b: f32, world_size: f32) -> f32 {
    let d = a - b;
    if d > world_size * 0.5 {
        d - world_size
    } else if d < -world_size * 0.5 {
        d + world_size
    } else {
        d
    }
}

/// Minimum-image vector from `from` to `to` on a toroidal world.
pub fn wrapped_offset(to: Vec2, from: Vec2, world_size: f32) -> Vec2 {
    Vec2::new(
        wrapped_diff(to.x, from.x, world_size),
        wrapped_diff(to.y, from.y, world_size),
    )
}

// ── TorusGrid ─────────────────────────────────────────────────────────────────

/// Uniform toroidal spatial hash grid shared by the boids / flatworm / tactical
/// examples. Mechanics are payload-agnostic: a snapshot of `(Entity, T)` is
/// bucketed by a caller-supplied position accessor, and queries walk the
/// `ring`-cell neighbourhood with `rem_euclid` wraparound.
pub struct TorusGrid<T> {
    cell_size: f32,
    grid_w: usize,
    world_size: f32,
    cells: Vec<Vec<usize>>,
    positions: Vec<(f32, f32)>,
    pub snapshot: Vec<(Entity, T)>,
}

impl<T> TorusGrid<T> {
    pub fn new(cell_size: f32, world_size: f32) -> Self {
        Self {
            cell_size,
            grid_w: (world_size / cell_size).ceil() as usize,
            world_size,
            cells: Vec::new(),
            positions: Vec::new(),
            snapshot: Vec::new(),
        }
    }

    /// Replace the snapshot and rebucket by `pos_of`.
    pub fn set_snapshot(&mut self, snapshot: Vec<(Entity, T)>, pos_of: impl Fn(&T) -> (f32, f32)) {
        self.positions = snapshot.iter().map(|(_, t)| pos_of(t)).collect();
        self.snapshot = snapshot;
        let n = self.grid_w * self.grid_w;
        self.cells.clear();
        self.cells.resize_with(n, Vec::new);
        for (i, &(x, y)) in self.positions.iter().enumerate() {
            let (cx, cy) = self.cell_of(x, y);
            self.cells[cy * self.grid_w + cx].push(i);
        }
    }

    fn cell_of(&self, x: f32, y: f32) -> (usize, usize) {
        (
            ((x / self.cell_size) as usize).min(self.grid_w.saturating_sub(1)),
            ((y / self.cell_size) as usize).min(self.grid_w.saturating_sub(1)),
        )
    }

    /// Iterate snapshot entries in the `ring`-thick cell neighbourhood of
    /// `(x, y)` (ring = 1 → classic 3×3 neighbourhood), wrapping at edges.
    /// Yields `(entity, payload, position)` — position is cached at snapshot
    /// time so the filter paths don't recompute it.
    // ponytail: playout-sized grids (grid_w ≤ thousands); usize→i32 cast safe.
    #[allow(clippy::cast_possible_wrap)]
    pub fn neighbors(
        &self,
        x: f32,
        y: f32,
        ring: i32,
    ) -> impl Iterator<Item = (Entity, &T, (f32, f32))> {
        let (cx, cy) = self.cell_of(x, y);
        let grid_w = self.grid_w;
        (-ring..=ring).flat_map(move |dy| {
            (-ring..=ring).flat_map(move |dx| {
                let nx = (cx as i32 + dx).rem_euclid(grid_w as i32) as usize;
                let ny = (cy as i32 + dy).rem_euclid(grid_w as i32) as usize;
                self.cells[ny * grid_w + nx]
                    .iter()
                    .map(|&j| (self.snapshot[j].0, &self.snapshot[j].1, self.positions[j]))
            })
        })
    }

    /// Toroidal range filter: entries whose minimum-image distance from
    /// `(x, y)` is at most `range`.
    pub fn in_range(
        &self,
        x: f32,
        y: f32,
        range: f32,
    ) -> impl Iterator<Item = (Entity, &T, (f32, f32))> {
        let ring = (range / self.cell_size).ceil() as i32;
        let range_sq = range * range;
        let ws = self.world_size;
        self.neighbors(x, y, ring).filter(move |&(.., (ex, ey))| {
            let ddx = (ex - x).abs().min(ws - (ex - x).abs());
            let ddy = (ey - y).abs().min(ws - (ey - y).abs());
            ddx * ddx + ddy * ddy <= range_sq
        })
    }
}

/// Convenience alias for the common `(f32, f32)` payload (position only).
pub type PosGrid = TorusGrid<(f32, f32)>;

impl TorusGrid<(f32, f32)> {
    /// Snapshot from flat `(entity, x, y)` triples.
    pub fn set_positions(&mut self, snapshot: Vec<(Entity, f32, f32)>) {
        let snapshot = snapshot.into_iter().map(|(e, x, y)| (e, (x, y))).collect();
        self.set_snapshot(snapshot, |&pos| pos);
    }
}
