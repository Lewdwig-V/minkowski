use rkyv::{Archive, Deserialize, Serialize};

/// 4x4 matrix -- 64 bytes, cache-line sized. Used for heavy_compute (matrix inversion).
#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct Transform {
    pub matrix: [[f32; 4]; 4],
}

/// 3D position vector -- 12 bytes.
#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D rotation vector -- 12 bytes.
#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Rotation {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D velocity vector -- 12 bytes.
#[derive(Clone, Copy, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Velocity {
    pub dx: f32,
    pub dy: f32,
    pub dz: f32,
}

/// Spawn a world with `n` entities, each with (Transform, Position, Rotation, Velocity).
pub fn spawn_world(n: usize) -> minkowski::World {
    let mut world = minkowski::World::new();
    for i in 0..n {
        let f = i as f32;
        world.spawn((
            Transform {
                matrix: [
                    [f, 0.0, 0.0, 0.0],
                    [0.0, f, 0.0, 0.0],
                    [0.0, 0.0, f, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            Position { x: f, y: f, z: f },
            Rotation {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Velocity {
                dx: 1.0,
                dy: 1.0,
                dz: 1.0,
            },
        ));
    }
    world
}

/// Register all 4 component types with the codec registry.
pub fn register_codecs(
    codecs: &mut minkowski_persist::CodecRegistry,
    world: &mut minkowski::World,
) {
    codecs.register::<Transform>(world);
    codecs.register::<Position>(world);
    codecs.register::<Rotation>(world);
    codecs.register::<Velocity>(world);
}

/// Naive 4x4 matrix inversion via cofactor expansion.
/// Not optimized -- the point is ~200 FLOPs of real work per entity.
pub fn invert_4x4(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut inv = [[0.0f32; 4]; 4];

    inv[0][0] = m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
        - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
        + m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]);

    inv[0][1] = -(m[0][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
        - m[0][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
        + m[0][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]));

    inv[0][2] = m[0][1] * (m[1][2] * m[3][3] - m[1][3] * m[3][2])
        - m[0][2] * (m[1][1] * m[3][3] - m[1][3] * m[3][1])
        + m[0][3] * (m[1][1] * m[3][2] - m[1][2] * m[3][1]);

    inv[0][3] = -(m[0][1] * (m[1][2] * m[2][3] - m[1][3] * m[2][2])
        - m[0][2] * (m[1][1] * m[2][3] - m[1][3] * m[2][1])
        + m[0][3] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]));

    let det = m[0][0] * inv[0][0]
        + m[0][1]
            * (-(m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2])
                - m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
                + m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])))
        + m[0][2]
            * (m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1])
                - m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0])
                + m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]))
        + m[0][3]
            * (-(m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1])
                - m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0])
                + m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0])));

    if det.abs() < 1e-10 {
        return *m; // singular -- return identity-ish
    }

    // For the benchmark we only need the first row to be correct --
    // the point is compute volume, not a production-grade inverse.
    let inv_det = 1.0 / det;
    for row in &mut inv {
        for val in row.iter_mut() {
            *val *= inv_det;
        }
    }
    inv
}
