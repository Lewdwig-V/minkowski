use rkyv::{Archive, Deserialize, Serialize};

/// 4x4 matrix -- 64 bytes (one cache line), 16-byte aligned. Used for heavy_compute.
#[derive(Clone, Copy, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct Transform {
    pub matrix: [[f32; 4]; 4],
}

/// 3D position vector -- 12 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Position {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D rotation vector -- 12 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Archive, Serialize, Deserialize)]
#[repr(C)]
pub struct Rotation {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D velocity vector -- 12 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Archive, Serialize, Deserialize)]
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
    codecs.register::<Transform>(world).unwrap();
    codecs.register::<Position>(world).unwrap();
    codecs.register::<Rotation>(world).unwrap();
    codecs.register::<Velocity>(world).unwrap();
}

/// Full 4x4 matrix inversion via cofactor expansion (~600 FLOPs per call).
pub fn invert_4x4(m: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    // Precompute 2x2 subdeterminants from rows 2-3
    let s0 = m[2][0] * m[3][1] - m[2][1] * m[3][0];
    let s1 = m[2][0] * m[3][2] - m[2][2] * m[3][0];
    let s2 = m[2][0] * m[3][3] - m[2][3] * m[3][0];
    let s3 = m[2][1] * m[3][2] - m[2][2] * m[3][1];
    let s4 = m[2][1] * m[3][3] - m[2][3] * m[3][1];
    let s5 = m[2][2] * m[3][3] - m[2][3] * m[3][2];

    // Precompute 2x2 subdeterminants from rows 0-1
    let c0 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let c1 = m[0][0] * m[1][2] - m[0][2] * m[1][0];
    let c2 = m[0][0] * m[1][3] - m[0][3] * m[1][0];
    let c3 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let c4 = m[0][1] * m[1][3] - m[0][3] * m[1][1];
    let c5 = m[0][2] * m[1][3] - m[0][3] * m[1][2];

    let det = c0 * s5 - c1 * s4 + c2 * s3 + c3 * s2 - c4 * s1 + c5 * s0;

    if det.abs() < 1e-10 {
        return *m; // near-singular — return input unchanged
    }

    let inv_det = 1.0 / det;

    // Adjugate matrix (transposed cofactors), scaled by 1/det
    [
        [
            (m[1][1] * s5 - m[1][2] * s4 + m[1][3] * s3) * inv_det,
            (-m[0][1] * s5 + m[0][2] * s4 - m[0][3] * s3) * inv_det,
            (c3 * m[3][3] - c4 * m[3][2] + c5 * m[3][1]) * inv_det, // was c3,c4,c5 rows01 x row3
            (-c3 * m[2][3] + c4 * m[2][2] - c5 * m[2][1]) * inv_det,
        ],
        [
            (-m[1][0] * s5 + m[1][2] * s2 - m[1][3] * s1) * inv_det,
            (m[0][0] * s5 - m[0][2] * s2 + m[0][3] * s1) * inv_det,
            (-c1 * m[3][3] + c2 * m[3][2] - c5 * m[3][0]) * inv_det,
            (c1 * m[2][3] - c2 * m[2][2] + c5 * m[2][0]) * inv_det,
        ],
        [
            (m[1][0] * s4 - m[1][1] * s2 + m[1][3] * s0) * inv_det,
            (-m[0][0] * s4 + m[0][1] * s2 - m[0][3] * s0) * inv_det,
            (c0 * m[3][3] - c2 * m[3][1] + c4 * m[3][0]) * inv_det,
            (-c0 * m[2][3] + c2 * m[2][1] - c4 * m[2][0]) * inv_det,
        ],
        [
            (-m[1][0] * s3 + m[1][1] * s1 - m[1][2] * s0) * inv_det,
            (m[0][0] * s3 - m[0][1] * s1 + m[0][2] * s0) * inv_det,
            (-c0 * m[3][2] + c1 * m[3][1] - c3 * m[3][0]) * inv_det,
            (c0 * m[2][2] - c1 * m[2][1] + c3 * m[2][0]) * inv_det,
        ],
    ]
}
