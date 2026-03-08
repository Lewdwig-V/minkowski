use arbitrary::Arbitrary;

// --- Component types ---
// Different sizes and alignments to exercise BlobVec layout code.

#[derive(
    Clone, Copy, Debug, PartialEq, Arbitrary, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct A(pub u32); // 4 bytes, 4-align

#[derive(
    Clone, Copy, Debug, PartialEq, Arbitrary, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct B(pub u64); // 8 bytes, 8-align

#[derive(
    Clone, Copy, Debug, PartialEq, Arbitrary, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct C(pub [u8; 3]); // 3 bytes, 1-align (odd size)

#[derive(
    Clone, Copy, Debug, PartialEq, Arbitrary, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct D(pub f32); // 4 bytes, 4-align

// --- Enums for fuzz-driven selection ---

#[derive(Clone, Debug, Arbitrary)]
pub enum BundleKind {
    JustA(A),
    AB(A, B),
    AC(A, C),
    BD(B, D),
    ABC(A, B, C),
    ABCD(A, B, C, D),
}

#[derive(Clone, Debug, Arbitrary)]
pub enum CompVal {
    A(A),
    B(B),
    C(C),
    D(D),
}

#[derive(Clone, Copy, Debug, Arbitrary)]
pub enum CompKind {
    A,
    B,
    C,
    D,
}

#[derive(Clone, Copy, Debug, Arbitrary)]
pub enum QueryShape {
    RefA,
    RefAB,
    RefABC,
    MutA,
    MutARefB,
    MutARefBC,
    RefB,
    MutD,
}

/// Spawn a bundle into the world based on the fuzzed BundleKind.
pub fn spawn_bundle(world: &mut minkowski::World, kind: &BundleKind) -> minkowski::Entity {
    match kind {
        BundleKind::JustA(a) => world.spawn((*a,)),
        BundleKind::AB(a, b) => world.spawn((*a, *b)),
        BundleKind::AC(a, c) => world.spawn((*a, *c)),
        BundleKind::BD(b, d) => world.spawn((*b, *d)),
        BundleKind::ABC(a, b, c) => world.spawn((*a, *b, *c)),
        BundleKind::ABCD(a, b, c, d) => world.spawn((*a, *b, *c, *d)),
    }
}

/// Run a query of the given shape, returning the number of matched entities.
pub fn run_query(world: &mut minkowski::World, shape: QueryShape) -> usize {
    match shape {
        QueryShape::RefA => world.query::<(&A,)>().count(),
        QueryShape::RefAB => world.query::<(&A, &B)>().count(),
        QueryShape::RefABC => world.query::<(&A, &B, &C)>().count(),
        QueryShape::MutA => world.query::<(&mut A,)>().count(),
        QueryShape::MutARefB => world.query::<(&mut A, &B)>().count(),
        QueryShape::MutARefBC => world.query::<(&mut A, &B, &C)>().count(),
        QueryShape::RefB => world.query::<(&B,)>().count(),
        QueryShape::MutD => world.query::<(&mut D,)>().count(),
    }
}

/// Assert world invariants against a live entity tracker.
pub fn assert_invariants(world: &minkowski::World, live: &[minkowski::Entity]) {
    assert_eq!(
        world.entity_count(),
        live.len(),
        "entity_count mismatch: world has {} but tracker has {}",
        world.entity_count(),
        live.len()
    );
    for (i, &entity) in live.iter().enumerate() {
        assert!(
            world.is_alive(entity),
            "entity at tracker index {i} is not alive"
        );
        assert!(
            world.is_placed(entity),
            "entity at tracker index {i} is not placed"
        );
    }
}
