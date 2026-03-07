"""Minkowski ECS — Python bindings for the Minkowski entity-component system.

Quick start::

    import minkowski_py as mk

    world = mk.World()
    registry = mk.ReducerRegistry(world)

    # Spawn entities
    world.spawn("Position,Velocity", pos_x=0.0, pos_y=0.0, vel_x=1.0, vel_y=0.0)

    # Query as Polars DataFrame
    df = world.query("Position", "Velocity")

    # Run a Rust reducer
    registry.run("movement", world, dt=0.016)
"""

try:
    from minkowski_py._minkowski import CircuitSim, ReducerRegistry, SpatialGrid, World
except ImportError as e:
    raise ImportError(
        "Failed to import Minkowski native module. "
        "Build with: cd crates/minkowski-py && maturin develop --release"
    ) from e

__all__ = ["CircuitSim", "ReducerRegistry", "SpatialGrid", "World"]
__version__ = "0.2.0"
