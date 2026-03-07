# CircuitSim + Flatworm Notebook Design

**Date**: 2026-03-07

## CircuitSim

### API

```python
sim = mk.CircuitSim(world, vcc=12.0, dt=1e-7, sample_every=20)
sim.step(200_000)              # run 200K timesteps
df = sim.waveform()            # → Polars DataFrame(time, v_555, v_filter, v_opamp)
sim.step(100_000)              # run more (appends to waveform)
sim.reset()                    # clear waveform buffer, reset circuit state
```

### Implementation

- `#[pyclass]` in `crates/minkowski-py/src/circuit.rs`
- Hardcoded 555→LCR→741 topology (not configurable from Python)
- Owns all circuit state internally (Timer555, Inductor, Resistor, OpAmp741, node
  voltages, current accumulators)
- `step(n)` runs the 6-phase simulation loop N times, sampling every `sample_every`
  steps into internal `Vec<f64>` buffers
- `waveform()` converts buffers to Arrow RecordBatch → Polars DataFrame
  - Columns: time (f64), v_555 (f64), v_filter (f64), v_opamp (f64)
- `reset()` clears buffers and reinitializes circuit to t=0

### Circuit topology

```
  +12V ─── 555 astable (R1=10k, R2=10k, C_t=10nF → ~4.8 kHz square wave)
           │
           OUT ─── R_src=1kΩ ─── L=10mH ─── C_filt=100nF ─── GND
                                   │
                                   ├─── R_load=10kΩ ─── GND
                                   │
                                   └─── [741 follower] ─── V_out
```

### Touch points

1. New file `crates/minkowski-py/src/circuit.rs`
2. `lib.rs` — register CircuitSim
3. `__init__.py` — export CircuitSim
4. New notebook `notebooks/circuit.ipynb`

### Notebook content

1. Create sim, run 200K steps
2. Plot full waveform (3 subplots: 555 square, filter sinusoid, opamp follower)
3. Zoom to last 2ms for steady-state
4. FFT of filter output to show bandpass
5. Overlay 555 + filter to show harmonic filtering

## Flatworm

### New components

| Component | Type | Arrow field |
|-----------|------|-------------|
| WormSize | f32 | `worm_size` |
| Nutrition | f32 | `nutrition` |

### Discriminator pattern

- Worms: have Position + Heading + Energy + WormSize
- Food: have Position + Nutrition
- No marker components needed — query by component presence

### New reducers

| Reducer | Reads | Writes | Args |
|---------|-------|--------|------|
| `worm_move` | Position, Heading, Energy | Position | dt, world_size |
| `worm_metabolism` | Energy, WormSize | Energy, WormSize | dt |
| `worm_feed` | Position, Energy, WormSize (worms) + Position, Nutrition (food snapshot) | Energy, WormSize | feed_radius, world_size |
| `worm_fission` | Energy, WormSize, Position, Heading (CommandBuffer spawn) | — | min_energy |
| `worm_starve` | Energy (CommandBuffer despawn) | — | — |

### New spawn dispatch arms

- `Energy,Heading,Position,WormSize` (worm bundle)
- `Nutrition,Position` (food bundle)

### Notebook content

1. Spawn 200 worms + 500 food on 500×500 toroidal grid
2. Per-frame: move → feed → metabolism → fission → starve
3. Scatter: worms (size ∝ WormSize, color by energy), food (green dots)
4. Population dynamics chart (worm + food count over time)
5. Energy histogram at end

### Touch points

1. `components.rs` — register WormSize, Nutrition
2. `reducers.rs` — 5 new reducers + dispatch arms
3. `pyworld.rs` — 2 new spawn dispatch arms
4. New notebook `notebooks/flatworm.ipynb`

## Environment setup

```bash
cd crates/minkowski-py
uv sync --all-extras && source .venv/bin/activate
maturin develop --release
jupyter lab notebooks/circuit.ipynb
```
