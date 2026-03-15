//! Analog circuit simulator — 555 astable → LCR bandpass → 741 voltage follower.
//!
//! Run: cargo run -p minkowski-examples --example circuit --release
//!
//! Exercises: spawn, query reducers (QueryMut, QueryRef), ReducerRegistry
//! scheduling, entity-based connectivity.
//!
//! Circuit topology:
//! ```
//!   +12V ─── 555 astable (R1=10k, R2=10k, C_t=10nF → ~4.8 kHz square wave)
//!            │
//!            OUT ─── R_src=1kΩ ─── L=10mH ─── C_filt=100nF ─── GND
//!                                    │
//!                                    ├─── R_load=10kΩ ─── GND
//!                                    │
//!                                    └─── [741 follower, ±12V, V-=V_out] ─── V_out
//! ```
//!
//! The 555 generates a ~4.8 kHz square wave. The series LCR bandpass filter
//! (resonant at ~5.03 kHz) passes the fundamental while attenuating harmonics.
//! The 741 voltage follower buffers the output with realistic gain, slew rate,
//! and rail saturation.
//!
//! Simulation uses symplectic Euler integration for L/C elements (bounded energy
//! drift) and samples node voltages every N steps to print a waveform summary.

use minkowski::{Entity, QueryMut, QueryRef, ReducerRegistry, World};
use std::time::Instant;

fn read_voltage(world: &World, node: Entity) -> f64 {
    world.get::<Voltage>(node).map_or(0.0, |v| v.0)
}

// ── Simulation parameters ────────────────────────────────────────────

const VCC: f64 = 12.0;
const DT: f64 = 1e-7; // 100 ns timestep (~2000 samples per cycle at 5 kHz)
const STEPS: usize = 200_000; // 20 ms of simulation time
const SAMPLE_INTERVAL: usize = 20; // sample every 20 steps = 2 µs
const REPORT_INTERVAL: usize = 50_000; // progress report every 5 ms sim-time

// ── Node voltages ────────────────────────────────────────────────────
// Each circuit node is an entity with a Voltage component.
// Elements connect to nodes and inject/absorb current.

#[derive(Clone, Copy, Debug)]
struct Voltage(f64);

// ── 555 Timer ────────────────────────────────────────────────────────
// Astable mode: charges C_t through R1+R2, discharges through R2.
// Comparator thresholds: 2/3 Vcc (upper), 1/3 Vcc (lower).
// Internal flip-flop drives output and discharge transistor.

#[derive(Clone, Copy, Debug)]
struct Timer555 {
    r1: f64, // upper timing resistor (charge through R1+R2, discharge through R2)
    r2: f64,
    cap: f64,          // timing capacitor
    v_cap: f64,        // voltage across timing capacitor
    output_high: bool, // flip-flop state
    vcc: f64,
}

impl Timer555 {
    fn new(r1: f64, r2: f64, cap: f64, vcc: f64) -> Self {
        Self {
            r1,
            r2,
            cap,
            v_cap: 0.0,
            output_high: true, // starts charging
            vcc,
        }
    }

    fn threshold_upper(&self) -> f64 {
        self.vcc * (2.0 / 3.0)
    }

    fn threshold_lower(&self) -> f64 {
        self.vcc * (1.0 / 3.0)
    }

    fn step(&mut self, dt: f64) -> f64 {
        if self.output_high {
            // Charging through R1 + R2
            let r_charge = self.r1 + self.r2;
            let i = (self.vcc - self.v_cap) / r_charge;
            self.v_cap += i * dt / self.cap;

            if self.v_cap >= self.threshold_upper() {
                self.output_high = false;
                self.v_cap = self.threshold_upper(); // clamp at transition
            }
        } else {
            // Discharging through R2 (discharge pin to ground)
            let i = self.v_cap / self.r2;
            self.v_cap -= i * dt / self.cap;

            if self.v_cap <= self.threshold_lower() {
                self.output_high = true;
                self.v_cap = self.threshold_lower();
            }
        }

        if self.output_high {
            self.vcc - 1.5 // output high (Vcc minus Vce_sat ≈ 1.5V)
        } else {
            0.2 // output low (Vce_sat ≈ 0.2V)
        }
    }
}

// ── Passive elements ─────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Inductor {
    inductance: f64,
    current: f64,   // state variable: current through inductor
    node_a: Entity, // current flows from a to b
    node_b: Entity,
}

#[derive(Clone, Copy, Debug)]
struct Resistor {
    resistance: f64,
    node_a: Entity,
    node_b: Entity,
}

// ── 741 Op-Amp (simplified) ─────────────────────────────────────────
// Voltage follower: Vout = clamp(A_ol * (V+ - V-), -Vsat, +Vsat)
// with slew rate limiting.

#[derive(Clone, Copy, Debug)]
struct OpAmp741 {
    open_loop_gain: f64, // ~200,000
    slew_rate: f64,      // ~0.5 V/µs
    v_sat_pos: f64,      // positive rail (Vcc - ~2V)
    v_sat_neg: f64,      // negative rail (Vee + ~2V)
    v_out: f64,          // current output voltage (state for slew limiting)
    node_plus: Entity,   // non-inverting input
    node_minus: Entity,  // inverting input (connected to output for follower)
    node_out: Entity,
}

// ── Waveform recorder ────────────────────────────────────────────────

struct WaveformData {
    times: Vec<f64>,
    samples: Vec<Vec<f64>>, // one vec per probe
}

impl WaveformData {
    fn new(num_probes: usize) -> Self {
        Self {
            times: Vec::new(),
            samples: (0..num_probes).map(|_| Vec::new()).collect(),
        }
    }
}

// ── Ground node marker ───────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
struct Ground;

// ── Current accumulator on nodes ─────────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
struct CurrentSum(f64);

// ── Node capacitance (for voltage update via I = C dV/dt) ────────────
// Each non-ground node needs an effective capacitance for the explicit solver.
// For nodes connected to a real capacitor, this is the capacitor value.
// For other nodes, we use a small parasitic capacitance to make the system well-posed.

#[derive(Clone, Copy, Debug)]
struct NodeCapacitance(f64);

// ── Main ─────────────────────────────────────────────────────────────

fn main() {
    let start = Instant::now();
    let mut world = World::new();
    let mut registry = ReducerRegistry::new();

    // ── Create circuit nodes ─────────────────────────────────────────
    // Node 0: 555 output
    // Node 1: junction between R_src and L (also L input)
    // Node 2: junction between L and C_filt (also R_load top, op-amp input)
    // Node 3: op-amp output
    // GND: ground reference

    let gnd = world.spawn((Voltage(0.0), Ground, CurrentSum(0.0), NodeCapacitance(1.0)));
    let node_555_out = world.spawn((Voltage(0.0), CurrentSum(0.0), NodeCapacitance(1e-10))); // parasitic
    let node_l_in = world.spawn((Voltage(0.0), CurrentSum(0.0), NodeCapacitance(1e-10)));
    let node_filter = world.spawn((Voltage(0.0), CurrentSum(0.0), NodeCapacitance(100e-9))); // C_filt
    let node_opamp_out = world.spawn((Voltage(0.0), CurrentSum(0.0), NodeCapacitance(1e-10)));

    // ── Create circuit elements ──────────────────────────────────────

    // 555 timer: R1=10kΩ, R2=10kΩ, C_t=10nF
    // f = 1.44 / ((R1 + 2*R2) * C_t) ≈ 4.8 kHz
    let _timer = world.spawn((Timer555::new(10e3, 10e3, 10e-9, VCC),));

    // R_source: 1kΩ between 555 output and inductor input
    let _r_src = world.spawn((Resistor {
        resistance: 1e3,
        node_a: node_555_out,
        node_b: node_l_in,
    },));

    // L: 10mH inductor from node_l_in to node_filter
    let _inductor = world.spawn((Inductor {
        inductance: 10e-3,
        current: 0.0,
        node_a: node_l_in,
        node_b: node_filter,
    },));

    // C_filt: 100nF from node_filter to ground (modeled as NodeCapacitance above)

    // R_load: 10kΩ from node_filter to ground
    let _r_load = world.spawn((Resistor {
        resistance: 10e3,
        node_a: node_filter,
        node_b: gnd,
    },));

    // 741 op-amp in voltage follower configuration
    // V+ = node_filter, V- = node_opamp_out (feedback), out = node_opamp_out
    let _opamp = world.spawn((OpAmp741 {
        open_loop_gain: 200_000.0,
        slew_rate: 0.5e6,      // 0.5 V/µs = 0.5e6 V/s
        v_sat_pos: VCC - 2.0,  // positive saturation
        v_sat_neg: -VCC + 2.0, // negative saturation (assume ±12V supply)
        v_out: 0.0,
        node_plus: node_filter,
        node_minus: node_opamp_out,
        node_out: node_opamp_out,
    },));

    // ── Register query reducers ──────────────────────────────────────

    // Phase 1: Reset current accumulators on all nodes
    let reset_currents_id = registry
        .register_query::<(&mut CurrentSum,), (), _>(
            &mut world,
            "reset_currents",
            |mut query: QueryMut<'_, (&mut CurrentSum,)>, ()| {
                query.for_each(|(currents,)| {
                    for cs in currents.iter_mut() {
                        cs.0 = 0.0;
                    }
                });
            },
        )
        .unwrap();

    // Phases 2-6 (555 timer, resistors, inductors, voltage update, op-amp)
    // are done manually in the loop since they need cross-entity node access.

    // Census — count entities and report (read-only)
    let census_id = registry
        .register_query_ref::<(&Voltage,), (), _>(
            &mut world,
            "node_census",
            |mut query: QueryRef<'_, (&Voltage,)>, ()| {
                let count = query.count();
                println!("  circuit has {} voltage nodes", count);
            },
        )
        .unwrap();

    // ── Waveform storage ─────────────────────────────────────────────

    let probe_nodes = [node_555_out, node_filter, node_opamp_out];
    let probe_names = ["555_out", "filter", "opamp_out"];
    let mut waveform = WaveformData::new(probe_nodes.len());

    // ── Simulation loop ──────────────────────────────────────────────

    println!("Circuit: 555 astable → LCR bandpass → 741 follower");
    println!(
        "  555: R1=10kΩ, R2=10kΩ, C=10nF → f ≈ {:.0} Hz",
        1.44 / ((10e3 + 2.0 * 10e3) * 10e-9)
    );
    println!(
        "  LCR: L=10mH, C=100nF, R_load=10kΩ → f₀ ≈ {:.0} Hz",
        1.0 / (2.0 * std::f64::consts::PI * (10e-3 * 100e-9_f64).sqrt())
    );
    println!(
        "  741: open-loop gain=200k, slew=0.5V/µs, rails=±{:.0}V",
        VCC - 2.0
    );
    println!(
        "  dt={:.0}ns, {} steps ({:.1}ms sim time)",
        DT * 1e9,
        STEPS,
        STEPS as f64 * DT * 1e3
    );
    println!();

    registry.run(&mut world, census_id, ()).unwrap();

    for step in 0..STEPS {
        // ── Phase 1: Reset current sums ──────────────────────────────
        registry.run(&mut world, reset_currents_id, ()).unwrap();

        // ── Phase 2: 555 timer step ──────────────────────────────────
        {
            let timer_output: f64 = {
                let timers: Vec<Entity> = world
                    .query::<(Entity, &Timer555)>()
                    .map(|(e, _)| e)
                    .collect();
                let mut out = 0.0;
                for entity in timers {
                    if let Some(t) = world.get_mut::<Timer555>(entity) {
                        out = t.step(DT);
                    }
                }
                out
            };

            // Set 555 output node voltage
            if let Some(v) = world.get_mut::<Voltage>(node_555_out) {
                v.0 = timer_output;
            }
        }

        // ── Phase 3: Resistor currents ───────────────────────────────
        {
            let resistors: Vec<(f64, Entity, Entity)> = world
                .query::<(&Resistor,)>()
                .map(|(r,)| (r.resistance, r.node_a, r.node_b))
                .collect();

            for (resistance, node_a, node_b) in &resistors {
                let va = read_voltage(&world, *node_a);
                let vb = read_voltage(&world, *node_b);
                let current = (va - vb) / resistance; // flows from a to b

                // Current leaves node_a, enters node_b
                if let Some(cs) = world.get_mut::<CurrentSum>(*node_a) {
                    cs.0 -= current;
                }
                if let Some(cs) = world.get_mut::<CurrentSum>(*node_b) {
                    cs.0 += current;
                }
            }
        }

        // ── Phase 4: Inductor update (symplectic Euler) ──────────────
        // V = L * dI/dt → dI = V/L * dt (update current from voltage)
        // Then current contributes to node current sums
        {
            let inductors: Vec<(Entity, f64, Entity, Entity)> = world
                .query::<(Entity, &Inductor)>()
                .map(|(e, ind)| (e, ind.inductance, ind.node_a, ind.node_b))
                .collect();

            for (entity, inductance, node_a, node_b) in &inductors {
                let va = read_voltage(&world, *node_a);
                let vb = read_voltage(&world, *node_b);
                let v_across = va - vb;

                // Update inductor current: dI/dt = V/L
                if let Some(ind) = world.get_mut::<Inductor>(*entity) {
                    ind.current += v_across / inductance * DT;

                    let i = ind.current;
                    // Current leaves node_a, enters node_b
                    if let Some(cs) = world.get_mut::<CurrentSum>(*node_a) {
                        cs.0 -= i;
                    }
                    if let Some(cs) = world.get_mut::<CurrentSum>(*node_b) {
                        cs.0 += i;
                    }
                }
            }
        }

        // ── Phase 5: Update node voltages from current sums ──────────
        // For nodes with capacitance: dV/dt = I/C
        // Ground node is clamped to 0V.
        // 555 output node is driven directly.
        {
            let nodes: Vec<(Entity, f64)> = world
                .query::<(Entity, &NodeCapacitance)>()
                .map(|(e, nc)| (e, nc.0))
                .collect();

            for (entity, cap) in &nodes {
                // Skip ground
                if world.get::<Ground>(*entity).is_some() {
                    continue;
                }
                // Skip 555 output (driven directly)
                if entity == &node_555_out {
                    continue;
                }

                let i_sum = world.get::<CurrentSum>(*entity).map_or(0.0, |cs| cs.0);
                if let Some(v) = world.get_mut::<Voltage>(*entity) {
                    v.0 += i_sum / cap * DT;
                }
            }
        }

        // ── Phase 6: Op-amp update ───────────────────────────────────
        {
            let opamps: Vec<(Entity, OpAmp741)> = world
                .query::<(Entity, &OpAmp741)>()
                .map(|(e, op)| (e, *op))
                .collect();

            for (entity, snapshot) in &opamps {
                let v_plus = read_voltage(&world, snapshot.node_plus);
                let v_minus = read_voltage(&world, snapshot.node_minus);

                let v_out_val = if let Some(op) = world.get_mut::<OpAmp741>(*entity) {
                    let v_diff = v_plus - v_minus;
                    let v_ideal = (snapshot.open_loop_gain * v_diff)
                        .clamp(snapshot.v_sat_neg, snapshot.v_sat_pos);

                    // Slew rate limiting
                    let dv = v_ideal - op.v_out;
                    let max_dv = snapshot.slew_rate * DT;
                    op.v_out = if dv.abs() > max_dv {
                        op.v_out + max_dv * dv.signum()
                    } else {
                        v_ideal
                    };
                    op.v_out
                } else {
                    0.0
                };

                // Drive output node
                if let Some(v) = world.get_mut::<Voltage>(snapshot.node_out) {
                    v.0 = v_out_val;
                }
            }
        }

        // ── Waveform sampling ────────────────────────────────────────
        if step % SAMPLE_INTERVAL == 0 {
            let t = step as f64 * DT;
            waveform.times.push(t);
            for (i, node) in probe_nodes.iter().enumerate() {
                waveform.samples[i].push(read_voltage(&world, *node));
            }
        }

        // ── Progress report ──────────────────────────────────────────
        if step > 0 && step % REPORT_INTERVAL == 0 {
            let t_ms = step as f64 * DT * 1e3;
            let v_555 = read_voltage(&world, node_555_out);
            let v_filt = read_voltage(&world, node_filter);
            let v_out = read_voltage(&world, node_opamp_out);
            println!(
                "  t={:.2}ms | 555={:+.2}V | filter={:+.3}V | opamp={:+.3}V",
                t_ms, v_555, v_filt, v_out
            );
        }
    }

    let elapsed = start.elapsed();

    // ── Waveform analysis ────────────────────────────────────────────
    println!();
    println!(
        "Simulation complete in {:.1}ms",
        elapsed.as_secs_f64() * 1e3
    );
    println!();

    // Analyze steady-state (last 25% of simulation)
    let n_samples = waveform.times.len();
    let steady_start = n_samples * 3 / 4;

    for (i, name) in probe_names.iter().enumerate() {
        let steady = &waveform.samples[i][steady_start..];
        let min = steady.iter().copied().fold(f64::INFINITY, f64::min);
        let max = steady.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = steady.iter().sum::<f64>() / steady.len() as f64;
        let rms = (steady.iter().map(|v| v * v).sum::<f64>() / steady.len() as f64).sqrt();
        let vpp = max - min;

        println!(
            "{:>10}: mean={:+.3}V  Vpp={:.3}V  rms={:.3}V  min={:+.3}V  max={:+.3}V",
            name, mean, vpp, rms, min, max
        );
    }

    // Estimate frequency from zero-crossings of the filter output
    let filter_steady = &waveform.samples[1][steady_start..];
    let filter_times = &waveform.times[steady_start..];
    let filter_mean = filter_steady.iter().sum::<f64>() / filter_steady.len() as f64;
    let mut zero_crossings = 0usize;
    for i in 1..filter_steady.len() {
        let a = filter_steady[i - 1] - filter_mean;
        let b = filter_steady[i] - filter_mean;
        if a * b < 0.0 {
            zero_crossings += 1;
        }
    }
    if zero_crossings >= 2 {
        let duration = filter_times.last().unwrap() - filter_times[0];
        let freq = zero_crossings as f64 / (2.0 * duration);
        println!();
        println!(
            "Estimated filter output frequency: {:.0} Hz ({} zero-crossings in {:.2}ms)",
            freq,
            zero_crossings,
            duration * 1e3
        );
    }

    // Print ASCII waveform of filter output (last 2ms)
    println!();
    println!("Filter output waveform (last 2ms):");
    print_ascii_waveform(&waveform, 1, 2e-3);

    // Print final entity count
    registry.run(&mut world, census_id, ()).unwrap();
}

// ── ASCII waveform plotter ───────────────────────────────────────────

#[allow(clippy::cast_possible_wrap)]
fn print_ascii_waveform(waveform: &WaveformData, probe_idx: usize, window_secs: f64) {
    let total_time = *waveform.times.last().unwrap_or(&0.0);
    let start_time = (total_time - window_secs).max(0.0);

    let samples: Vec<(f64, f64)> = waveform
        .times
        .iter()
        .zip(waveform.samples[probe_idx].iter())
        .filter(|(t, _)| **t >= start_time)
        .map(|(t, v)| (*t, *v))
        .collect();

    if samples.is_empty() {
        println!("  (no data)");
        return;
    }

    let v_min = samples
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::INFINITY, f64::min);
    let v_max = samples
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);
    let v_range = (v_max - v_min).max(1e-6);

    let width = 72;
    let height = 16;
    let mut grid = vec![vec![' '; width]; height];

    // Draw zero line if in range
    let zero_row = ((0.0 - v_min) / v_range * (height - 1) as f64) as i32;
    if (0..height as i32).contains(&zero_row) {
        let row = (height - 1) - zero_row as usize;
        for col in &mut grid[row] {
            *col = '·';
        }
    }

    // Plot samples
    for (t, v) in &samples {
        let col = ((*t - start_time) / window_secs * (width - 1) as f64) as usize;
        let row_f = ((*v - v_min) / v_range * (height - 1) as f64) as usize;
        let row = (height - 1) - row_f.min(height - 1);
        let col = col.min(width - 1);
        grid[row][col] = '█';
    }

    // Print with voltage axis
    for (i, row) in grid.iter().enumerate() {
        let v = v_max - (i as f64 / (height - 1) as f64) * v_range;
        let line: String = row.iter().collect();
        println!("  {:+6.2}V │{}", v, line);
    }
    println!("         └{}", "─".repeat(width));
    println!(
        "          {:.1}ms{:>width$}",
        start_time * 1e3,
        format!("{:.1}ms", total_time * 1e3),
        width = width - 6
    );
}
