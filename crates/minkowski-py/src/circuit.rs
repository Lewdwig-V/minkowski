//! Self-contained 555→LCR→741 circuit simulator exposed as a Python class.
//!
//! The circuit topology is hardcoded. Python controls simulation stepping and
//! retrieves waveform data as a Polars DataFrame via Arrow.

use crate::pyworld::PyWorld;
use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3_arrow::PyRecordBatch;
use std::sync::Arc;

// ── 555 Timer ────────────────────────────────────────────────────────

struct Timer555 {
    r1: f64,
    r2: f64,
    cap: f64,
    v_cap: f64,
    output_high: bool,
    vcc: f64,
}

impl Timer555 {
    fn new(r1: f64, r2: f64, cap: f64, vcc: f64) -> Self {
        Self {
            r1,
            r2,
            cap,
            v_cap: 0.0,
            output_high: true,
            vcc,
        }
    }

    fn step(&mut self, dt: f64) -> f64 {
        let thresh_hi = self.vcc * (2.0 / 3.0);
        let thresh_lo = self.vcc * (1.0 / 3.0);

        if self.output_high {
            let r_charge = self.r1 + self.r2;
            let i = (self.vcc - self.v_cap) / r_charge;
            self.v_cap += i * dt / self.cap;
            if self.v_cap >= thresh_hi {
                self.output_high = false;
                self.v_cap = thresh_hi;
            }
        } else {
            let i = self.v_cap / self.r2;
            self.v_cap -= i * dt / self.cap;
            if self.v_cap <= thresh_lo {
                self.output_high = true;
                self.v_cap = thresh_lo;
            }
        }

        if self.output_high {
            self.vcc - 1.5
        } else {
            0.2
        }
    }
}

// ── Circuit nodes and elements ───────────────────────────────────────

struct Node {
    voltage: f64,
    current_sum: f64,
    capacitance: f64,
    is_ground: bool,
    is_driven: bool, // 555 output: voltage set directly
}

struct Resistor {
    resistance: f64,
    node_a: usize,
    node_b: usize,
}

struct Inductor {
    inductance: f64,
    current: f64,
    node_a: usize,
    node_b: usize,
}

struct OpAmp {
    open_loop_gain: f64,
    slew_rate: f64,
    v_sat_pos: f64,
    v_sat_neg: f64,
    v_out: f64,
    node_plus: usize,
    node_minus: usize,
    node_out: usize,
}

// ── Circuit topology ─────────────────────────────────────────────────

struct CircuitTopology {
    timer: Timer555,
    nodes: Vec<Node>,
    resistors: Vec<Resistor>,
    inductors: Vec<Inductor>,
    opamps: Vec<OpAmp>,
    idx_555_out: usize,
    idx_filter: usize,
    idx_opamp_out: usize,
}

// ── CircuitSim pyclass ───────────────────────────────────────────────

#[pyclass(name = "CircuitSim")]
pub struct PyCircuitSim {
    dt: f64,
    sample_every: usize,
    step_count: usize,
    timer: Timer555,
    nodes: Vec<Node>,
    resistors: Vec<Resistor>,
    inductors: Vec<Inductor>,
    opamps: Vec<OpAmp>,
    idx_555_out: usize,
    idx_filter: usize,
    idx_opamp_out: usize,
    // Waveform buffers
    times: Vec<f64>,
    v_555: Vec<f64>,
    v_filter: Vec<f64>,
    v_opamp: Vec<f64>,
}

impl PyCircuitSim {
    fn build_circuit(vcc: f64) -> CircuitTopology {
        let timer = Timer555::new(10e3, 10e3, 10e-9, vcc);

        // Node indices: 0=GND, 1=555_out, 2=L_in, 3=filter, 4=opamp_out
        let nodes = vec![
            Node {
                voltage: 0.0,
                current_sum: 0.0,
                capacitance: 1.0,
                is_ground: true,
                is_driven: false,
            },
            Node {
                voltage: 0.0,
                current_sum: 0.0,
                capacitance: 1e-10,
                is_ground: false,
                is_driven: true,
            },
            Node {
                voltage: 0.0,
                current_sum: 0.0,
                capacitance: 1e-10,
                is_ground: false,
                is_driven: false,
            },
            Node {
                voltage: 0.0,
                current_sum: 0.0,
                capacitance: 100e-9,
                is_ground: false,
                is_driven: false,
            },
            Node {
                voltage: 0.0,
                current_sum: 0.0,
                capacitance: 1e-10,
                is_ground: false,
                is_driven: false,
            },
        ];

        let resistors = vec![
            Resistor {
                resistance: 1e3,
                node_a: 1,
                node_b: 2,
            }, // R_src
            Resistor {
                resistance: 10e3,
                node_a: 3,
                node_b: 0,
            }, // R_load
        ];

        let inductors = vec![Inductor {
            inductance: 10e-3,
            current: 0.0,
            node_a: 2,
            node_b: 3,
        }];

        let opamps = vec![OpAmp {
            open_loop_gain: 200_000.0,
            slew_rate: 0.5e6,
            v_sat_pos: vcc - 2.0,
            v_sat_neg: -vcc + 2.0,
            v_out: 0.0,
            node_plus: 3,
            node_minus: 4,
            node_out: 4,
        }];

        // Node indices: 0=GND, 1=555_out, 2=L_in, 3=filter, 4=opamp_out
        let idx_555_out = 1;
        let idx_filter = 3;
        let idx_opamp_out = 4;
        debug_assert!(idx_555_out < nodes.len());
        debug_assert!(idx_filter < nodes.len());
        debug_assert!(idx_opamp_out < nodes.len());

        CircuitTopology {
            timer,
            nodes,
            resistors,
            inductors,
            opamps,
            idx_555_out,
            idx_filter,
            idx_opamp_out,
        }
    }

    fn simulate_step(&mut self) {
        let dt = self.dt;

        // Phase 1: Reset current sums
        for node in &mut self.nodes {
            node.current_sum = 0.0;
        }

        // Phase 2: 555 timer
        let v_out = self.timer.step(dt);
        self.nodes[self.idx_555_out].voltage = v_out;

        // Phase 3: Resistor currents
        for r in &self.resistors {
            let va = self.nodes[r.node_a].voltage;
            let vb = self.nodes[r.node_b].voltage;
            let current = (va - vb) / r.resistance;
            self.nodes[r.node_a].current_sum -= current;
            self.nodes[r.node_b].current_sum += current;
        }

        // Phase 4: Inductor update (symplectic Euler)
        for ind in &mut self.inductors {
            let va = self.nodes[ind.node_a].voltage;
            let vb = self.nodes[ind.node_b].voltage;
            ind.current += (va - vb) / ind.inductance * dt;
            self.nodes[ind.node_a].current_sum -= ind.current;
            self.nodes[ind.node_b].current_sum += ind.current;
        }

        // Phase 5: Update node voltages (dV/dt = I/C)
        for node in &mut self.nodes {
            if node.is_ground || node.is_driven {
                continue;
            }
            node.voltage += node.current_sum / node.capacitance * dt;
        }

        // Phase 6: Op-amp update
        for op in &mut self.opamps {
            let v_plus = self.nodes[op.node_plus].voltage;
            let v_minus = self.nodes[op.node_minus].voltage;
            let v_diff = v_plus - v_minus;
            let v_ideal = (op.open_loop_gain * v_diff).clamp(op.v_sat_neg, op.v_sat_pos);
            let dv = v_ideal - op.v_out;
            let max_dv = op.slew_rate * dt;
            op.v_out = if dv.abs() > max_dv {
                op.v_out + max_dv * dv.signum()
            } else {
                v_ideal
            };
            self.nodes[op.node_out].voltage = op.v_out;
        }

        self.step_count += 1;
    }
}

#[pymethods]
impl PyCircuitSim {
    /// Create a new circuit simulator.
    ///
    /// The circuit is a hardcoded 555 astable → LCR bandpass → 741 follower.
    /// `vcc` sets the supply voltage, `dt` the timestep in seconds,
    /// `sample_every` how many steps between waveform samples.
    ///
    /// The `_world` parameter is unused but required for API consistency with
    /// other Minkowski Python classes (SpatialGrid, ReducerRegistry).
    #[new]
    #[pyo3(signature = (_world, vcc=12.0, dt=1e-7, sample_every=20))]
    fn new(_world: &mut PyWorld, vcc: f64, dt: f64, sample_every: usize) -> PyResult<Self> {
        if !vcc.is_finite() || vcc <= 0.0 {
            return Err(PyValueError::new_err("vcc must be finite and positive"));
        }
        if !dt.is_finite() || dt <= 0.0 {
            return Err(PyValueError::new_err("dt must be finite and positive"));
        }
        if sample_every == 0 {
            return Err(PyValueError::new_err("sample_every must be >= 1"));
        }

        let topo = Self::build_circuit(vcc);

        Ok(PyCircuitSim {
            dt,
            sample_every,
            step_count: 0,
            timer: topo.timer,
            nodes: topo.nodes,
            resistors: topo.resistors,
            inductors: topo.inductors,
            opamps: topo.opamps,
            idx_555_out: topo.idx_555_out,
            idx_filter: topo.idx_filter,
            idx_opamp_out: topo.idx_opamp_out,
            times: Vec::new(),
            v_555: Vec::new(),
            v_filter: Vec::new(),
            v_opamp: Vec::new(),
        })
    }

    /// Run `n` simulation steps. Waveform samples are appended to the internal buffer.
    fn step(&mut self, n: usize) {
        for _ in 0..n {
            self.simulate_step();
            if self.step_count.is_multiple_of(self.sample_every) {
                self.times.push(self.step_count as f64 * self.dt);
                self.v_555.push(self.nodes[self.idx_555_out].voltage);
                self.v_filter.push(self.nodes[self.idx_filter].voltage);
                self.v_opamp.push(self.nodes[self.idx_opamp_out].voltage);
            }
        }
    }

    /// Return the accumulated waveform as a Polars DataFrame.
    fn waveform<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let schema = Schema::new(vec![
            Field::new("time", DataType::Float64, false),
            Field::new("v_555", DataType::Float64, false),
            Field::new("v_filter", DataType::Float64, false),
            Field::new("v_opamp", DataType::Float64, false),
        ]);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Float64Array::from(self.times.clone())),
                Arc::new(Float64Array::from(self.v_555.clone())),
                Arc::new(Float64Array::from(self.v_filter.clone())),
                Arc::new(Float64Array::from(self.v_opamp.clone())),
            ],
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let py_batch = PyRecordBatch::new(batch);
        let table = py_batch.into_pyarrow(py)?;
        let polars = py.import("polars")?;
        polars.getattr("DataFrame")?.call1((table,))
    }

    /// Clear the waveform buffer and reset circuit state to t=0.
    fn reset(&mut self) {
        self.times.clear();
        self.v_555.clear();
        self.v_filter.clear();
        self.v_opamp.clear();
        self.step_count = 0;

        let topo = Self::build_circuit(self.timer.vcc);
        self.timer = topo.timer;
        self.nodes = topo.nodes;
        self.resistors = topo.resistors;
        self.inductors = topo.inductors;
        self.opamps = topo.opamps;
        self.idx_555_out = topo.idx_555_out;
        self.idx_filter = topo.idx_filter;
        self.idx_opamp_out = topo.idx_opamp_out;
    }

    /// Total number of simulation steps executed since creation or last reset.
    #[getter]
    fn total_steps(&self) -> usize {
        self.step_count
    }

    /// Number of waveform samples collected.
    #[getter]
    fn sample_count(&self) -> usize {
        self.times.len()
    }
}
