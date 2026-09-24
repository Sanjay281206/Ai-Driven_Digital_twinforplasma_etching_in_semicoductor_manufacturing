# AI-Driven Digital Twin for Plasma Etching in Semiconductor Manufacturing

A working implementation of the project proposed in `project_1.pptx`: a
digital twin that simulates plasma etching (Si / SiO2 / photoresist),
emulates chamber sensor feedback, and uses machine learning to predict
etch behavior and recommend process parameters.

## How this maps to the proposal

| Slide | This project |
|---|---|
| System Architecture — Physical/Data Acquisition Layer | `sensor_emulator.py` — synthetic pressure, RF power, gas flow, and OES-style endpoint trace |
| System Architecture — Digital Twin Core (Simulation Engine) | `calculate_etch_depth.py` — physics-based (Arrhenius-style) etch rate + 2D trench profile model |
| System Architecture — AI Models (Regression) | `ml_predictor.py` — LinearRegression + RandomForest trained on simulated process data |
| Key Features — Etch Rate Calculator, Material Selector | `calculate_etch_depth.etch_rate()`, `config.MATERIALS` |
| Key Features — 2D Profile Visualization | `plot_etch_profile.plot_profile_2d()` |
| Key Features — Parameter Sensitivity | `plot_etch_profile.plot_sensitivity()` |
| Key Features — Basic AI Prediction | `ml_predictor.train_models()` / `recommend_parameters()` |
| Control & Feedback Layer — Dashboard | `dashboard.py` (optional live Dash app) |
| Project Structure Overview (slide 9) | Same idea, implemented in Python instead of MATLAB (see note below) |

**Note on MATLAB vs. Python:** the proposal's file tree (`etch_simulator_matlab/`)
sketches a MATLAB layout. This build uses Python instead, because it's
directly runnable and testable here — the module boundaries
(`config` / `calculateEtchDepth` / `plotEtchProfile` / `mlPredictor` / `main`)
are kept 1:1 with the proposal, so porting to MATLAB later is mostly a
file-by-file translation if your course requires MATLAB specifically.

## The physics model (what it actually simulates)

Etch rate follows a semi-empirical rate law common in simplified RIE
process modeling:

```
ER = k · (RF_power/RF_ref)^alpha · (pressure/p_ref)^beta · (gas_flow/Q_ref)^gamma · f(ion_energy)
```

- `k`, `alpha`, `beta`, `gamma` are material-specific constants (`config.py`) chosen
  so Si etches fastest, SiO2 slowest (realistic mask selectivity), and photoresist
  in between.
- Ion energy is derived from RF power and pressure (higher power / lower pressure
  → higher ion energy → more directional/anisotropic etching), which sets the
  sidewall slope in the 2D trench profile.
- This is a **simulation model for learning/demonstration**, not measured fab data —
  useful for showing the right qualitative trends (rate vs. power/pressure/flow,
  anisotropy vs. ion energy) that a real digital twin would calibrate against
  actual sensor logs.

## Quick start

```bash
pip install -r requirements.txt
python main.py --material Si --rf-power 150 --pressure 20 --gas-flow 40 --time 3 --target-depth 500
```

This prints the sensor emulation summary, physics-model etch rate/depth,
ML model accuracy (MAE/R²), a recommended process recipe for your target
depth, and saves three plots to `output/`:

- `depth_vs_time.png` — etch depth vs. time for all three materials
- `trench_profile.png` — 2D cross-section for the chosen material/conditions
- `sensitivity.png` — etch rate vs. RF power / pressure / gas flow

### Optional live dashboard

```bash
pip install dash plotly
python dashboard.py
```

Opens a local web app (`http://127.0.0.1:8050`) with sliders for RF power,
pressure, gas flow, and time, updating the depth curve and trench profile live.

## File structure

```
etch_digital_twin/
├── config.py               # material properties, process bounds
├── sensor_emulator.py       # synthetic chamber sensor feed (Data Acquisition Layer)
├── calculate_etch_depth.py  # physics-based etch rate + 2D profile model
├── plot_etch_profile.py     # depth-vs-time, trench cross-section, sensitivity plots
├── ml_predictor.py          # trains regression models, recommends process parameters
├── dashboard.py             # optional live Dash control panel
├── main.py                  # entry point — runs the full pipeline
├── requirements.txt
└── output/                  # generated plots (created on first run)
```

## Extending this for a stronger submission

- Swap the simplified rate law for a physically-derived model (e.g. a
  Langmuir ion/neutral flux balance) if your course wants first-principles physics.
- Replace `sensor_emulator.py` with logged data if you have lab/chamber access.
- Add an LSTM over the OES trace for real endpoint detection (slide 4 mentions
  this as a stretch AI model).
- Swap the grid-search in `recommend_parameters()` for Bayesian optimization
  for a more "AI-driven" optimizer.
