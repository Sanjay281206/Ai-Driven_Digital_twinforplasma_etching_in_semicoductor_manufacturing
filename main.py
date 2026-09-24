"""
main.py
Entry point for the AI-Driven Digital Twin of a Plasma Etching Chamber.

Runs the full pipeline described in the System Architecture / Methodology
slides:
  1. Emulate real-time chamber sensor data (Data Acquisition Layer)
  2. Run the physics-based etch simulation for the requested material
     and process conditions (Digital Twin Core / Simulation Engine)
  3. Train AI models on simulated process data and use them to predict
     etch rate + recommend optimal parameters for a target depth
     (Digital Twin Core / AI Models)
  4. Visualize etch-depth-vs-time, the 2D trench cross-section, and
     parameter sensitivity (Control & Feedback Layer / Dashboard)

Usage:
    python main.py --material Si --rf-power 150 --pressure 20 --gas-flow 40 \
                    --time 3 --target-depth 500
"""

import argparse
import os

from config import MATERIALS
from sensor_emulator import generate_sensor_stream, derive_ion_energy
from calculate_etch_depth import etch_depth_over_time, profile_2d
from plot_etch_profile import plot_depth_vs_time, plot_profile_2d, plot_sensitivity
from ml_predictor import train_models, recommend_parameters

OUT_DIR = "output"


def parse_args():
    p = argparse.ArgumentParser(description="Plasma etching digital twin")
    p.add_argument("--material", choices=list(MATERIALS.keys()), default="Si")
    p.add_argument("--rf-power", type=float, default=150.0, help="RF power (W)")
    p.add_argument("--pressure", type=float, default=20.0, help="Chamber pressure (mTorr)")
    p.add_argument("--gas-flow", type=float, default=40.0, help="Reactive gas flow (sccm)")
    p.add_argument("--time", type=float, default=3.0, help="Etch time (min)")
    p.add_argument("--target-depth", type=float, default=500.0,
                    help="Target etch depth (nm) for the ML parameter recommendation")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("AI-DRIVEN DIGITAL TWIN — PLASMA ETCHING CHAMBER")
    print("=" * 60)

    # 1. Physical / Data Acquisition Layer: emulate live sensor feed
    print(f"\n[1] Emulating chamber sensors for {args.material} "
          f"(RF={args.rf_power}W, P={args.pressure}mTorr, Q={args.gas_flow}sccm)...")
    sensors = generate_sensor_stream(args.rf_power, args.pressure, args.gas_flow, seed=1)
    ion_energy = derive_ion_energy(args.rf_power, args.pressure)
    print(f"    Derived mean ion energy: {ion_energy:.1f} eV")
    print(f"    OES intensity range: {sensors['oes_intensity'].min():.2f} - "
          f"{sensors['oes_intensity'].max():.2f} (a.u.)")

    # 2. Digital Twin Core: physics-based simulation
    print(f"\n[2] Running physics-based etch simulation for {args.time} min...")
    depth_nm, rate = etch_depth_over_time(
        args.material, args.rf_power, args.pressure, args.gas_flow, [args.time]
    )
    print(f"    Etch rate: {rate:.1f} nm/min")
    print(f"    Final depth: {depth_nm[0]:.1f} nm after {args.time} min")

    x, y, depth_um, lateral_um, aniso = profile_2d(
        args.material, args.rf_power, args.pressure, args.gas_flow, args.time
    )
    print(f"    Sidewall recess: {lateral_um:.4f} um (anisotropy factor={aniso:.3f})")

    # 3. AI Models: train + recommend optimal parameters
    print(f"\n[3] Training AI models on simulated process data ({args.material})...")
    results, _ = train_models(args.material)
    for name, r in results.items():
        print(f"    {name}: MAE={r['mae']:.2f} nm/min, R2={r['r2']:.4f}")

    best_name = max(results, key=lambda k: results[k]["r2"])
    best_model = results[best_name]["model"]
    print(f"    Best model: {best_name}")

    rec = recommend_parameters(
        args.material, args.target_depth, args.time, best_model
    )
    print(f"\n    Recommended process for {args.target_depth:.0f} nm in {args.time} min:")
    print(f"      RF power : {rec['rf_power']:.1f} W")
    print(f"      Pressure : {rec['pressure']:.1f} mTorr")
    print(f"      Gas flow : {rec['gas_flow']:.1f} sccm")
    print(f"      -> predicted depth: {rec['predicted_depth_nm']:.1f} nm")

    # 4. Control & Feedback Layer: visualization / "dashboard"
    print(f"\n[4] Rendering visualizations to ./{OUT_DIR}/ ...")
    plot_depth_vs_time(args.rf_power, args.pressure, args.gas_flow,
                        max_time_min=max(5, args.time * 1.5),
                        savepath=f"{OUT_DIR}/depth_vs_time.png")
    plot_profile_2d(args.material, args.rf_power, args.pressure, args.gas_flow,
                     args.time, savepath=f"{OUT_DIR}/trench_profile.png")
    plot_sensitivity(args.material, savepath=f"{OUT_DIR}/sensitivity.png")
    print("    Saved: depth_vs_time.png, trench_profile.png, sensitivity.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
