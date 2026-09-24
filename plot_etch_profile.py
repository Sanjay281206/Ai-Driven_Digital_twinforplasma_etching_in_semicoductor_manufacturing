"""
plot_etch_profile.py
Visualization layer: etch depth vs time, 2D cross-sectional trench
profile, and parameter-sensitivity sweeps (pressure / gas flow / RF
power vs etch rate) as called for in the Key Features slide.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MATERIALS, PROCESS_BOUNDS, REFERENCE
from calculate_etch_depth import etch_depth_over_time, profile_2d, etch_rate


def plot_depth_vs_time(rf_power, pressure, gas_flow, max_time_min=5,
                        materials=("Si", "SiO2", "PR"), savepath=None):
    t = np.linspace(0, max_time_min, 100)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for mat in materials:
        depth, rate = etch_depth_over_time(mat, rf_power, pressure, gas_flow, t)
        ax.plot(t, depth, label=f"{MATERIALS[mat]['label']} ({rate:.0f} nm/min)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Etch depth (nm)")
    ax.set_title(f"Etch depth vs. time\nRF={rf_power}W, P={pressure}mTorr, Q={gas_flow}sccm")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140)
    plt.close(fig)
    return fig


def plot_profile_2d(material, rf_power, pressure, gas_flow, etch_time_min,
                     mask_open_um=1.0, savepath=None):
    x, y, depth_um, lateral_um, aniso = profile_2d(
        material, rf_power, pressure, gas_flow, etch_time_min, mask_open_um
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.fill(x, y, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Lateral position (\u00b5m)")
    ax.set_ylabel("Depth (\u00b5m)")
    ax.set_title(
        f"{MATERIALS[material]['label']} trench profile @ {etch_time_min} min\n"
        f"depth={depth_um:.3f}\u00b5m, recess={lateral_um:.3f}\u00b5m, "
        f"anisotropy={aniso:.2f}",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140)
    plt.close(fig)
    return fig


def plot_sensitivity(material="Si", savepath=None):
    """
    Sweep RF power, pressure, and gas flow independently (holding the
    other two at reference conditions) and plot etch rate response —
    the "Parameter Sensitivity" feature from the slides.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sweeps = [
        ("rf_power", "RF Power (W)"),
        ("pressure", "Pressure (mTorr)"),
        ("gas_flow", "Gas Flow (sccm)"),
    ]
    for ax, (param, label) in zip(axes, sweeps):
        lo, hi = PROCESS_BOUNDS[param]
        xs = np.linspace(lo, hi, 60)
        rates = []
        for x in xs:
            kwargs = {
                "rf_power": REFERENCE["rf_power"],
                "pressure": REFERENCE["pressure"],
                "gas_flow": REFERENCE["gas_flow"],
            }
            kwargs[param] = x
            rates.append(
                etch_rate(material, kwargs["rf_power"], kwargs["pressure"], kwargs["gas_flow"])
            )
        ax.plot(xs, rates, color="#C44E52", linewidth=2)
        ax.set_xlabel(label)
        ax.set_ylabel("Etch rate (nm/min)")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Parameter sensitivity \u2014 {MATERIALS[material]['label']}")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140)
    plt.close(fig)
    return fig
