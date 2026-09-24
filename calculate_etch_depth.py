"""
calculate_etch_depth.py
Physics-based (semi-empirical) etch rate and profile model.

Model form (Arrhenius-style RIE rate law commonly used for simplified
process modeling):

    ER(nm/min) = k * (P_rf/P_ref)^alpha * (p/p_ref)^beta
                   * (Q/Q_ref)^gamma * exp(-Ea / (K_SCALE + Ea_ion_term))

Ion energy raises the rate (more physical sputtering) and also
increases anisotropy (directional etching): lateral etch rate is
suppressed relative to vertical as ion energy increases.
"""

import numpy as np

from config import MATERIALS, REFERENCE, K_SCALE
from sensor_emulator import derive_ion_energy


def etch_rate(material, rf_power, pressure, gas_flow, ion_energy=None):
    """
    Vertical etch rate (nm/min) for a given material and process point.
    """
    m = MATERIALS[material]
    if ion_energy is None:
        ion_energy = derive_ion_energy(rf_power, pressure)

    rf_term = (rf_power / REFERENCE["rf_power"]) ** m["alpha"]
    p_term = (pressure / REFERENCE["pressure"]) ** m["beta"]
    q_term = (gas_flow / REFERENCE["gas_flow"]) ** m["gamma"]

    # Normalize ion energy contribution relative to the reference point so
    # k already represents the rate AT reference conditions.
    ref_ion_energy = derive_ion_energy(REFERENCE["rf_power"], REFERENCE["pressure"])
    energy_term = np.exp(-m["Ea"] / (K_SCALE + ion_energy / 1000.0))
    ref_energy_term = np.exp(-m["Ea"] / (K_SCALE + ref_ion_energy / 1000.0))

    rate = m["k"] * rf_term * p_term * q_term * (energy_term / ref_energy_term)
    return rate


def anisotropy_factor(material, ion_energy):
    """
    Fraction of vertical rate that also proceeds laterally (0 = perfectly
    anisotropic / vertical walls, 1 = fully isotropic). Decreases with
    higher ion energy (ions arrive normal to the surface -> more directional).
    """
    m = MATERIALS[material]
    return float(np.clip(np.exp(-m["aniso_k"] * ion_energy), 0.02, 1.0))


def etch_depth_over_time(material, rf_power, pressure, gas_flow, time_min,
                          ion_energy=None):
    """
    Vectorized etch depth (nm) at each time point in time_min (array-like),
    assuming constant process conditions (steady-state rate).
    """
    time_min = np.asarray(time_min, dtype=float)
    rate = etch_rate(material, rf_power, pressure, gas_flow, ion_energy)
    return rate * time_min, rate


def profile_2d(material, rf_power, pressure, gas_flow, etch_time_min,
               mask_open_um=1.0, n_points=200, ion_energy=None):
    """
    Build a simplified 2D trench cross-section profile at the end of
    an etch step. Returns (x_um, top_y, bottom_y) arrays describing a
    trapezoidal trench: mask opening at the top, vertical depth from
    etch_rate * time, and sidewall slope set by the anisotropy factor.

    This is a simplified geometric approximation (not a full level-set /
    Monte-Carlo topography simulation), suitable for teaching/visualizing
    the anisotropic-vs-isotropic tradeoff described in the objectives.
    """
    if ion_energy is None:
        ion_energy = derive_ion_energy(rf_power, pressure)

    depth_nm, v_rate = etch_depth_over_time(
        material, rf_power, pressure, gas_flow, etch_time_min, ion_energy
    )
    depth_um = depth_nm / 1000.0
    a = anisotropy_factor(material, ion_energy)
    lateral_um = a * depth_um  # sidewall recess per side

    half_open = mask_open_um / 2.0
    # Top of trench = mask opening; bottom widens by `lateral_um` per side
    top_left, top_right = -half_open, half_open
    bottom_left, bottom_right = -half_open - lateral_um, half_open + lateral_um

    x = np.array([top_left, bottom_left, bottom_right, top_right, top_left])
    y = np.array([0, -depth_um, -depth_um, 0, 0])
    return x, y, depth_um, lateral_um, a


def trench_height_map(material, rf_power, pressure, gas_flow, etch_time_min,
                       mask_open_um=1.0, trench_length_um=2.0, nx=60, ny=20,
                       ion_energy=None):
    """
    Build a 3D height map (x, y, Z) of the trench for 3D visualization:
    a straight channel extruded along y, with the same trapezoidal
    cross-section as profile_2d (flat bottom, sloped sidewalls set by
    the anisotropy factor). Z is negative (depth below the original
    surface, 0 = original wafer surface).

    Returns (x, y, Z, depth_um, lateral_um, aniso) where x, y are 1D
    arrays and Z has shape (len(y), len(x)), suitable for plotly's
    go.Surface(x=x, y=y, z=Z).
    """
    if ion_energy is None:
        ion_energy = derive_ion_energy(rf_power, pressure)

    depth_nm, _ = etch_depth_over_time(
        material, rf_power, pressure, gas_flow, etch_time_min, ion_energy
    )
    depth_um = float(np.asarray(depth_nm)) / 1000.0
    aniso = anisotropy_factor(material, ion_energy)
    lateral_um = aniso * depth_um

    half_open = mask_open_um / 2.0
    margin = 0.3
    x_min = -(half_open + lateral_um) - margin
    x_max = (half_open + lateral_um) + margin
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(0, trench_length_um, ny)
    X, _ = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    absx = np.abs(X)
    flat_mask = absx <= half_open
    Z[flat_mask] = -depth_um
    slope_mask = (absx > half_open) & (absx <= half_open + lateral_um)
    if lateral_um > 1e-6:
        frac = (absx[slope_mask] - half_open) / lateral_um
        Z[slope_mask] = -depth_um * (1 - frac)

    return x, y, Z, depth_um, lateral_um, aniso


if __name__ == "__main__":
    d, r = etch_depth_over_time("Si", 150, 20, 40, [0, 1, 2, 3])
    print("Si etch rate (nm/min):", r)
    print("Depth over time (nm):", d)
