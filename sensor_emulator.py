"""
sensor_emulator.py
Emulates real-time chamber sensor feedback so the digital twin can be
exercised without a physical tool. Mirrors the "Data Acquisition Layer"
in the system architecture (pressure, RF power, gas flow, plus an
Optical-Emission-Spectroscopy-like signal that tracks etch progress).
"""

import numpy as np


def derive_ion_energy(rf_power, pressure):
    """
    Simplified proxy for mean ion bombardment energy.
    Ion energy rises with RF power (stronger sheath) and falls with
    pressure (more collisions randomize/thermalize ions).
    """
    return 40.0 + 1.6 * rf_power / (1.0 + 0.03 * pressure)


def generate_sensor_stream(rf_power, pressure, gas_flow, duration_s=120,
                            sample_rate_hz=2, noise_level=0.02, seed=None):
    """
    Generate a synthetic real-time sensor stream for one etch run.

    Returns a dict of numpy arrays: time_s, pressure_mtorr, rf_power_w,
    gas_flow_sccm, ion_energy_ev, oes_intensity (arbitrary units, decays
    as reactant is consumed / etch approaches endpoint).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate_hz)
    t = np.linspace(0, duration_s, n)

    def noisy(value):
        return value * (1 + noise_level * rng.standard_normal(n))

    pressure_t = noisy(pressure)
    rf_power_t = noisy(rf_power)
    gas_flow_t = noisy(gas_flow)
    ion_energy_t = derive_ion_energy(rf_power_t, pressure_t)

    # OES intensity: starts high, decays toward an endpoint asymptote as
    # the film clears (classic endpoint-detection trace), plus noise.
    decay_rate = 0.02 + 0.0006 * rf_power
    oes = 0.3 + 0.7 * np.exp(-decay_rate * t)
    oes = noisy(oes)

    return {
        "time_s": t,
        "pressure_mtorr": pressure_t,
        "rf_power_w": rf_power_t,
        "gas_flow_sccm": gas_flow_t,
        "ion_energy_ev": ion_energy_t,
        "oes_intensity": oes,
    }


if __name__ == "__main__":
    data = generate_sensor_stream(rf_power=150, pressure=20, gas_flow=40, seed=1)
    print("Simulated", len(data["time_s"]), "sensor samples")
    for k, v in data.items():
        print(f"  {k}: mean={v.mean():.2f}, std={v.std():.2f}")
