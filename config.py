"""
config.py
Material properties and default process-window settings for the
plasma etching digital twin.

Values are representative/simplified for simulation purposes (not
measured lab data) — they are chosen to produce physically sensible
trends (Si etches fastest under fluorine chemistry, SiO2 slower,
photoresist has its own erosion rate that sets selectivity limits).
"""

# --- Material library -------------------------------------------------
# k        : rate constant, nm/min at reference conditions
# Ea       : activation-energy-like term (eV) controlling ion-energy sensitivity
# alpha    : RF power exponent
# beta     : pressure exponent (higher pressure -> more chemical, less physical)
# gamma    : gas flow exponent
# aniso_k  : controls how quickly sidewall (lateral) etching is suppressed
#            as ion energy increases (higher = more anisotropic at lower energy)
MATERIALS = {
    "Si": {
        "label": "Silicon",
        "k": 180.0,
        "Ea": 0.35,
        "alpha": 0.55,
        "beta": 0.30,
        "gamma": 0.15,
        "aniso_k": 0.020,
    },
    "SiO2": {
        "label": "Silicon Dioxide",
        "k": 60.0,
        "Ea": 0.55,
        "alpha": 0.65,
        "beta": 0.20,
        "gamma": 0.10,
        "aniso_k": 0.014,
    },
    "PR": {
        "label": "Photoresist",
        "k": 90.0,
        "Ea": 0.25,
        "alpha": 0.40,
        "beta": 0.45,
        "gamma": 0.20,
        "aniso_k": 0.008,
    },
}

# --- Reference (normalizing) process conditions ------------------------
REFERENCE = {
    "rf_power": 150.0,   # W
    "pressure": 20.0,    # mTorr
    "gas_flow": 40.0,    # sccm
    "ion_energy": 100.0,  # eV (nominal, derived from RF power/pressure in sensor_emulator)
}

# --- Allowed process window (used for sensitivity sweeps & ML training) --
PROCESS_BOUNDS = {
    "rf_power": (50.0, 400.0),   # W
    "pressure": (5.0, 100.0),    # mTorr
    "gas_flow": (10.0, 100.0),   # sccm
}

# Boltzmann-like constant used in the Arrhenius-style ion-energy term.
# (Not physical k_B — scaled so Ea in eV gives a reasonable sensitivity
# over the ion-energy range produced by sensor_emulator.py)
K_SCALE = 0.0259  # eV, ~ thermal voltage at 300K, used as a convenient scale
