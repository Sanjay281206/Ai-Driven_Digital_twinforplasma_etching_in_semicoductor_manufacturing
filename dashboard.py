"""
dashboard.py
Control & Feedback Layer — a fuller "operator HMI" style dashboard for
the plasma etching digital twin: live process controls, live sensor
readout, a system-architecture / hardware explainer, and a panel
showing the trained AI models' live performance + a recipe recommender.

Run with:  python dashboard.py
Then open http://127.0.0.1:8050 in a browser.
"""

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import numpy as np

from config import MATERIALS, PROCESS_BOUNDS
from calculate_etch_depth import etch_depth_over_time, profile_2d, etch_rate, trench_height_map
from sensor_emulator import generate_sensor_stream, derive_ion_energy
from ml_predictor import train_models, recommend_parameters

# ---------------------------------------------------------------------
# Simple in-memory cache so we don't retrain the ML models on every
# single slider drag — only when the selected material changes.
# ---------------------------------------------------------------------
_model_cache = {}


def get_models(material):
    if material not in _model_cache:
        results, _ = train_models(material)
        best_name = max(results, key=lambda k: results[k]["r2"])
        _model_cache[material] = (results, best_name)
    return _model_cache[material]


# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------
COLORS = {
    "bg": "#0f1620",
    "panel": "#161f2c",
    "panel_border": "#26323f",
    "text": "#6648BB",
    "muted": "#8b98a5",
    "accent": "#2dd4bf",
    "accent2": "#60a5fa",
    "warn": "#f59e0b",
}

CARD_STYLE = {
    "backgroundColor": COLORS["panel"],
    "border": f"1px solid {COLORS['panel_border']}",
    "borderRadius": "10px",
    "padding": "20px",
    "marginBottom": "20px",
}

PLOTLY_TEMPLATE = "plotly_dark"


def graph_layout_defaults(fig, title):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        paper_bgcolor=COLORS["panel"],
        plot_bgcolor=COLORS["panel"],
        margin=dict(l=50, r=20, t=50, b=40),
        font=dict(color=COLORS["text"], family="Inter, system-ui, sans-serif"),
    )
    return fig


app = Dash(__name__)
app.title = "Plasma Etching Digital Twin"

# Custom page shell so we can load a nicer font + set the page background
# (Dash serves this index_string once; app.layout below is what mounts
# into the %app_entry% slot).
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
            body { background-color: BGCOLOR; margin: 0; font-family: 'Inter', system-ui, sans-serif; }
            .mono { font-family: 'JetBrains Mono', monospace; }
            ::-webkit-scrollbar { width: 10px; }
            ::-webkit-scrollbar-thumb { background: BORDERCOLOR; border-radius: 6px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
""".replace("BGCOLOR", COLORS["bg"]).replace("BORDERCOLOR", COLORS["panel_border"])


def section_title(text, subtitle=None):
    children = [html.H3(text, style={"color": COLORS["text"], "marginBottom": "4px"})]
    if subtitle:
        children.append(html.P(subtitle, style={"color": COLORS["muted"], "marginTop": 0,
                                                  "fontSize": "14px"}))
    return html.Div(children)


def labeled_slider(label, slider_id, bounds, value):
    lo, hi = bounds
    return html.Div([
        html.Label(label, style={"color": COLORS["muted"], "fontSize": "13px"}),
        dcc.Slider(lo, hi, id=slider_id, value=value, marks=None,
                   tooltip={"placement": "bottom", "always_visible": False}),
    ], style={"marginBottom": "18px"})


def stat_box_children(label, value, unit):
    return html.Div([
        html.Div(label, style={"color": COLORS["muted"], "fontSize": "12px",
                                "textTransform": "uppercase", "letterSpacing": "0.5px"}),
        html.Div([
            html.Span(value, className="mono",
                       style={"fontSize": "24px", "color": COLORS["accent"], "fontWeight": 600}),
            html.Span(f" {unit}", style={"color": COLORS["muted"], "fontSize": "13px"}),
        ]),
    ], style={"minWidth": "130px"})


# =======================================================================
# Layout
# =======================================================================
app.layout = html.Div(style={"maxWidth": "1400px", "margin": "0 auto", "padding": "28px",
                              "color": COLORS["text"]}, children=[

    # ---- Header -------------------------------------------------------
    html.Div([
        html.H1("AI-Driven Digital Twin — Plasma Etching Chamber",
                style={"marginBottom": "6px", "fontSize": "30px"}),
        html.P(
            "A simulated reactive-ion etch (RIE) process: a physics-based simulation engine "
            "stands in for the physical chamber, synthetic sensors emulate real-time chamber "
            "feedback, and two machine-learning models learn the process well enough to "
            "recommend recipes for a target etch depth — the same closed loop a real fab "
            "control system would run, just without the physical hardware attached.",
            style={"color": COLORS["muted"], "maxWidth": "900px", "lineHeight": "1.5"}
        ),
    ], style={"marginBottom": "10px"}),

    # ---- Controls + live graphs ---------------------------------------
    html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}, children=[

        html.Div(style={**CARD_STYLE, "flex": "1", "minWidth": "300px"}, children=[
            section_title("Process Controls", "Drive the simulated chamber"),
            html.Label("Material", style={"color": COLORS["muted"], "fontSize": "13px"}),
            dcc.Dropdown(
                id="material",
                options=[{"label": v["label"], "value": k} for k, v in MATERIALS.items()],
                value="Si", clearable=False,
                style={"marginBottom": "16px", "color": "#000"},
            ),
            labeled_slider("RF Power (W)", "rf_power", PROCESS_BOUNDS["rf_power"], 150),
            labeled_slider("Pressure (mTorr)", "pressure", PROCESS_BOUNDS["pressure"], 20),
            labeled_slider("Gas Flow (sccm)", "gas_flow", PROCESS_BOUNDS["gas_flow"], 40),
            labeled_slider("Etch Time (min)", "etch_time", (0.5, 10), 3),
        ]),

        html.Div(style={**CARD_STYLE, "flex": "2", "minWidth": "420px"}, children=[
            section_title("Live Simulation Output"),
            html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                dcc.Graph(id="depth-graph", style={"flex": "1", "minWidth": "320px"}),
                dcc.Graph(id="profile-graph", style={"flex": "1", "minWidth": "320px"}),
            ]),
        ]),
    ]),

    # ---- Live "sensor" readout -----------------------------------------
    html.Div(style=CARD_STYLE, children=[
        section_title("Live Chamber Sensor Feed (simulated)",
                       "Synthetic Data Acquisition Layer — stands in for real chamber telemetry"),
        html.Div(id="sensor-readout",
                 style={"display": "flex", "gap": "24px", "flexWrap": "wrap", "marginTop": "10px"}),
    ]),

    # ---- 3D visualization ------------------------------------------------
    html.Div(style=CARD_STYLE, children=[
        section_title("3D Etch Visualization",
                       "How the trench geometry and etch rate respond to your process settings — drag to rotate"),
        html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginTop": "10px"}, children=[
            dcc.Graph(id="trench-3d-graph", style={"flex": "1", "minWidth": "420px", "height": "480px"}),
            dcc.Graph(id="response-surface-3d-graph", style={"flex": "1", "minWidth": "420px", "height": "480px"}),
        ]),
        html.P(
            "Left: the actual trench cross-section extruded into 3D, showing sidewall slope "
            "(anisotropy) at your current settings. Right: etch rate across the full RF power / "
            "pressure range (gas flow held at your current slider value) — the marker shows exactly "
            "where your current settings sit on that landscape.",
            style={"color": COLORS["muted"], "fontSize": "13px", "marginTop": "8px"}
        ),
    ]),

    # ---- AI model performance + recommender ----------------------------
    html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}, children=[
        html.Div(style={**CARD_STYLE, "flex": "1", "minWidth": "320px"}, children=[
            section_title("AI Model Performance",
                           "Trained live on simulated process data for the selected material"),
            html.Div(id="model-performance", style={"marginTop": "10px"}),
        ]),
        html.Div(style={**CARD_STYLE, "flex": "1", "minWidth": "320px"}, children=[
            section_title("Recipe Recommender",
                           "Ask the trained model for a process recipe hitting a target depth"),
            html.Label("Target etch depth (nm)", style={"color": COLORS["muted"], "fontSize": "13px"}),
            dcc.Input(id="target-depth", type="number", value=500, min=10, max=5000,
                      style={"width": "100%", "padding": "8px", "marginBottom": "12px",
                             "borderRadius": "6px", "border": f"1px solid {COLORS['panel_border']}",
                             "backgroundColor": "#0b121a", "color": COLORS["text"]}),
            html.Button("Recommend Parameters", id="recommend-btn", n_clicks=0,
                        style={"backgroundColor": COLORS["accent"], "color": "#04201c",
                               "border": "none", "borderRadius": "6px", "padding": "10px 16px",
                               "fontWeight": 600, "cursor": "pointer"}),
            html.Div(id="recommendation-output", style={"marginTop": "16px"}),
        ]),
    ]),

    # ---- System architecture / hardware explainer -----------------------
    html.Div(style=CARD_STYLE, children=[
        section_title("System Architecture & Real-World Hardware Equivalents",
                       "What each layer of this digital twin would connect to on an actual etch tool"),
        html.Div(style={"display": "grid",
                         "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
                         "gap": "18px", "marginTop": "12px"}, children=[

            html.Div([
                html.Div("1. Data Acquisition Layer", style={"color": COLORS["accent2"], "fontWeight": 600,
                                                               "marginBottom": "6px"}),
                html.Ul([
                    html.Li("RF power meter / V-I probe on the RF match network"),
                    html.Li("Capacitance manometer for chamber pressure"),
                    html.Li("Mass flow controllers (MFCs) for each process gas"),
                    html.Li("Optical Emission Spectrometer (OES) for endpoint detection"),
                    html.Li([html.Em("Here: "), "sensor_emulator.py generates synthetic versions of all of these"]),
                ], style={"color": COLORS["muted"], "fontSize": "13px", "paddingLeft": "18px"}),
            ]),

            html.Div([
                html.Div("2. Digital Twin Core", style={"color": COLORS["accent2"], "fontWeight": 600,
                                                          "marginBottom": "6px"}),
                html.Ul([
                    html.Li("Physics-based etch-rate model (Arrhenius-style rate law)"),
                    html.Li("Depends on RF power, pressure, gas flow, and derived ion energy"),
                    html.Li("2D trench cross-section model (depth + sidewall anisotropy)"),
                    html.Li([html.Em("Here: "), "calculate_etch_depth.py"]),
                ], style={"color": COLORS["muted"], "fontSize": "13px", "paddingLeft": "18px"}),
            ]),

            html.Div([
                html.Div("3. AI Models", style={"color": COLORS["accent2"], "fontWeight": 600,
                                                 "marginBottom": "6px"}),
                html.Ul([
                    html.Li("Linear Regression — simple, interpretable baseline"),
                    html.Li("Random Forest — captures nonlinear power/pressure/flow interactions"),
                    html.Li("Trained on simulated process logs (1500 samples per material)"),
                    html.Li([html.Em("Here: "), "ml_predictor.py"]),
                ], style={"color": COLORS["muted"], "fontSize": "13px", "paddingLeft": "18px"}),
            ]),

            html.Div([
                html.Div("4. Control & Feedback Layer", style={"color": COLORS["accent2"], "fontWeight": 600,
                                                                 "marginBottom": "6px"}),
                html.Ul([
                    html.Li("This live dashboard — the operator-facing HMI"),
                    html.Li("Recipe recommendation for a target depth"),
                    html.Li("In a real tool: would write setpoints back to the RF generator / MFCs"),
                    html.Li([html.Em("Here: "), "dashboard.py (this page)"]),
                ], style={"color": COLORS["muted"], "fontSize": "13px", "paddingLeft": "18px"}),
            ]),
        ]),
    ]),

    html.Div(
        "Simulation model for demonstration/learning purposes — rate constants are illustrative, "
        "not measured fab data.",
        style={"color": COLORS["muted"], "fontSize": "12px", "textAlign": "center",
               "marginTop": "10px", "paddingBottom": "20px"}
    ),
])


# =======================================================================
# Callbacks
# =======================================================================

@app.callback(
    Output("depth-graph", "figure"),
    Output("profile-graph", "figure"),
    Output("sensor-readout", "children"),
    Output("trench-3d-graph", "figure"),
    Output("response-surface-3d-graph", "figure"),
    Input("material", "value"),
    Input("rf_power", "value"),
    Input("pressure", "value"),
    Input("gas_flow", "value"),
    Input("etch_time", "value"),
)
def update_simulation(material, rf_power, pressure, gas_flow, etch_time):
    # --- depth vs time ---
    t = np.linspace(0, max(5, etch_time * 1.5), 100)
    depth, rate = etch_depth_over_time(material, rf_power, pressure, gas_flow, t)

    depth_fig = go.Figure()
    depth_fig.add_trace(go.Scatter(x=t, y=depth, mode="lines", name="Etch depth",
                                     line=dict(color=COLORS["accent"], width=3)))
    graph_layout_defaults(depth_fig, f"Etch depth vs. time ({rate:.0f} nm/min)")
    depth_fig.update_layout(xaxis_title="Time (min)", yaxis_title="Depth (nm)")

    # --- 2D trench profile ---
    x, y, depth_um, lateral_um, aniso = profile_2d(material, rf_power, pressure, gas_flow, etch_time)
    profile_fig = go.Figure()
    profile_fig.add_trace(go.Scatter(x=x, y=y, fill="toself", mode="lines",
                                       line=dict(color=COLORS["accent2"]),
                                       fillcolor="rgba(96,165,250,0.35)", name="Trench"))
    graph_layout_defaults(
        profile_fig,
        f"Cross-section (depth={depth_um:.3f}\u00b5m, anisotropy={aniso:.2f})"
    )
    profile_fig.update_layout(xaxis_title="Lateral (\u00b5m)", yaxis_title="Depth (\u00b5m)")

    # --- live sensor readout ---
    sensors = generate_sensor_stream(rf_power, pressure, gas_flow, duration_s=10,
                                      sample_rate_hz=5, seed=None)
    ion_energy = derive_ion_energy(rf_power, pressure)
    readout = html.Div(style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}, children=[
        stat_box_children("RF Power", f"{sensors['rf_power_w'][-1]:.1f}", "W"),
        stat_box_children("Pressure", f"{sensors['pressure_mtorr'][-1]:.1f}", "mTorr"),
        stat_box_children("Gas Flow", f"{sensors['gas_flow_sccm'][-1]:.1f}", "sccm"),
        stat_box_children("Ion Energy", f"{ion_energy:.1f}", "eV"),
        stat_box_children("OES Intensity", f"{sensors['oes_intensity'][-1]:.2f}", "a.u."),
    ])

    # --- 3D trench visualization (extruded cross-section) ---
    tx, ty, tZ, t_depth_um, t_lateral_um, t_aniso = trench_height_map(
        material, rf_power, pressure, gas_flow, etch_time
    )
    trench_3d_fig = go.Figure(data=[go.Surface(
        x=tx, y=ty, z=tZ, colorscale="Tealgrn", showscale=False,
        contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
    )])
    graph_layout_defaults(
        trench_3d_fig,
        f"3D trench profile (depth={t_depth_um:.3f}\u00b5m, anisotropy={t_aniso:.2f})"
    )
    trench_3d_fig.update_layout(
        scene=dict(
            xaxis_title="Lateral (\u00b5m)", yaxis_title="Along trench (\u00b5m)",
            zaxis_title="Depth (\u00b5m)",
            xaxis=dict(backgroundcolor=COLORS["panel"], gridcolor=COLORS["panel_border"]),
            yaxis=dict(backgroundcolor=COLORS["panel"], gridcolor=COLORS["panel_border"]),
            zaxis=dict(backgroundcolor=COLORS["panel"], gridcolor=COLORS["panel_border"]),
            aspectmode="data",
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    )

    # --- 3D response surface: etch rate vs RF power & pressure ---
    rf_vals = np.linspace(*PROCESS_BOUNDS["rf_power"], 35)
    p_vals = np.linspace(*PROCESS_BOUNDS["pressure"], 35)
    RF, P = np.meshgrid(rf_vals, p_vals)
    rate_grid = etch_rate(material, RF, P, gas_flow)

    response_fig = go.Figure(data=[go.Surface(
        x=rf_vals, y=p_vals, z=rate_grid, colorscale="Viridis", showscale=True,
        colorbar=dict(title="nm/min", tickfont=dict(color=COLORS["text"])),
        opacity=0.92,
    )])
    current_rate = etch_rate(material, rf_power, pressure, gas_flow)
    response_fig.add_trace(go.Scatter3d(
        x=[rf_power], y=[pressure], z=[current_rate],
        mode="markers", marker=dict(size=6, color=COLORS["warn"]),
        name="Current setting",
    ))
    graph_layout_defaults(response_fig, f"Etch rate landscape (gas flow={gas_flow:.0f} sccm)")
    response_fig.update_layout(
        scene=dict(
            xaxis_title="RF Power (W)", yaxis_title="Pressure (mTorr)",
            zaxis_title="Etch rate (nm/min)",
            xaxis=dict(backgroundcolor=COLORS["panel"], gridcolor=COLORS["panel_border"]),
            yaxis=dict(backgroundcolor=COLORS["panel"], gridcolor=COLORS["panel_border"]),
            zaxis=dict(backgroundcolor=COLORS["panel"], gridcolor=COLORS["panel_border"]),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )

    return depth_fig, profile_fig, readout, trench_3d_fig, response_fig


@app.callback(
    Output("model-performance", "children"),
    Input("material", "value"),
)
def update_model_performance(material):
    results, best_name = get_models(material)
    rows = []
    for name, r in results.items():
        is_best = " \u2b50" if name == best_name else ""
        rows.append(html.Div([
            html.Span(f"{name}{is_best}", style={"color": COLORS["text"], "fontWeight": 600}),
            html.Span(f"  MAE: {r['mae']:.2f} nm/min   R\u00b2: {r['r2']:.4f}",
                       className="mono", style={"color": COLORS["muted"], "marginLeft": "10px"}),
        ], style={"marginBottom": "8px", "fontSize": "14px"}))
    rows.append(html.P(
        f"Best model for {MATERIALS[material]['label']}: {best_name} "
        "(used by the Recipe Recommender on the right).",
        style={"color": COLORS["muted"], "fontSize": "13px", "marginTop": "10px"}
    ))
    return rows


@app.callback(
    Output("recommendation-output", "children"),
    Input("recommend-btn", "n_clicks"),
    State("material", "value"),
    State("etch_time", "value"),
    State("target-depth", "value"),
    prevent_initial_call=True,
)
def update_recommendation(n_clicks, material, etch_time, target_depth):
    if not target_depth or target_depth <= 0:
        return html.Div("Enter a positive target depth.", style={"color": COLORS["warn"]})

    results, best_name = get_models(material)
    best_model = results[best_name]["model"]

    rec = recommend_parameters(material, target_depth, etch_time, best_model)

    return html.Div([
        html.P(f"Recommended recipe for {target_depth:.0f} nm in {etch_time:.1f} min "
               f"({MATERIALS[material]['label']}):",
               style={"color": COLORS["text"], "marginBottom": "8px"}),
        html.Div([
            html.Div(f"RF Power:  {rec['rf_power']:.1f} W", className="mono"),
            html.Div(f"Pressure:  {rec['pressure']:.1f} mTorr", className="mono"),
            html.Div(f"Gas Flow:  {rec['gas_flow']:.1f} sccm", className="mono"),
        ], style={"color": COLORS["accent"], "fontSize": "15px", "lineHeight": "1.8"}),
        html.P(f"Predicted result: {rec['predicted_depth_nm']:.1f} nm "
               f"({rec['predicted_rate_nm_min']:.1f} nm/min)",
               style={"color": COLORS["muted"], "fontSize": "13px", "marginTop": "8px"}),
    ])


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
