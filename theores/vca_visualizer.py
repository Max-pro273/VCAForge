"""
VCAForge Surface Explorer — Standalone Bayesian Optimization Visualizer
══════════════════════════════════════════════════════════════════════════
Maps DFT data to a Gaussian Process surface.
Z-axis    : μ (Predicted Mean of target property)
Color map : σ (Predicted Model Uncertainty) - Plasma scale.

Academic Visualization Note:
Encoding uncertainty as color (instead of Z-height) naturally guides
researchers to optimal next-step calculations. Bright yellow/white areas
indicate unexplored, highly uncertain regimes (high σ). Tall, dark areas
indicate confirmed optimal regions.
"""

import base64
import io
import re
from dataclasses import dataclass
from typing import Any, Tuple, List, Dict

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

# ─────────────────────────────────────────────────────────────────────────────
# 1. Core Data Models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SystemInfo:
    metals: List[str]
    nonmetal: str
    n_dims: int               # Independent concentration dimensions (N - 1)
    conc_cols: List[str]      # Explicit columns used as GP inputs
    all_conc_cols: List[str]  # All x_{E} columns including the implicit one

# ─────────────────────────────────────────────────────────────────────────────
# 2. DataLoader (Pure, No Side Effects)
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """Loads, validates, and cleans DFT results from VCAForge CSVs."""

    def load(self, file_contents: str) -> pd.DataFrame:
        """Parses base64 CSV content and applies strict scientific filters."""
        content_type, content_string = file_contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), comment='#')

        # 1. Status check
        if 'status' in df.columns:
            df = df[df['status'] == 'done']

        # 2. GP Fit Quality (R² filter)
        if 'elastic_R2_min' in df.columns:
            df['elastic_R2_min'] = pd.to_numeric(df['elastic_R2_min'], errors='coerce')
            # Relaxed filter: keep almost everything, even if fit was poor.
            # (Users want to see all calculated points)
            # df = df[df['elastic_R2_min'] >= 0.5] 
            pass

        # 3. Exclude Vegard interpolations
        if 'elastic_source' in df.columns:
            df = df[df['elastic_source'] != 'Vegard_interpolation']

        # 4. Born Stability Check
        # RELAXED: Don't drop unstable points. Keep them for visualization.
        born_cols = ['C11', 'C12', 'C44']
        if all(c in df.columns for c in born_cols):
            for c in born_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if len(df) < 1:
            raise ValueError(f"No data points survived initial loading.")

        return df

    def detect_system(self, df: pd.DataFrame) -> SystemInfo:
        """Auto-detects the alloy system from column names."""
        x_cols = sorted([c for c in df.columns if re.match(r"^x_[A-Z][a-z]?$", c)])
        if len(x_cols) < 1:
            raise ValueError("CSV must contain at least one 'x_{Element}' column.")

        metals = [c.split("_")[1] for c in x_cols]

        # Heuristic for nonmetal: look for Mulliken charges of common nonmetals
        nonmetal = ""
        for c in df.columns:
            m = re.match(r"^mulliken_q_([CNOBPSF])$", c)
            if m:
                nonmetal = m.group(1)
                break
        
        # Fallback for system names
        if not nonmetal:
             # Check if any metal is known to be non-metal in config (if we had access)
             # Here we use common sense
             for nm in ['C', 'N', 'B']:
                 if nm in metals:
                     nonmetal = nm
                     metals.remove(nm)
                     x_cols = sorted([c for c in x_cols if c != f"x_{nm}"])
                     break

        return SystemInfo(
            metals=metals,
            nonmetal=nonmetal,
            n_dims=len(metals) - 1,
            conc_cols=x_cols[1:],      # Independent variables
            all_conc_cols=x_cols       # Including the implicit dependent variable (metals[0])
        )

# ─────────────────────────────────────────────────────────────────────────────
# 3. SurfaceModel (Gaussian Process Regressor)
# ─────────────────────────────────────────────────────────────────────────────

class SurfaceModel:
    """Fits GP to DFT points, predicts smooth surface + uncertainty."""

    def __init__(self):
        self.gpr = None
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray, frozen_params: Dict = None) -> None:
        """
        Fits the GP. If frozen_params is provided, skips L-BFGS-B optimization
        and uses the cached hyper-parameters for instant prediction.
        """
        self.X_train = X
        self.y_train = y

        kernel = Matern(nu=50, length_scale_bounds=(1e-3, 5.0)) + \
                 WhiteKernel(noise_level_bounds=(1e-7, 1.0))

        if frozen_params:
            # Re-instantiate with cached parameters and disable optimizer
            kernel = kernel.clone_with_theta(frozen_params)
            self.gpr = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, random_state=42)
        else:
            # Full expensive optimization
            self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42)

        self.gpr.fit(self.X_train, self.y_train)

    def get_params(self) -> Dict:
        """Returns optimized kernel theta for caching."""
        if self.gpr and hasattr(self.gpr.kernel_, 'theta'):
            return self.gpr.kernel_.theta
        return None

    def predict_grid(
        self,
        system: SystemInfo,
        fixed_coords: Dict[str, float] = None,
        resolution: int = 60,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates dense grid over the valid simplex."""
        fixed_coords = fixed_coords or {}

        sum_fixed = sum(fixed_coords.values())
        if sum_fixed >= 0.999:
            return np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0)), np.zeros((0,0))

        avail_budget = 1.0 - sum_fixed

        # Grid is always mapped to the first two independent variables (metals[1], metals[2])
        v1 = np.linspace(0, avail_budget, resolution)
        v2 = np.linspace(0, avail_budget, resolution)
        X2, X3 = np.meshgrid(v1, v2)

        # Vectorized simplex constraint
        mask = (X2 >= 0) & (X3 >= 0) & (X2 + X3 <= avail_budget + 1e-9)

        X_flat = np.zeros((resolution * resolution, system.n_dims))
        X_flat[:, 0] = X2.ravel()
        if system.n_dims >= 2:
            X_flat[:, 1] = X3.ravel()

        # Inject fixed slider coordinates for N>3 systems
        for i, col in enumerate(system.conc_cols[2:], start=2):
            X_flat[:, i] = fixed_coords.get(col, 0.0)

        mu_flat, sigma_flat = self.gpr.predict(X_flat, return_std=True)

        mu_grid = mu_flat.reshape((resolution, resolution))
        sigma_grid = sigma_flat.reshape((resolution, resolution))

        # Apply strict geometric cutoff (no artificial walls)
        mu_grid[~mask] = np.nan
        sigma_grid[~mask] = np.nan

        return X2, X3, mu_grid, sigma_grid

    def predict_point(self, x_independent: np.ndarray) -> Tuple[float, float]:
        mu, std = self.gpr.predict(x_independent.reshape(1, -1), return_std=True)
        return float(mu[0]), float(std[0])

    def get_next_candidate(self, system: SystemInfo, fixed_coords: Dict[str, float] = None, kappa: float = 2.0) -> Dict[str, float]:
        """
        Знаходить наступну оптимальну точку за допомогою Upper Confidence Bound (UCB).
        kappa контролює баланс:
            - мале kappa = експлуатація (шукаємо там, де μ найвище)
            - велике kappa = дослідження (шукаємо там, де σ найвище)
        """
        x2_grid, x3_grid, mu_grid, sigma_grid = self.predict_grid(system, fixed_coords)

        # UCB формула: Очікуване значення + (kappa * Невизначеність)
        ucb_grid = mu_grid + (kappa * sigma_grid)

        # Знаходимо індекси максимуму в сітці
        max_idx = np.unravel_index(np.nanargmax(ucb_grid), ucb_grid.shape)

        c2 = x2_grid[max_idx]
        c3 = x3_grid[max_idx] if system.n_dims >= 2 else 0.0

        fixed_sum = sum(fixed_coords.values()) if fixed_coords else 0.0
        c1 = 1.0 - c2 - c3 - fixed_sum

        return {
            system.metals[0]: c1,
            system.metals[1]: c2,
            system.metals[2] if len(system.metals) > 2 else "N/A": c3
        }
# ─────────────────────────────────────────────────────────────────────────────
# 4. FigureBuilder
# ─────────────────────────────────────────────────────────────────────────────

class FigureBuilder:
    """Stateless Plotly figure constructor."""

    def build(
        self,
        system: SystemInfo,
        x2_grid: np.ndarray,
        x3_grid: np.ndarray,
        mu_grid: np.ndarray,
        sigma_grid: np.ndarray,
        df_points: pd.DataFrame,
        target_col: str,
        fixed_coords: Dict[str, float],
    ) -> go.Figure:

        fig = go.Figure()

        # Dependent element (implicit)
        el1 = system.metals[0]
        # Independent axes
        el2 = system.metals[1]
        el3 = system.metals[2] if len(system.metals) > 2 else "None"

        # ── Layer 1: Gaussian Process Surface ────────────────────────────
        if len(system.metals) >= 3:
            # Custom Hover Template mapping grid back to all components
            custom_hover = np.empty(x2_grid.shape, dtype=object)
            sum_fixed = sum(fixed_coords.values())
            for i in range(x2_grid.shape[0]):
                for j in range(x2_grid.shape[1]):
                    if not np.isnan(mu_grid[i, j]):
                        c2, c3 = x2_grid[i, j], x3_grid[i, j]
                        c1 = 1.0 - c2 - c3 - sum_fixed
                        h = f"<b>Predicted {target_col}</b><br><br>"
                        h += f"μ (Mean) : {mu_grid[i, j]:.2f}<br>"
                        h += f"σ (Uncert): {sigma_grid[i, j]:.2f}<br>──────<br>"
                        h += f"{el1}: {c1:.4f}<br>{el2}: {c2:.4f}<br>{el3}: {c3:.4f}<br>"
                        for f_col, f_val in fixed_coords.items():
                            h += f"{f_col.split('_')[1]}: {f_val:.4f}<br>"
                        custom_hover[i, j] = h

            fig.add_trace(go.Surface(
                x=x2_grid, y=x3_grid, z=mu_grid,
                surfacecolor=sigma_grid,
                colorscale="Plasma",
                opacity=0.85,
                colorbar=dict(title="σ Uncert.", thickness=15, x=0.95),
                customdata=custom_hover,
                hovertemplate="%{customdata}<extra></extra>",
                name="GP Surface"
            ))

        # ── Layer 2: DFT Scatter Points ──────────────────────────────────
        if not df_points.empty:
            scatter_hover = []
            for _, row in df_points.iterrows():
                h = f"<b>DFT Target: {row[target_col]:.2f}</b><br>──────<br>"
                for c in system.all_conc_cols:
                    h += f"{c.split('_')[1]}: {row[c]:.4f}<br>"
                h += f"R²: {row.get('elastic_R2_min', 'N/A')}<br>"
                h += f"Source: {row.get('elastic_source', 'N/A')}<br>"
                scatter_hover.append(h)

            fig.add_trace(go.Scatter3d(
                x=df_points[system.conc_cols[0]],
                y=df_points[system.conc_cols[1]] if len(system.conc_cols) > 1 else np.zeros(len(df_points)),
                z=df_points[target_col],
                mode="markers",
                marker=dict(
                    size=6,
                    color=df_points[target_col],
                    colorscale="Viridis",
                    line=dict(color="white", width=2),
                    symbol="circle",
                ),
                text=scatter_hover,
                hoverinfo="text",
                name="DFT Points"
            ))

        # ── Layout Formatting ────────────────────────────────────────────
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9"),
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis_title=f"x_{el2}",
                yaxis_title=f"x_{el3}" if el3 != "None" else "",
                zaxis_title=target_col,
                xaxis=dict(range=[0, 1], gridcolor="#30363d", backgroundcolor="#0d1117"),
                yaxis=dict(range=[0, 1], gridcolor="#30363d", backgroundcolor="#0d1117"),
                zaxis=dict(gridcolor="#30363d", backgroundcolor="#0d1117"),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
        )
        return fig

# ─────────────────────────────────────────────────────────────────────────────
# 5. Dash App Setup & Layout
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="VCAForge Explorer")

app.layout = html.Div(
    style={"backgroundColor": "#0d1117", "color": "#c9d1d9", "minHeight": "100vh", "fontFamily": "sans-serif", "padding": "20px"},
    children=[
        # Stores
        dcc.Store(id='raw-data-store'),
        dcc.Store(id='gp-cache-store'),
        dcc.Store(id='system-info-store'),

        # Header Control Panel
        html.Div(style={"display": "flex", "gap": "20px", "marginBottom": "20px", "alignItems": "center"}, children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag & Drop or ', html.A('Select vca_results.csv', style={"color": "#58a6ff", "cursor": "pointer"})]),
                style={
                    'width': '300px', 'height': '40px', 'lineHeight': '40px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderColor': '#30363d',
                    'borderRadius': '5px', 'textAlign': 'center'
                }
            ),
            html.Div([
                html.Label("Target Property: ", style={"marginRight": "10px"}),
                dcc.Dropdown(id='target-dropdown', style={'width': '250px', 'color': 'black'})
            ]),
            html.Div(id='error-banner', style={"color": "#ff7b72", "fontWeight": "bold"})
        ]),

        # N>3 Sliders Panel (Hidden by default)
        html.Div(id='sliders-container', style={"marginBottom": "20px"}),

        # Main Visualization
        dcc.Graph(id='main-graph', style={"height": "70vh"}),

        # Click Info Panel
        html.Div(id='click-panel', style={
            "marginTop": "20px", "padding": "15px", "border": "1px solid #30363d",
            "borderRadius": "5px", "backgroundColor": "#161b22", "display": "flex",
            "justifyContent": "space-between", "alignItems": "center"
        }, children=[
            html.Div(id='click-text', children="Click any point on the surface to see exact concentrations and generate VCAForge commands."),
            html.Div([
                html.Button("Copy Coords", id="btn-copy-coords", style={"marginRight": "10px", "padding": "8px", "cursor": "pointer"}),
                dcc.Clipboard(target_id="hidden-coords-clip", id="clip-1", style={"display": "none"}),
                html.Div(id="hidden-coords-clip", style={"display": "none"}),

                html.Button("Copy CLI Command", id="btn-copy-cli", style={"padding": "8px", "cursor": "pointer"}),
                dcc.Clipboard(target_id="hidden-cli-clip", id="clip-2", style={"display": "none"}),
                html.Div(id="hidden-cli-clip", style={"display": "none"}),
            ])
        ])
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Callbacks
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output('raw-data-store', 'data'),
    Output('system-info-store', 'data'),
    Output('target-dropdown', 'options'),
    Output('target-dropdown', 'value'),
    Output('error-banner', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, ""

    loader = DataLoader()
    try:
        df = loader.load(contents)
        sys_info = loader.detect_system(df)

        # Identify numeric targets
        numerics = df.select_dtypes(include=[np.number]).columns.tolist()
        targets = [c for c in numerics if c not in sys_info.all_conc_cols and c not in ['step', 'status']]

        default_target = "H_Vickers_GPa" if "H_Vickers_GPa" in targets else (targets[0] if targets else None)

        return df.to_dict('records'), sys_info.__dict__, [{'label': t, 'value': t} for t in targets], default_target, ""
    except Exception as e:
        return None, None, [], None, f"Upload Error: {str(e)}"


@app.callback(
    Output('sliders-container', 'children'),
    Input('system-info-store', 'data')
)
def generate_sliders(sys_dict):
    if not sys_dict:
        return []
    sys_info = SystemInfo(**sys_dict)

    # Only need sliders if N >= 4 (which means n_dims >= 3)
    if sys_info.n_dims < 3:
        return []

    sliders = []
    # metals[0] is implicit, metals[1], metals[2] are axes. Sliders for rest.
    for col in sys_info.conc_cols[2:]:
        el = col.split("_")[1]
        sliders.append(html.Div([
            html.Label(f"Fix {el} Concentration ({col}):", style={"display": "block"}),
            dcc.Slider(
                id={'type': 'dim-slider', 'col': col},
                min=0.0, max=1.0, step=0.01, value=0.0,
                marks={i/10: str(i/10) for i in range(11)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], style={"width": "45%", "display": "inline-block", "marginRight": "5%"}))
    return sliders


@app.callback(
    Output('main-graph', 'figure'),
    Output('gp-cache-store', 'data'),
    Input('target-dropdown', 'value'),
    Input({'type': 'dim-slider', 'col': dash.dependencies.ALL}, 'value'),
    State('raw-data-store', 'data'),
    State('system-info-store', 'data'),
    State('gp-cache-store', 'data')
)
def update_graph(target_col, slider_vals, data_records, sys_dict, gp_cache):
    if not data_records or not target_col or not sys_dict:
        return dash.no_update, dash.no_update

    df = pd.DataFrame(data_records)
    sys_info = SystemInfo(**sys_dict)

    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else ""

    # Map sliders
    fixed_coords = {}
    if slider_vals:
        slider_inputs = ctx.inputs_list[1]
        for s_input in slider_inputs:
            fixed_coords[s_input['id']['col']] = s_input['value']

    sum_fixed = sum(fixed_coords.values())
    if sum_fixed >= 0.99:
        fig = go.Figure().update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            annotations=[dict(text="Constraint Overload: Sum of fixed metals >= 1.0", showarrow=False, font=dict(color="#ff7b72", size=20))]
        )
        return fig, dash.no_update

    # Prepare ML inputs
    X = df[sys_info.conc_cols].values
    y = df[target_col].values

    model = SurfaceModel()

    # Smart Caching Logic
    is_target_change = "target-dropdown" in trigger or gp_cache is None

    if is_target_change:
        # Full Optimization (Expensive)
        model.fit(X, y)
        new_cache = {'theta': model.get_params().tolist()}
    else:
        # Instant UI Update (Cheap) - restore optimized theta
        frozen_theta = np.array(gp_cache['theta'])
        model.fit(X, y, frozen_params=frozen_theta)
        new_cache = dash.no_update

    # Predict and Build
    x2_grid, x3_grid, mu_grid, sigma_grid = model.predict_grid(sys_info, fixed_coords=fixed_coords)

    builder = FigureBuilder()
    fig = builder.build(sys_info, x2_grid, x3_grid, mu_grid, sigma_grid, df, target_col, fixed_coords)

    return fig, new_cache


@app.callback(
    Output('click-text', 'children'),
    Output('hidden-coords-clip', 'children'),
    Output('hidden-cli-clip', 'children'),
    Input('main-graph', 'clickData'),
    State('system-info-store', 'data'),
    State({'type': 'dim-slider', 'col': dash.dependencies.ALL}, 'value')
)
def handle_click(clickData, sys_dict, slider_vals):
    if not clickData or not sys_dict:
        return dash.no_update, dash.no_update, dash.no_update

    sys_info = SystemInfo(**sys_dict)
    pt = clickData['points'][0]

    # Extract coordinate axes from the click
    c2 = pt.get('x', 0)
    c3 = pt.get('y', 0)

    # Reconstruct fixed sliders
    fixed_coords = {}
    if slider_vals:
        ctx = callback_context
        slider_inputs = ctx.states_list[2]
        for s_input in slider_inputs:
            fixed_coords[s_input['id']['col']] = s_input['value']

    c1 = 1.0 - c2 - c3 - sum(fixed_coords.values())

    # Construct mappings
    compositions = {sys_info.metals[0]: c1, sys_info.metals[1]: c2}
    if len(sys_info.metals) > 2:
        compositions[sys_info.metals[2]] = c3
    for col, val in fixed_coords.items():
        compositions[col.split("_")[1]] = val

    # Display Text
    disp_text = "  │  ".join([f"{k}: {v:.4f}" for k, v in compositions.items()])
    if 'z' in pt:
        disp_text += f"  │  Predicted: {pt['z']:.2f}"
    if 'surfacecolor' in pt:
         disp_text += f"  (σ = {pt['surfacecolor']:.2f})"

    # Copy Text 1
    clip_coords = " ".join([f"{k}={v:.4f}" for k, v in compositions.items()])

    # Copy Text 2 (CLI)
    metals_str = " ".join(compositions.keys())
    cli_cmd = f"--species {metals_str}"

    return html.B(disp_text), clip_coords, cli_cmd

# ─────────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n═════════════════════════════════════════════════════════════")
    print(" VCAForge Bayesian Surface Explorer Active")
    print(" Running at: http://localhost:8050")
    print("═════════════════════════════════════════════════════════════\n")
    app.run(debug=True)
