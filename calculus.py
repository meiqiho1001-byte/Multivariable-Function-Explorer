import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

# ---------------------------
# 1. Page Configuration & Aesthetic
# ---------------------------
st.set_page_config(page_title="Calculus Analysis Pro", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    h1, h2, h3 { color: #00d4ff !important; font-family: 'Inter', sans-serif; }
    .topic-card {
        background-color: #161b22;
        padding: 25px;
        border-radius: 15px;
        border-left: 5px solid #00d4ff;
        margin-bottom: 25px;
    }
    .explanation { color: #8b949e; font-style: italic; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 2. Sidebar: Domain Control
# ---------------------------
with st.sidebar:
    st.header("🌐 Domain Settings")
    x_min, x_max = st.slider("x range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.slider("y range", -10.0, 10.0, (-5.0, 5.0))
    z_min, z_max = st.slider("z range", -10.0, 10.0, (-5.0, 5.0))
    st.info("Adjust the sliders to change the calculation boundaries.")

# ---------------------------
# 3. User Function Input
# ---------------------------
st.title("📊 Multivariable Calculus Analyzer")
func_input = st.text_input("Enter your function f(x, y, z):", value="x**2 - y**2")
st.caption("Use standard Python notation: ** for power, sin(), cos(), exp(), sqrt()")

# ---------------------------
# 4. Mathematical Engine
# ---------------------------
x, y, z = sp.symbols('x y z')
try:
    f_sym = sp.sympify(func_input)
    vars_present = f_sym.free_symbols
except Exception as e:
    st.error(f"Invalid Mathematical Expression: {e}")
    st.stop()

# ---------------------------
# 5. Detailed Topic Sections
# ---------------------------

# --- Topic 1: Geometric Visualization ---
st.markdown('<div class="topic-card">', unsafe_allow_html=True)
st.subheader("1. Geometric Visualization (Surface Plot)")
st.markdown('<p class="explanation">This 3D plot represents the function as a topography. For functions of three variables, we visualize a "slice" by fixing z.</p>', unsafe_allow_html=True)

if z in vars_present:
    z_val = st.slider("Fixed Z-slice for visualization", float(z_min), float(z_max), 0.0)
    f_plot = f_sym.subs(z, z_val)
else:
    f_plot = f_sym

x_arr = np.linspace(x_min, x_max, 50)
y_arr = np.linspace(y_min, y_max, 50)
X, Y = np.meshgrid(x_arr, y_arr)
f_lamb = sp.lambdify((x, y), f_plot, "numpy")
Z_vals = f_lamb(X, Y)
if np.isscalar(Z_vals): Z_vals = np.full(X.shape, Z_vals)

fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z_vals, colorscale="IceFire")])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=500, paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Topic 2: Partial Derivatives ---
st.markdown('<div class="topic-card">', unsafe_allow_html=True)
st.subheader("2. Partial Derivatives")
st.markdown('<p class="explanation">Partial derivatives measure the rate of change of the function along one specific axis while holding the other variables constant.</p>', unsafe_allow_html=True)
df_dx = sp.diff(f_sym, x)
df_dy = sp.diff(f_sym, y)
st.latex(rf"\frac{{\partial f}}{{\partial x}} = {sp.latex(df_dx)}")
st.latex(rf"\frac{{\partial f}}{{\partial y}} = {sp.latex(df_dy)}")
if z in vars_present:
    df_dz = sp.diff(f_sym, z)
    st.latex(rf"\frac{{\partial f}}{{\partial z}} = {sp.latex(df_dz)}")
st.markdown('</div>', unsafe_allow_html=True)

# --- Topic 3: The Gradient Vector ---
st.markdown('<div class="topic-card">', unsafe_allow_html=True)
st.subheader("3. Gradient Vector ($\nabla f$)")
st.markdown('<p class="explanation">The gradient points in the direction of the steepest ascent at any given point on the surface.</p>', unsafe_allow_html=True)
grad = [df_dx, df_dy]
if z in vars_present: grad.append(df_dz)
st.latex(rf"\nabla f = \langle {', '.join([sp.latex(g) for g in grad])} \rangle")
st.markdown('</div>', unsafe_allow_html=True)

# --- Topic 4: Total Differential ---
st.markdown('<div class="topic-card">', unsafe_allow_html=True)
st.subheader("4. Total Differential (df)")
st.markdown('<p class="explanation">The total differential represents the change in the function value resulting from small changes in all independent variables.</p>', unsafe_allow_html=True)
diff_expr = rf"df = \left({sp.latex(df_dx)}\right)dx + \left({sp.latex(df_dy)}\right)dy"
if z in vars_present:
    diff_expr += rf" + \left({sp.latex(df_dz)}\right)dz"
st.latex(diff_expr)
st.markdown('</div>', unsafe_allow_html=True)

# --- Topic 5: Critical Points & Optimization ---
st.markdown('<div class="topic-card">', unsafe_allow_html=True)
st.subheader("5. Critical Points and Classification")
st.markdown('<p class="explanation">Critical points occur where the gradient is zero. We use the Second Derivative Test (Hessian Determinant) to classify these as Maxima, Minima, or Saddle Points.</p>', unsafe_allow_html=True)
try:
    pts = sp.solve([df_dx, df_dy], (x, y), dict=True)
    if pts:
        for pt in pts:
            fxx = sp.diff(df_dx, x).subs(pt)
            fyy = sp.diff(df_dy, y).subs(pt)
            fxy = sp.diff(df_dx, y).subs(pt)
            D = fxx*fyy - fxy**2
            
            label = "Saddle Point" if D < 0 else ("Local Min" if fxx > 0 else "Local Max")
            st.write(f"📍 **Point {pt}:** Identified as a **{label}**")
    else:
        st.write("No critical points found for the given function.")
except:
    st.write("Solver encountered complex values or transcendental expressions too difficult to isolate.")
st.markdown('</div>', unsafe_allow_html=True)
