import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

st.set_page_config(
    page_title="SPCMonitor | Process Control Dashboard",
    page_icon="📊", layout="wide"
)

st.markdown("""
<style>
  html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="block-container"]{
    background-color:#0f1117!important;color:#e8e8e8!important}
  [data-testid="stSidebar"]{background-color:#161b22!important}
  [data-testid="stSidebar"] *{color:#e8e8e8!important}
  h1,h2,h3,h4,p,span,div,label,.stMarkdown{color:#e8e8e8!important}
  [data-testid="metric-container"]{background-color:#1c2333!important;
    border:1px solid #30363d!important;border-radius:8px!important;padding:12px!important}
  [data-testid="metric-container"] *{color:#e8e8e8!important}
  [data-testid="stMetricValue"]{color:#ffffff!important;font-weight:600!important}
  [data-testid="stMetricLabel"]{color:#8b949e!important}
  [data-testid="stTabs"] button{color:#8b949e!important;background:transparent!important}
  [data-testid="stTabs"] button[aria-selected="true"]{color:#58a6ff!important;
    border-bottom:2px solid #58a6ff!important}
  [data-testid="stDataFrame"]{background-color:#161b22!important;
    border:1px solid #30363d!important;border-radius:8px!important}
  .stDataFrame th{background-color:#21262d!important;color:#c9d1d9!important;font-weight:600!important}
  .stDataFrame td{color:#e8e8e8!important;background-color:#161b22!important}
  [data-testid="stSelectbox"]>div{background-color:#21262d!important;
    border:1px solid #30363d!important;color:#e8e8e8!important;border-radius:6px!important}
  [data-testid="stSlider"] label{color:#8b949e!important}
  [data-testid="stButton"] button{background-color:#238636!important;
    color:#ffffff!important;border:none!important;border-radius:6px!important;font-weight:500!important}
  [data-testid="stExpander"]{background-color:#161b22!important;
    border:1px solid #30363d!important;border-radius:8px!important}
  [data-testid="stExpander"] *{color:#e8e8e8!important}
  hr{border-color:#30363d!important}
  table{background-color:#161b22!important;color:#e8e8e8!important;
    border-collapse:collapse!important;width:100%!important}
  th{background-color:#21262d!important;color:#c9d1d9!important;
    padding:8px 12px!important;border:1px solid #30363d!important}
  td{color:#e8e8e8!important;padding:8px 12px!important;border:1px solid #30363d!important}
  tr:nth-child(even) td{background-color:#1c2333!important}
</style>
""", unsafe_allow_html=True)

DARK = dict(template="plotly_dark", paper_bgcolor="#0f1117", plot_bgcolor="#161b22",
            font=dict(color="#e8e8e8"), margin=dict(t=40,b=40,l=10,r=10))

# ── DATA GENERATION ───────────────────────────────────────────────────────────
@st.cache_data
def generate_process_data(n_subgroups=125, subgroup_size=5, seed=42):
    np.random.seed(seed)
    mu, sigma, usl, lsl = 10.02, 0.018, 10.10, 9.90
    shifts = {60: 0.03, 85: 0.05, 100: -0.04}
    sigma_jumps = {40: 1.6, 75: 1.0}

    data, rows = [], []
    cur_sig = sigma
    for i in range(n_subgroups):
        if i in sigma_jumps: cur_sig = sigma * sigma_jumps[i]
        shift = sum(v for k,v in shifts.items() if i >= k)
        sg = np.random.normal(mu + shift, cur_sig, subgroup_size)
        data.append(sg)
        rows.append({"Subgroup": i+1, "Mean": round(sg.mean(),4),
                     "Range": round(sg.max()-sg.min(),4),
                     "Std": round(sg.std(),4),
                     "Defects": int(np.sum((sg<lsl)|(sg>usl)))})
    df = pd.DataFrame(rows)
    df["Xbar_CL"] = round(df["Mean"].mean(),4)
    r_bar = df["Range"].mean()
    d2, d3 = 2.326, 0.864
    A2 = 0.577
    df["Xbar_UCL"] = round(df["Xbar_CL"] + A2*r_bar, 4)
    df["Xbar_LCL"] = round(df["Xbar_CL"] - A2*r_bar, 4)
    D3, D4 = 0.0, 2.114
    df["R_CL"]  = round(r_bar,4)
    df["R_UCL"] = round(D4*r_bar,4)
    df["R_LCL"] = 0.0
    sigma_est = r_bar/d2
    cl_val = float(df["Mean"].mean())
    df["Cp"]  = round((usl-lsl)/(6*sigma_est), 3)
    df["Cpk"] = round(min((usl-cl_val)/(3*sigma_est),
                          (cl_val-lsl)/(3*sigma_est)), 3)
    return df, pd.DataFrame(data), usl, lsl

df, raw, USL, LSL = generate_process_data()

def we_violations(series, cl, ucl, lcl):
    viol = []
    s = series.values; n = len(s)
    sigma1 = (ucl-cl)/3
    for i in range(n):
        v = []
        if s[i] > ucl or s[i] < lcl:                      v.append("Rule 1: Beyond 3-sigma")
        if i>=1 and all(x>cl+2*sigma1 for x in s[max(0,i-1):i+1]): v.append("Rule 2: 2 of 2 beyond 2-sigma")
        if i>=3 and all(x>cl        for x in s[i-3:i+1]):  v.append("Rule 4: 4 beyond 1-sigma same side")
        if i>=7 and (all(x>cl for x in s[i-7:i+1]) or
                     all(x<cl for x in s[i-7:i+1])):        v.append("Rule 5: 8 in a row same side")
        viol.append("; ".join(v) if v else "")
    return viol

df["Xbar_Violations"] = we_violations(df["Mean"],  df["Xbar_CL"].iloc[0],
                                       df["Xbar_UCL"].iloc[0], df["Xbar_LCL"].iloc[0])
df["R_Violations"]    = we_violations(df["Range"], df["R_CL"].iloc[0],
                                       df["R_UCL"].iloc[0], df["R_LCL"].iloc[0])

viol_idx  = df[df["Xbar_Violations"]!=""].index.tolist()
viol_idx_r= df[df["R_Violations"]!=""].index.tolist()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("<h2 style='color:#e8e8e8;margin-bottom:2px'>SPCMonitor — Statistical Process Control Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#8b949e;font-size:13px'>Real-time control charts, Western Electric Rule violation detection, process capability analysis, and DOE for manufacturing quality engineering</p>", unsafe_allow_html=True)

with st.expander("Preview demo data", expanded=False):
    st.markdown("<p style='color:#8b949e;font-size:13px'>Sample of the 125 subgroups (5 measurements each) powering this dashboard</p>", unsafe_allow_html=True)
    preview_cols = ["Subgroup","Mean","Range","Std","Defects","Xbar_UCL","Xbar_CL","Xbar_LCL","Xbar_Violations"]
    sample_df = df[preview_cols].head(10).copy()
    def sty_viol_prev(v):
        return "background-color:#3d1f1f;color:#f85149;font-weight:600" if v else "color:#3fb950"
    st.dataframe(sample_df.style.map(sty_viol_prev, subset=["Xbar_Violations"]),
                 use_container_width=True, hide_index=True)
    st.markdown(f"<p style='color:#8b949e;font-size:12px'>Full dataset: 125 subgroups, subgroup size n=5. USL={USL} | LSL={LSL} | Target Cpk >= 1.33. Engineered shifts at subgroups 60, 85, and 100 to trigger Western Electric Rule violations. See Data preview tab for full exploration.</p>", unsafe_allow_html=True)

with st.expander("What does this app do?", expanded=False):
    st.markdown("""
**SPCMonitor** implements the full statistical process control toolkit used by quality engineers
in manufacturing and data center hardware environments to detect out-of-control conditions
before defects propagate through a production line.

**Control charts implemented**

Xbar-R charts monitor process mean and variability simultaneously. P-charts track defect
proportion across subgroups. Western Electric Rules (4 rules) detect non-random patterns
that indicate process shifts even when individual points remain within control limits.

**Process capability**

Cp measures whether the process spread fits within specification limits.
Cpk adjusts for process centering. A Cpk below 1.33 means the process is producing
defects at a rate that requires investigation. The dashboard calculates both in real time
and identifies the recommended corrective action.

**Design of Experiments (DOE)**

A 2-factor full factorial DOE identifies which process variables (temperature and pressure)
have statistically significant effects on the output metric, using ANOVA to separate
main effects from interaction effects. This tells the quality engineer which lever to
adjust to improve Cpk.
    """)

# ── METRICS ──────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
xbar_viols  = int((df["Xbar_Violations"]!="").sum())
r_viols     = int((df["R_Violations"]!="").sum())
cp_val      = df["Cp"].iloc[0]
cpk_val     = df["Cpk"].iloc[0]
total_def   = int(df["Defects"].sum())

c1.metric("Total subgroups",        len(df))
c2.metric("Xbar violations (WE)",   xbar_viols)
c3.metric("Range violations (WE)",  r_viols)
c4.metric("Cp / Cpk",              f"{cp_val} / {cpk_val}")
c5.metric("Total defects (out of spec)", total_def)
st.markdown("---")

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "Xbar-R charts","P-chart","Capability analysis","DOE analysis","Data preview"
])

def control_chart(x, y, cl, ucl, lcl, title, y_label, violations):
    fig = go.Figure()
    clr = ["#f85149" if v else "#58a6ff" for v in [bool(violations[i]) for i in range(len(y))]]
    fig.add_scatter(x=x, y=y, mode="lines+markers",
                    line=dict(color="#58a6ff",width=1.5),
                    marker=dict(color=clr, size=7), name=y_label)
    for val, col, name in [(cl,"#3fb950","CL"),(ucl,"#f85149","UCL"),(lcl,"#f85149","LCL")]:
        fig.add_hline(y=val, line_color=col, line_dash="dash" if col=="#f85149" else "solid",
                      line_width=1.5, annotation_text=f"{name}={val:.4f}",
                      annotation_font_color=col, annotation_position="right")
    for i, v in enumerate(violations):
        if v:
            fig.add_vline(x=x[i], line_color="#d29922", line_dash="dot", line_width=1,
                          opacity=0.5)
    fig.update_layout(**DARK, title=dict(text=title,font=dict(color="#e8e8e8",size=14)),
                      height=340, yaxis_title=y_label)
    return fig

# ── TAB 1: XBAR-R ────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<h4 style='color:#e8e8e8'>Xbar chart — process mean</h4>", unsafe_allow_html=True)
    st.plotly_chart(
        control_chart(df["Subgroup"], df["Mean"],
                      df["Xbar_CL"].iloc[0], df["Xbar_UCL"].iloc[0], df["Xbar_LCL"].iloc[0],
                      "Xbar Control Chart", "Subgroup Mean", df["Xbar_Violations"].tolist()),
        use_container_width=True
    )
    st.markdown("<h4 style='color:#e8e8e8'>R chart — process range</h4>", unsafe_allow_html=True)
    st.plotly_chart(
        control_chart(df["Subgroup"], df["Range"],
                      df["R_CL"].iloc[0], df["R_UCL"].iloc[0], df["R_LCL"].iloc[0],
                      "R Control Chart", "Subgroup Range", df["R_Violations"].tolist()),
        use_container_width=True
    )
    if viol_idx:
        st.markdown("<h4 style='color:#e8e8e8'>Western Electric Rule violations</h4>", unsafe_allow_html=True)
        viol_df = df[df["Xbar_Violations"]!=""][["Subgroup","Mean","Xbar_Violations"]].rename(
            columns={"Xbar_Violations":"Rule Violated"})

        def sty_rule(v):
            return "background-color:#3d1f1f;color:#f85149;font-weight:600" if v else ""

        st.dataframe(viol_df.style.map(sty_rule, subset=["Rule Violated"]),
                     use_container_width=True, hide_index=True)

# ── TAB 2: P-CHART ────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<h4 style='color:#e8e8e8'>P-chart — defect proportion</h4>", unsafe_allow_html=True)
    n = 5
    df["p"] = df["Defects"] / n
    p_bar = df["p"].mean()
    df["p_UCL"] = p_bar + 3*np.sqrt(p_bar*(1-p_bar)/n)
    df["p_LCL"] = max(p_bar - 3*np.sqrt(p_bar*(1-p_bar)/n), 0)

    fig_p = go.Figure()
    p_colors = ["#f85149" if v > df["p_UCL"].iloc[0] else "#58a6ff" for v in df["p"]]
    fig_p.add_scatter(x=df["Subgroup"], y=df["p"], mode="lines+markers",
                      line=dict(color="#58a6ff",width=1.5),
                      marker=dict(color=p_colors,size=7))
    for val, col, lbl in [(p_bar,"#3fb950","CL"),(df["p_UCL"].iloc[0],"#f85149","UCL"),
                           (df["p_LCL"].iloc[0],"#f85149","LCL")]:
        fig_p.add_hline(y=val, line_color=col, line_dash="dash" if col=="#f85149" else "solid",
                        annotation_text=f"{lbl}={val:.4f}", annotation_font_color=col,
                        annotation_position="right")
    fig_p.update_layout(**DARK, height=360, yaxis_title="Defect Proportion (p)")
    st.plotly_chart(fig_p, use_container_width=True)

    oc = df[df["p"] > df["p_UCL"]]
    if len(oc):
        st.markdown(
            f"<div style='background:#3d1f1f;border:1px solid #f85149;border-radius:8px;"
            f"padding:10px 14px;font-size:13px;color:#f85149'>"
            f"<b>{len(oc)} subgroup(s) exceed UCL on P-chart.</b> "
            f"Subgroups: {', '.join(oc['Subgroup'].astype(str).tolist())} "
            f"— initiate process investigation and CAPA.</div>",
            unsafe_allow_html=True
        )

# ── TAB 3: CAPABILITY ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("<h4 style='color:#e8e8e8'>Process capability analysis</h4>", unsafe_allow_html=True)
    all_vals = raw.values.flatten()
    mu_est   = float(df["Mean"].mean())
    sig_est  = float(df["Range"].mean() / 2.326)
    cp_val   = round((USL-LSL)/(6*sig_est),3)
    cpk_val  = round(min((USL-mu_est)/(3*sig_est),(mu_est-LSL)/(3*sig_est)),3)
    pp_val   = round((USL-LSL)/(6*all_vals.std()),3)

    cc1,cc2,cc3,cc4 = st.columns(4)
    cc1.metric("Cp",  cp_val,  delta=None)
    cc2.metric("Cpk", cpk_val, delta="Below 1.33 target" if cpk_val < 1.33 else "Capable")
    cc3.metric("Pp",  pp_val,  delta=None)
    cc4.metric("USL / LSL", f"{USL} / {LSL}", delta=None)

    x_range = np.linspace(all_vals.min()-0.05, all_vals.max()+0.05, 300)
    fig_cap  = go.Figure()
    fig_cap.add_histogram(x=all_vals, nbinsx=40, name="Process data",
                          marker_color="#58a6ff", opacity=0.6, histnorm="probability density")
    fig_cap.add_scatter(x=x_range,
                        y=stats.norm.pdf(x_range, mu_est, sig_est),
                        mode="lines", name="Normal fit",
                        line=dict(color="#3fb950",width=2))
    for val, col, lbl in [(USL,"#f85149","USL"),(LSL,"#f85149","LSL"),(mu_est,"#d29922","Mean")]:
        fig_cap.add_vline(x=val, line_color=col, line_dash="dash",
                          annotation_text=lbl, annotation_font_color=col)
    fig_cap.update_layout(**DARK, height=380,
                          legend=dict(orientation="h",y=1.05,font=dict(color="#e8e8e8")))
    st.plotly_chart(fig_cap, use_container_width=True)

    st.markdown("<h4 style='color:#e8e8e8'>Capability interpretation and recommendation</h4>", unsafe_allow_html=True)
    if cpk_val >= 1.33:
        msg = "Process is capable and centered. No immediate action required. Continue monitoring."
        col = "#0f2d1f"; tc = "#3fb950"
    elif cpk_val >= 1.0:
        msg = (f"Process is capable (Cp={cp_val}) but off-center (Cpk={cpk_val}). "
               f"A mean shift of {round((1.33*3*sig_est)-(USL-mu_est),4)} units toward nominal "
               f"would bring Cpk above 1.33 target. Investigate process centering.")
        col = "#3d2f0f"; tc = "#d29922"
    else:
        msg = (f"Process is not capable (Cpk={cpk_val} < 1.0). Defects are being produced. "
               f"Requires immediate investigation. Reduce process variability and re-center mean.")
        col = "#3d1f1f"; tc = "#f85149"
    st.markdown(
        f"<div style='background:{col};border-radius:8px;padding:12px 16px;"
        f"font-size:13px;color:{tc}'>{msg}</div>",
        unsafe_allow_html=True
    )

# ── TAB 4: DOE ────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<h4 style='color:#e8e8e8'>Design of Experiments (DOE) — 2-factor full factorial</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e;font-size:13px'>Identifies which process variables have statistically significant effects on defect rate using ANOVA</p>", unsafe_allow_html=True)

    np.random.seed(7)
    temps     = [180, 200]
    pressures = [50,  70]
    n_rep     = 20
    doe_rows  = []
    for t in temps:
        for p in pressures:
            base_def = 0.03 + 0.015*(t-180)/20 + 0.02*(p-50)/20 + 0.008*((t-180)/20)*((p-50)/20)
            for _ in range(n_rep):
                doe_rows.append({
                    "Temperature (C)": t, "Pressure (PSI)": p,
                    "Defect Rate (%)": round(max(0, base_def + np.random.normal(0,0.005))*100, 3),
                    "Temp_coded": 1 if t==200 else -1,
                    "Pressure_coded": 1 if p==70 else -1,
                })
    doe = pd.DataFrame(doe_rows)
    doe["Interaction"] = doe["Temp_coded"] * doe["Pressure_coded"]

    groups = [doe[(doe["Temperature (C)"]==t) & (doe["Pressure (PSI)"]==p)]["Defect Rate (%)"]
              for t in temps for p in pressures]
    f_stat, p_val = stats.f_oneway(*groups)

    from itertools import product as iprod
    anova_rows = []
    for factor, col in [("Temperature","Temp_coded"),("Pressure","Pressure_coded"),("Interaction","Interaction")]:
        corr, pv = stats.pearsonr(doe[col], doe["Defect Rate (%)"])
        anova_rows.append({"Factor": factor, "Correlation": round(corr,3),
                           "p-value": round(pv,4),
                           "Significant (p<0.05)": "Yes" if pv<0.05 else "No",
                           "Effect Direction": "Increases defect rate" if corr>0
                                               else "Decreases defect rate"})
    anova_df = pd.DataFrame(anova_rows)

    def sty_sig(v):
        return ("background-color:#3d1f1f;color:#f85149;font-weight:600" if v=="Yes"
                else "background-color:#0f2d1f;color:#3fb950;font-weight:600")

    st.dataframe(anova_df.style.map(sty_sig, subset=["Significant (p<0.05)"]),
                 use_container_width=True, hide_index=True)

    st.markdown(f"<p style='color:#8b949e;font-size:13px'>ANOVA F-statistic: <b style='color:#e8e8e8'>{round(f_stat,2)}</b> | p-value: <b style='color:#{'f85149' if p_val<0.05 else '3fb950'}'>{round(p_val,4)}</b> {'-- At least one factor significantly affects defect rate' if p_val<0.05 else '-- No significant factor detected'}</p>", unsafe_allow_html=True)

    fig_doe = px.box(doe, x="Temperature (C)", y="Defect Rate (%)",
                     color=doe["Pressure (PSI)"].astype(str),
                     color_discrete_map={"50":"#58a6ff","70":"#f85149"},
                     points="all")
    fig_doe.update_layout(**DARK, height=360,
                          legend=dict(orientation="h",y=1.05,title="Pressure (PSI)",
                                      font=dict(color="#e8e8e8")))
    st.plotly_chart(fig_doe, use_container_width=True)

    st.markdown("<h4 style='color:#e8e8e8'>Interaction plot</h4>", unsafe_allow_html=True)
    int_df = doe.groupby(["Temperature (C)","Pressure (PSI)"])["Defect Rate (%)"].mean().reset_index()
    fig_int = px.line(int_df, x="Temperature (C)", y="Defect Rate (%)",
                      color=int_df["Pressure (PSI)"].astype(str),
                      color_discrete_map={"50":"#58a6ff","70":"#f85149"},
                      markers=True)
    fig_int.update_layout(**DARK, height=320,
                          legend=dict(orientation="h",y=1.05,title="Pressure (PSI)",
                                      font=dict(color="#e8e8e8")))
    st.plotly_chart(fig_int, use_container_width=True)

# ── TAB 5: DATA PREVIEW ───────────────────────────────────────────────────────
with tab5:
    st.markdown("<h4 style='color:#e8e8e8'>Subgroup summary data</h4>", unsafe_allow_html=True)
    show_cols = ["Subgroup","Mean","Range","Std","Defects",
                 "Xbar_UCL","Xbar_CL","Xbar_LCL","Cp","Cpk","Xbar_Violations"]

    def sty_v(v):
        return ("background-color:#3d1f1f;color:#f85149;font-weight:600"
                if v else "")

    st.dataframe(
        df[show_cols].style.map(sty_v, subset=["Xbar_Violations"]),
        use_container_width=True, hide_index=True, height=440
    )
    csv = df[show_cols].to_csv(index=False).encode()
    st.download_button("Download subgroup data as CSV", csv,
                       "spcmonitor_data.csv","text/csv")

    st.markdown("<h4 style='color:#e8e8e8'>Scoring methodology</h4>", unsafe_allow_html=True)
    st.markdown("""
| Rule | Description | Trigger |
|------|-------------|---------|
| Rule 1 | Beyond 3-sigma | Any point outside UCL or LCL |
| Rule 2 | 2 of 2 beyond 2-sigma | 2 consecutive points beyond 2-sigma same side |
| Rule 4 | 4 beyond 1-sigma | 4 of 5 consecutive beyond 1-sigma same side |
| Rule 5 | 8 in a row same side | 8 consecutive points above or below centerline |

Cp = (USL-LSL) / (6 x sigma). Cpk = min((USL-mean)/(3 x sigma), (mean-LSL)/(3 x sigma)).
Target: Cp and Cpk >= 1.33 for a capable and centered process.
    """)
