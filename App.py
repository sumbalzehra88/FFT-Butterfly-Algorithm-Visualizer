"""
app.py  —  Person 4
===================

Streamlit dashboard: "Hardware-Aware FFT Butterfly Explorer".

Run with:
    pip install streamlit numpy pandas plotly
    streamlit run app.py

It glues together:
    fft_engine.py     -> ideal floating-point radix-2 DIT FFT + butterfly schedule
    fixed_point.py    -> bit-accurate fixed-point FFT, overflow, bit growth, SNR
    memory_mapper.py  -> bit-reversal + dual-port RAM address map + collision check
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fft_engine import bit_reverse, butterfly_schedule, num_stages
from fixed_point import (QFormat, bit_growth_profile, enob, fixed_point_fft, format_binary,
                         growth_profile, output_snr_db, snr_db, snr_vs_bitwidth,
                         theoretical_snr_db)
from memory_mapper import (bit_reversal_table, check_collisions, cycle_count_summary,
                           generate_access_schedule, in_place_swap_pairs,
                           memory_contents, stage_parameters, stage_partner_matrix)

st.set_page_config(page_title="FFT Butterfly Hardware Explorer", page_icon="🔀", layout="wide")

# Colour constants used consistently across charts
C_IDEAL = "#1f77b4"     # floating point / ideal
C_FIXED = "#ff7f0e"     # fixed point
C_BAD = "#d62728"       # overflow / error / selection
C_GOOD = "#2ca02c"


# =========================================================================== #
# Helpers
# =========================================================================== #
def generate_signal(kind: str, N: int, k: int, amplitude: float, seed: int) -> np.ndarray:
    """Test signals. Amplitudes are kept < 1 so they fit a Q1.x input register."""
    n = np.arange(N)
    if kind == "Sine":
        return amplitude * np.sin(2 * np.pi * k * n / N)
    if kind == "Square":
        return np.where((n * k) % N < N / 2, amplitude, -amplitude).astype(float)
    rng = np.random.default_rng(seed)          # "Noise": bounded uniform white noise
    return amplitude * rng.uniform(-1.0, 1.0, N)


def fmt_c(z: complex, digits: int = 5) -> str:
    """Compact complex formatting: +0.12345-0.67890j"""
    return f"{z.real:+.{digits}f}{z.imag:+.{digits}f}j"


def err_lsb(fixed: complex, ideal: complex, lsb: float) -> float:
    """Worst component error expressed in LSBs."""
    d = fixed - ideal
    return max(abs(d.real), abs(d.imag)) / lsb


def fmt_db(v: float) -> str:
    if np.isinf(v):
        return "∞ dB" if v > 0 else "-∞ dB"
    return f"{v:.2f} dB"


@st.cache_data(show_spinner=False)
def cached_sweep(sig: tuple, int_bits: int, rounding: str, overflow: str, scale: bool):
    """SNR vs word length 4..32 bits (cached: it reruns 29 fixed-point FFTs)."""
    rows = snr_vs_bitwidth(np.array(sig), range(4, 33), int_bits, rounding, overflow, scale)
    return pd.DataFrame(rows)


# =========================================================================== #
# Sidebar — controls
# =========================================================================== #
st.sidebar.title("🎛️ Controls")

st.sidebar.subheader("Input signal")
signal_type = st.sidebar.selectbox("Signal type", ["Sine", "Square", "Noise"])
N = st.sidebar.radio("Transform size N", [8, 16], horizontal=True, index=1)
freq_bin = st.sidebar.slider("Frequency bin k", 1, N // 2 - 1, 1,
                             disabled=signal_type == "Noise",
                             help="Number of cycles across the N-sample window.")
amplitude = st.sidebar.slider("Amplitude", 0.05, 0.99, 0.90, 0.01,
                              help="Peak value relative to full scale (±1).")
seed = st.sidebar.number_input("Noise seed", 0, 9999, 7, disabled=signal_type != "Noise")

st.sidebar.subheader("Hardware precision")
bits = st.sidebar.slider("Word length (bits)", 4, 32, 12,
                         help="Total register width B, including sign bit.")

with st.sidebar.expander("⚙️ Advanced datapath options"):
    int_bits = st.slider("Integer bits (incl. sign)", 1, min(4, bits - 1), 1,
                         help="Extra integer bits are 'guard bits' that absorb bit growth.")
    rounding = st.radio("Quantization", ["round", "truncate"], horizontal=True)
    overflow_mode = st.radio("On overflow", ["saturate", "wrap"], horizontal=True)
    scale = st.checkbox("Scale by ½ every stage (shift right)", value=True,
                        help="Standard FPGA trick: dividing by 2 per stage cancels the "
                             "~1 bit/stage growth, so the data never overflows.")

fmt = QFormat.from_bits(bits, int_bits)
st.sidebar.info(f"**Datapath:** `{fmt.name}` · {fmt.total_bits}-bit\n\n"
                f"Range: [{fmt.min_value:g}, {fmt.max_value:.6g}]  \nLSB = 2⁻{fmt.frac_bits} "
                f"≈ {fmt.lsb:.3g}")

# =========================================================================== #
# Core computation (shared by all tabs)
# =========================================================================== #
x = generate_signal(signal_type, N, freq_bin, amplitude, int(seed))
S = num_stages(N)
fx = fixed_point_fft(x, fmt, rounding=rounding, overflow=overflow_mode, scale_per_stage=scale)
X_ideal = np.fft.fft(x)
X_fixed = fx.rescaled_output
out_snr = output_snr_db(x, fx)
in_snr = snr_db(x, fx.input_quant.values)

# =========================================================================== #
# Header + KPI row
# =========================================================================== #
st.title("🔀 Hardware-Aware FFT Butterfly Explorer")
st.caption(f"Radix-2 DIT FFT · N = {N} · {S} stages · {N // 2 * S} butterflies · "
           f"{fmt.name} datapath · twiddle ROM {fx.twiddle_fmt.name}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Q-format", fmt.name, f"{bits}-bit word", delta_color="off")
k2.metric("Input SNR", fmt_db(in_snr))
k3.metric("Output SNR", fmt_db(out_snr),
          None if np.isinf(out_snr) or np.isinf(in_snr) else f"{out_snr - in_snr:+.1f} dB through FFT")
k4.metric("ENOB (output)", "—" if np.isinf(out_snr) else f"{enob(out_snr):.1f} bits")
k5.metric("Overflow events", fx.total_overflows,
          "clean" if fx.total_overflows == 0 else "saturating!" if overflow_mode == "saturate" else "wrapping!",
          delta_color="normal" if fx.total_overflows == 0 else "inverse")

if fx.total_overflows:
    st.warning(f"⚠️ {fx.total_overflows} register overflow(s) detected. "
               + ("Enable ½-per-stage scaling or add integer (guard) bits." if not scale
                  else "Try lowering the amplitude or adding an integer bit."))

tab1, tab2, tab3 = st.tabs(["🦋 Butterfly & Precision Inspector",
                            "🧠 Memory & Bit-Reversal Map",
                            "📈 Spectral SNR Analysis"])


# =========================================================================== #
# TAB 1 — Butterfly & Precision Inspector
# =========================================================================== #
def butterfly_graph(sel_stage: int, sel_bf: int) -> go.Figure:
    """Signal-flow graph: columns = stages, rows = RAM addresses, node colour = error in LSB."""
    sched = butterfly_schedule(N)
    lsb = fmt.lsb
    fig = go.Figure()

    # --- edges (one trace for all normal edges, one for the selected butterfly)
    ex, ey, sx, sy = [], [], [], []
    tw_x, tw_y, tw_t = [], [], []
    for st_rows in sched:
        for bi, (s, g, j, top, bot, k) in enumerate(st_rows):
            selected = s == sel_stage and bi == sel_bf
            for src in (top, bot):
                for dst in (top, bot):
                    (sx if selected else ex).extend([s - 1, s, None])
                    (sy if selected else ey).extend([src, dst, None])
            tw_x.append(s - 0.82)
            tw_y.append(bot - 0.18)
            tw_t.append(f"W{k}")
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip", showlegend=False,
                             line=dict(color="rgba(128,128,128,0.35)", width=1)))
    fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", hoverinfo="skip", showlegend=False,
                             line=dict(color=C_BAD, width=3.5)))
    fig.add_trace(go.Scatter(x=tw_x, y=tw_y, mode="text", text=tw_t, hoverinfo="skip",
                             showlegend=False, textfont=dict(size=10, color="#8c564b")))

    # --- nodes, coloured by quantization error after that stage
    nx, ny, nc, nh = [], [], [], []
    for s in range(S + 1):
        for a in range(N):
            f, r = fx.stage_values[s, a], fx.reference_stage_values[s, a]
            e = err_lsb(f, r, lsb)
            nx.append(s); ny.append(a); nc.append(e)
            nh.append(f"<b>Addr {a} · {'load' if s == 0 else f'after stage {s}'}</b><br>"
                      f"ideal {fmt_c(r, 4)}<br>fixed {fmt_c(f, 4)}<br>error {e:.2f} LSB")
    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode="markers", hovertext=nh, hoverinfo="text", showlegend=False,
        marker=dict(size=16, color=nc, colorscale="YlOrRd", cmin=0,
                    cmax=max(1.0, max(nc)), line=dict(width=1, color="#444"),
                    colorbar=dict(title="err<br>(LSB)", thickness=12))))

    # --- input / output labels
    for a in range(N):
        fig.add_annotation(x=-0.12, y=a, text=f"x[{bit_reverse(a, S)}]", showarrow=False,
                           xanchor="right", font=dict(size=11))
        fig.add_annotation(x=S + 0.12, y=a, text=f"X[{a}]", showarrow=False,
                           xanchor="left", font=dict(size=11))

    fig.update_xaxes(tickvals=list(range(S + 1)),
                     ticktext=["Bit-reversed<br>input"] + [f"Stage {s}" for s in range(1, S + 1)],
                     range=[-0.6, S + 0.6], showgrid=False, zeroline=False)
    fig.update_yaxes(autorange="reversed", showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(height=60 + 34 * N, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


with tab1:
    st.markdown("Every column is one pass over the RAM; every red/orange dot is a register whose "
                "fixed-point value has drifted from the ideal value. **Pick a butterfly** to see "
                "its arithmetic bit-for-bit.")

    left, right = st.columns([3, 2], gap="large")
    with right:
        c1, c2 = st.columns(2)
        sel_stage = c1.selectbox("Stage", list(range(1, S + 1)), index=0)
        sel_bf = c2.selectbox("Butterfly", list(range(N // 2)),
                              format_func=lambda i: f"#{i}  ({fx.butterflies[sel_stage - 1][i].top}"
                                                    f" ↔ {fx.butterflies[sel_stage - 1][i].bottom})")
        b = fx.butterflies[sel_stage - 1][sel_bf]
        ref_prev, ref_next = fx.reference_stage_values[sel_stage - 1], fx.reference_stage_values[sel_stage]
        ideal_prod = b.twiddle * ref_prev[b.bottom]
        g = "·½" if scale else ""

        st.markdown(f"**Butterfly {sel_bf}, stage {sel_stage}** — addresses "
                    f"`{b.top}` (A) and `{b.bottom}` (B), twiddle "
                    f"**W<sub>{N}</sub><sup>{b.twiddle_k}</sup>**", unsafe_allow_html=True)
        # Build LaTeX pieces outside the f-string (Python < 3.12 forbids backslashes inside {...})
        half = r"\cdot\tfrac12" if scale else ""
        tw = "W_{%d}^{%d}" % (N, b.twiddle_k)
        st.latex(r"A' = (A + " + tw + "B)" + half + r"\qquad B' = (A - " + tw + "B)" + half)

        rows = [
            ("A in", ref_prev[b.top], b.a_in, fmt),
            ("B in", ref_prev[b.bottom], b.b_in, fmt),
            (f"W{b.twiddle_k} (ROM)", b.twiddle, b.twiddle_q, fx.twiddle_fmt),
            ("W·B (multiplier)", ideal_prod, b.product, fmt),
            (f"A' = (A+WB){g}", ref_next[b.top], b.a_out, fmt),
            (f"B' = (A−WB){g}", ref_next[b.bottom], b.b_out, fmt),
        ]
        df_bf = pd.DataFrame([{
            "Signal": name, "Ideal (float64)": fmt_c(ideal), "Fixed-point": fmt_c(fixed),
            "Error (LSB)": round(err_lsb(fixed, ideal, q.lsb), 3),
            "Re bits": format_binary(fixed.real, q), "Im bits": format_binary(fixed.imag, q),
        } for name, ideal, fixed, q in rows])
        st.dataframe(df_bf, hide_index=True, width="stretch")
        if b.overflow:
            st.error("🔥 This butterfly overflowed — a register hit the edge of its range.")
        else:
            st.success("No overflow in this butterfly.")
        st.caption("Binary columns are two's-complement words; the dot marks the binary point "
                   f"of the Q-format ({fmt.name} data, {fx.twiddle_fmt.name} twiddles).")

    with left:
        st.plotly_chart(butterfly_graph(sel_stage, sel_bf), width="stretch")
        st.caption("Lines: butterfly data paths (the lower output carries a −1). "
                   "Brown labels: twiddle exponent k applied to the lower input. "
                   "Node colour: worst real/imag error versus the ideal datapath, in LSBs.")

    st.divider()
    st.subheader("📶 Bit growth per stage")
    actual = growth_profile(fx.reference_stage_values)
    stage_snrs = fx.stage_snr_db()
    stage_errs = fx.stage_max_error_lsb()

    gc1, gc2 = st.columns([3, 2], gap="large")
    with gc1:
        labels = ["Input"] + [f"Stage {s}" for s in range(1, S + 1)]
        fig_g = go.Figure()
        fig_g.add_bar(x=labels, y=[r["int_bits_needed"] for r in bit_growth_profile(x)],
                      name="Needed without scaling", marker_color=C_BAD)
        fig_g.add_bar(x=labels, y=[r["int_bits_needed"] for r in actual],
                      name="Needed in this datapath", marker_color=C_IDEAL)
        fig_g.add_hline(y=fmt.int_bits, line_dash="dash", line_color=C_GOOD,
                        annotation_text=f"available: {fmt.int_bits} int bit(s)",
                        annotation_position="top left")
        fig_g.update_layout(barmode="group", height=330, yaxis_title="integer bits (incl. sign)",
                            margin=dict(l=10, r=10, t=30, b=10),
                            legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_g, width="stretch")
    with gc2:
        st.markdown(
            f"A radix-2 butterfly computes A ± WB, so magnitudes can **double each stage** "
            f"(≈1 bit; up to 1+√2 ≈ 2.41× for complex components). Across {S} stages that is "
            f"up to **{S} extra integer bits**.\n\n"
            "Hardware answers: (1) add guard bits, (2) shift right by 1 every stage "
            "(what the checkbox does — it costs ~1 bit of SNR per stage of resolution), or "
            "(3) block floating point (scale only when needed).")
    df_stage = pd.DataFrame({
        "Stage": labels,
        "Max |component| (unscaled)": [round(r["max_component"], 4) for r in bit_growth_profile(x)],
        "Int bits needed (unscaled)": [r["int_bits_needed"] for r in bit_growth_profile(x)],
        "Max |component| (datapath)": [round(r["max_component"], 4) for r in actual],
        "Overflow events": [fx.input_quant.n_overflow] + fx.overflow_per_stage,
        "Max error (LSB)": [round(e, 2) for e in stage_errs],
        "Stage SNR (dB)": [None if np.isinf(v) else round(v, 2) for v in stage_snrs],
    })
    st.dataframe(df_stage, hide_index=True, width="stretch")


# =========================================================================== #
# TAB 2 — Memory & Bit-Reversal Map
# =========================================================================== #
with tab2:
    st.markdown(f"The FFT runs **in place** on one {N}-word dual-port RAM (ports A and B) "
                f"with a {N // 2}-entry twiddle ROM.")

    cyc = cycle_count_summary(N)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Load cycles", cyc["load_cycles"])
    m2.metric("Compute cycles", cyc["compute_cycles"], f"{cyc['butterflies']} butterflies × 2",
              delta_color="off")
    m3.metric("Unload cycles", cyc["unload_cycles"])
    m4.metric("Total latency", f"{cyc['total_cycles']} cycles")

    st.subheader("1 · Bit-reversed input addressing")
    bc1, bc2 = st.columns([3, 2], gap="large")
    with bc1:
        st.dataframe(pd.DataFrame(bit_reversal_table(N)), hide_index=True, width="stretch",
                     height=min(36 * N + 40, 620))
    with bc2:
        fig_br = go.Figure()
        for i in range(N):
            r = bit_reverse(i, S)
            fig_br.add_trace(go.Scatter(x=[0, 1], y=[i, r], mode="lines+markers",
                                        line=dict(color=C_IDEAL if i == r else C_FIXED, width=2),
                                        marker=dict(size=9), hoverinfo="text", showlegend=False,
                                        hovertext=f"x[{i}] ({i:0{S}b}) → addr {r} ({r:0{S}b})"))
            fig_br.add_annotation(x=-0.04, y=i, text=f"x[{i}]", showarrow=False, xanchor="right")
            fig_br.add_annotation(x=1.04, y=i, text=f"addr {i}", showarrow=False, xanchor="left")
        fig_br.update_xaxes(range=[-0.35, 1.35], tickvals=[0, 1],
                            ticktext=["sample (natural)", "RAM address"], showgrid=False)
        fig_br.update_yaxes(autorange="reversed", visible=False)
        fig_br.update_layout(height=60 + 30 * N, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_br, width="stretch")
        swaps = in_place_swap_pairs(N)
        st.caption(f"Orange lines move; blue stay put (palindromic indices). An in-place "
                   f"reorder needs {len(swaps)} swaps: " + ", ".join(f"{a}↔{b}" for a, b in swaps))

    st.subheader("2 · Stage geometry and address pairing")
    st.dataframe(pd.DataFrame(stage_parameters(N)).astype({"Twiddle ROM addresses": str}),
                 hide_index=True, width="stretch")
    pm = stage_partner_matrix(N)
    fig_hm = go.Figure(go.Heatmap(
        z=pm["group"].T, x=[f"Stage {s}" for s in range(1, S + 1)], y=list(range(N)),
        text=[[f"↔{pm['partner'][s, a]}  W{pm['twiddle_k'][s, a]}" for s in range(S)] for a in range(N)],
        texttemplate="%{text}", colorscale="Blues", showscale=False,
        hovertemplate="addr %{y}, %{x}<br>%{text}<extra></extra>"))
    fig_hm.update_yaxes(autorange="reversed", title="RAM address", dtick=1)
    fig_hm.update_layout(height=60 + 30 * N, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_hm, width="stretch")
    st.caption("Each cell: the address this word is paired with and the twiddle used. "
               "Shading = butterfly group. Partner distance doubles every stage (1, 2, 4, …).")

    st.subheader("3 · Cycle-by-cycle dual-port RAM access log")
    sched = generate_access_schedule(N)
    df_s = pd.DataFrame(sched)
    phases = ["All"] + list(dict.fromkeys(df_s["phase"]))
    ph = st.selectbox("Show phase", phases, index=phases.index("STAGE 1"))  # default: first compute stage
    view = df_s if ph == "All" else df_s[df_s["phase"] == ph]

    tl1, tl2 = st.columns([2, 3], gap="large")
    with tl1:
        st.dataframe(view.drop(columns=["stage"]), hide_index=True, width="stretch", height=420)
    with tl2:
        fig_tl = go.Figure()
        for op, color in (("READ", C_IDEAL), ("WRITE", C_FIXED)):
            sub = df_s[df_s["op"] == op]
            for port, sym in (("A", "circle"), ("B", "square")):
                fig_tl.add_trace(go.Scatter(
                    x=sub["cycle"], y=sub[f"port_{port}_addr"], mode="markers",
                    name=f"{op} port {port}", marker=dict(color=color, symbol=sym, size=8,
                                                          line=dict(width=0.5, color="#333")),
                    hovertext=sub["note"]))
        for phase_name, grp in df_s.groupby("phase", sort=False):
            fig_tl.add_vrect(x0=grp["cycle"].min() - 0.5, x1=grp["cycle"].max() + 0.5,
                             fillcolor="gray", opacity=0.07 if phase_name.startswith("STAGE") and
                             int(phase_name.split()[1]) % 2 else 0.0, line_width=0,
                             annotation_text=phase_name, annotation_position="top left",
                             annotation_font_size=10)
        if ph != "All":
            fig_tl.add_vrect(x0=view["cycle"].min() - 0.5, x1=view["cycle"].max() + 0.5,
                             line=dict(color=C_BAD, width=2), fillcolor="rgba(0,0,0,0)")
        fig_tl.update_yaxes(autorange="reversed", title="RAM address", dtick=1 if N <= 8 else 2)
        fig_tl.update_xaxes(title="clock cycle")
        fig_tl.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                             legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_tl, width="stretch")

    report = check_collisions(sched, N)
    if report["ok"]:
        st.success(f"✅ Collision-free: all {report['total_cycles']} cycles verified.")
    else:
        st.error("❌ Memory conflicts detected:\n\n" + "\n".join(f"- {i}" for i in report["issues"]))
    for name, passed in report["checks"]:
        st.markdown(f"{'✅' if passed else '❌'} {name}")

    st.subheader("4 · What the RAM holds between passes")
    contents = memory_contents(N)
    st.dataframe(pd.DataFrame({"Address": range(N), "After load": contents[0],
                               **{f"After stage {s}": contents[s] for s in range(1, S + 1)}}),
                 hide_index=True, width="stretch")


# =========================================================================== #
# TAB 3 — Spectral SNR Analysis
# =========================================================================== #
with tab3:
    sp1, sp2 = st.columns(2, gap="large")
    eps = 1e-12
    ref_db = 20 * np.log10(np.max(np.abs(X_ideal)) + eps)
    bins = list(range(N))

    with sp1:
        st.subheader("Magnitude spectrum")
        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(x=bins, y=20 * np.log10(np.abs(X_ideal) + eps) - ref_db,
                                    mode="lines+markers", name="Float64 (ideal)",
                                    line=dict(color=C_IDEAL)))
        fig_sp.add_trace(go.Scatter(x=bins, y=20 * np.log10(np.abs(X_fixed) + eps) - ref_db,
                                    mode="markers", name=f"Fixed {fmt.name}",
                                    marker=dict(color=C_FIXED, size=10, symbol="x")))
        fig_sp.update_layout(height=360, xaxis_title="bin k", yaxis_title="dB re. peak",
                             yaxis_range=[-160, 10], margin=dict(l=10, r=10, t=10, b=10),
                             legend=dict(orientation="h", y=1.1))
        fig_sp.update_xaxes(dtick=1)
        st.plotly_chart(fig_sp, width="stretch")
        st.caption("Empty bins should be −∞ dB; in fixed point they fill with a quantization "
                   "noise floor whose height falls ~6 dB per added bit.")

    with sp2:
        st.subheader("Error spectrum |X_fixed − X_ideal|")
        err = np.abs(X_fixed - X_ideal)
        fig_er = go.Figure(go.Bar(x=bins, y=20 * np.log10(err + eps) - ref_db, marker_color=C_BAD))
        fig_er.add_hline(y=20 * np.log10(np.sqrt(np.mean(err ** 2)) + eps) - ref_db,
                         line_dash="dash", annotation_text="RMS error")
        fig_er.update_layout(height=360, xaxis_title="bin k", yaxis_title="dB re. peak",
                             margin=dict(l=10, r=10, t=10, b=10))
        fig_er.update_xaxes(dtick=1)
        st.plotly_chart(fig_er, width="stretch")

    st.divider()
    sw1, sw2 = st.columns([3, 2], gap="large")
    with sw1:
        st.subheader("SNR vs word length")
        df_sw = cached_sweep(tuple(x), int_bits, rounding, overflow_mode, scale)
        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(x=df_sw["bits"], y=df_sw["input_snr_db"].replace(np.inf, np.nan),
                                    name="Input quantizer SNR", line=dict(color=C_IDEAL, dash="dot")))
        fig_sw.add_trace(go.Scatter(x=df_sw["bits"], y=df_sw["output_snr_db"].replace(np.inf, np.nan),
                                    name="FFT output SNR", line=dict(color=C_FIXED, width=3)))
        fig_sw.add_trace(go.Scatter(x=df_sw["bits"], y=[theoretical_snr_db(b_) for b_ in df_sw["bits"]],
                                    name="6.02·B + 1.76 dB", line=dict(color="gray", dash="dash")))
        if not np.isinf(out_snr):
            fig_sw.add_trace(go.Scatter(x=[bits], y=[out_snr], mode="markers", name="current",
                                        marker=dict(color=C_BAD, size=14, symbol="star")))
        fig_sw.update_layout(height=380, xaxis_title="word length B (bits)", yaxis_title="SNR (dB)",
                             margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_sw, width="stretch")
    with sw2:
        st.subheader("Noise accumulation by stage")
        fig_acc = go.Figure(go.Scatter(
            x=["Input"] + [f"S{s}" for s in range(1, S + 1)],
            y=[np.nan if np.isinf(v) else v for v in fx.stage_snr_db()],
            mode="lines+markers", line=dict(color=C_FIXED, width=3)))
        fig_acc.update_layout(height=380, yaxis_title="SNR of RAM contents (dB)",
                              margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_acc, width="stretch")

    gap = theoretical_snr_db(bits) - out_snr if not np.isinf(out_snr) else 0.0
    st.info(
        f"**Reading the numbers:** at {bits} bits an ideal quantizer gives ≈ "
        f"{theoretical_snr_db(bits):.1f} dB. This FFT delivers {fmt_db(out_snr)}"
        + (f" — about {gap:.1f} dB ({gap / 6.02:.1f} bits) lost to "
           f"{'per-stage scaling, ' if scale else ''}twiddle and product rounding across {S} stages."
           if gap > 0 else ".")
        + (" Truncation adds a DC bias (−½ LSB per operation) that shows up in bin 0."
           if rounding == "truncate" else "")
        + (" Wrap-around overflow causes catastrophic errors — that's why DSPs saturate."
           if overflow_mode == "wrap" and fx.total_overflows else ""))

    with st.expander("🔎 Time-domain input: float vs quantized"):
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=list(range(N)), y=x, name="float", mode="lines+markers",
                                   line=dict(color=C_IDEAL)))
        fig_t.add_trace(go.Scatter(x=list(range(N)), y=fx.input_quant.values.real, name=fmt.name,
                                   mode="lines+markers", line=dict(color=C_FIXED, shape="hv")))
        fig_t.update_layout(height=300, xaxis_title="n", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_t, width="stretch")
        st.dataframe(pd.DataFrame({"n": range(N), "x[n] float": np.round(x, 6),
                                   "quantized": fx.input_quant.values.real,
                                   "code": fx.input_quant.codes_re,
                                   "bits": [format_binary(v, fmt) for v in fx.input_quant.values.real]}),
                     hide_index=True, width="stretch")