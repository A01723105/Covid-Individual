

from __future__ import annotations

import streamlit as st
import pandas as pd

import data_sources2 as ds
import charts2 as charts

st.set_page_config(
    page_title="The Mortality Mirage · WHO COVID-19 data story",
    page_icon="🩺", layout="wide", initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1280px; }
h1,h2,h3 { letter-spacing:-0.01em; color:#1f2937; }
.kicker { text-transform:uppercase; letter-spacing:.16em; font-size:.7rem;
  color:#4f8edc; font-weight:700; }
.lede { font-size:1.04rem; line-height:1.55; color:#475569; }
.panel { background:#ffffff; border:1px solid #e8edf3; border-radius:16px;
  padding:1.1rem 1.2rem; box-shadow:0 1px 2px rgba(16,24,40,.04); height:100%; }
.kpi { background:#ffffff; border:1px solid #e8edf3; border-radius:14px;
  padding:.9rem 1rem; height:100%; }
.kpi .v { font-size:1.8rem; font-weight:800; line-height:1.05; color:#1f2937; }
.kpi.blue .v { color:#2f6fc0; }
.kpi .l { font-size:.78rem; color:#64748b; margin-top:.2rem; }
.kpi .s { font-size:.72rem; color:#94a3b8; margin-top:.3rem; }
.note { border-left:3px solid #cbd5e1; background:#f8fafc; padding:.7rem .95rem;
  border-radius:0 10px 10px 0; font-size:.9rem; color:#475569; margin:.4rem 0; }
.note.blue { border-left-color:#4f8edc; background:#eef5fd; color:#33567f; }
.note.blue b { color:#2f6fc0; }
.charttitle { font-weight:700; font-size:1rem; color:#1f2937; margin:.1rem 0 .2rem; }
.charttitle.blue { color:#2f6fc0; }
.sub { font-size:.82rem; color:#94a3b8; }
.tag span { display:inline-block; background:#eef2f7; border:1px solid #e2e8f0;
  border-radius:999px; padding:.2rem .6rem; margin:.15rem .3rem .15rem 0;
  font-size:.78rem; color:#475569; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=60 * 60 * 12, show_spinner="Loading WHO data…")
def get_data():
    df, label, synthetic = ds.load_timeseries()
    region = ds.region_weekly(df)
    metrics = ds.add_rolling_and_cfr(region)
    head = ds.headline_numbers(df)
    under = ds.undercount_table(df)
    vax_long, vax_snap, vax_label = ds.load_vaccine_equity()
    return (df, metrics, head, under, vax_snap, label, vax_label, synthetic)


(df, metrics_all, head, under, vax_snap,
 source_label, vax_label, synthetic) = get_data()


def human(n):
    n = float(n)
    for u, d in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(n) >= d:
            return f"{n/d:.2f}{u}"
    return f"{n:,.0f}"


def kpi(col, v, l, s="", blue=False):
    cls = "kpi blue" if blue else "kpi"
    col.markdown(f'<div class="{cls}"><div class="v">{v}</div>'
                 f'<div class="l">{l}</div><div class="s">{s}</div></div>',
                 unsafe_allow_html=True)


def note(text, blue=False):
    st.markdown(f'<div class="note {"blue" if blue else ""}">{text}</div>',
                unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 🩺 The Mortality Mirage")

    st.markdown("#### Controls")
    plottable = [r for r in ds.REGION_ORDER
                 if r in metrics_all["WHO_region"].unique() and r != "OTHER"]

    st.markdown("**Global Arc — highlight regions**")
    default_hl = [r for r in ["EURO", "AMRO", "SEARO"] if r in plottable]
    sel_highlight = st.multiselect(
        "Regions to lift out in blue",
        options=plottable, default=default_hl,
        format_func=lambda r: ds.REGION_NAMES.get(r, r),
        help="Every region is drawn in gray; your selection is lifted out in blue.")

    dmin = metrics_all["Date_reported"].min().to_pydatetime()
    dmax = metrics_all["Date_reported"].max().to_pydatetime()
    date_range = st.slider("Date range", min_value=dmin, max_value=dmax,
                           value=(dmin, dmax), format="MMM YYYY")


lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
metrics = metrics_all[(metrics_all["Date_reported"] >= lo) &
                      (metrics_all["Date_reported"] <= hi)]


st.markdown('<div class="kicker">WHO COVID-19 data story</div>',
            unsafe_allow_html=True)
st.title("The Mortality Mirage")
st.markdown('<p class="lede">The world tracked a number that was wrong. The '
            'reported death toll measured <b>who could count the dead</b>, not '
            'who was dying. This dashboard follows that gap.</p>',
            unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🌏  Global Arc", "💀  Mortality Mirage"])

with tab1:
    st.markdown("### What the official record shows")
    k = st.columns(4)
    kpi(k[0], human(head["total_cases"]), "Reported cases",
        "all WHO regions, full series")
    kpi(k[1], human(head["total_deaths"]), "Reported deaths",
        f'{head["first_date"]:%b %Y} – {head["last_date"]:%b %Y}')
    kpi(k[2], f'{head["overall_cfr_pct"]:.2f}%', "Overall reported CFR",
        "deaths ÷ reported cases")
    kpi(k[3], f'{head["n_countries"]:,}', "Countries & areas",
        "reporting to WHO")

    st.markdown("&nbsp;", unsafe_allow_html=True)

    if sel_highlight:
        names = ", ".join(ds.REGION_NAMES.get(r, r) for r in
                          [x for x in ds.REGION_ORDER if x in sel_highlight])
        note(f'Highlighted in blue: <b>{names}</b>. Every other region stays '
             'gray in the background — use the sidebar to change the selection.',
             blue=True)
    else:
        note("Pick one or more regions in the sidebar to lift them out of the "
             "gray background in blue.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="charttitle">Weekly cases by region</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            charts.fig_region_lines(metrics, sel_highlight, metric="cases_smooth"),
            width='stretch')
    with c2:
        st.markdown('<div class="charttitle">Weekly deaths by region</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            charts.fig_region_lines(metrics, sel_highlight, metric="deaths_smooth"),
            width='stretch')

    st.markdown('<span class="sub">Each line is a WHO region. The regions '
                'reporting the most cases are not the ones reporting the most '
                'deaths — the first clue for the next tab.</span>',
                unsafe_allow_html=True)

with tab2:
    # --- KPIs at the top of the dashboard ---
    f = st.columns(4)
    kpi(f[0], "2.75×", "Excess vs reported deaths, 2020–21",
        "WHO: 14.9M vs 5.4M", blue=True)
    kpi(f[1], "4%", "Share of excess deaths in low-income countries",
        "a blind spot, not a success", blue=True)
    africa_uc = under.loc[under["study_region"] == "Africa", "undercount_factor"]
    africa_val = f"{africa_uc.iloc[0]:.1f}×" if len(africa_uc) else "—"
    kpi(f[2], africa_val, "Africa's implied undercount",
        "vs ~1× in Western Europe", blue=True)
    if vax_snap is not None:
        low = vax_snap.loc[vax_snap.income_group == "Low income", "end_2021_pct"]
        high = vax_snap.loc[vax_snap.income_group == "High income", "end_2021_pct"]
        gap = (f"{high.iloc[0]:.0f}% vs {low.iloc[0]:.0f}%"
               if len(low) and len(high) else "—")
    else:
        gap = "—"
    kpi(f[3], gap, "Vaccinated, high- vs low-income (end 2021)",
        "≥1 dose, derived from WHO data", blue=True)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    INCOME_TO_REGION = {
        "High income": "Europe",
        "Upper-middle income": "Americas",
        "Lower-middle income": "South & South-East Asia",
        "Low income": "Africa",
    }
    REGION_TO_INCOME = {v: k for k, v in INCOME_TO_REGION.items()}

    ss = st.session_state
    ss.setdefault("focus", None)      
    ss.setdefault("k_income", 0)      
    ss.setdefault("k_region", 0)      

    def _clicked_label(state_key):
        state = ss.get(state_key)
        if not state:
            return None
        try:
            points = state["selection"]["points"]
        except Exception:
            return None
        if not points:
            return None
        cd = points[0].get("customdata") if hasattr(points[0], "get") else None
        return cd[0] if cd else None

    def on_income_select():
        ss.focus = _clicked_label(f"chart_income_{ss.k_income}")
        ss.k_region += 1             

    def on_region_select():
        reg = _clicked_label(f"chart_region_{ss.k_region}")
        ss.focus = REGION_TO_INCOME.get(reg) if reg else None
        ss.k_income += 1             

    focus_income = ss.focus
    focus_region = INCOME_TO_REGION.get(focus_income) if focus_income else None

    
    cap, btn = st.columns([5, 1])
   
    with btn:
        if st.button("Reset", width='stretch'):
            ss.focus = None
            ss.k_income += 1
            ss.k_region += 1
            st.rerun()

    
    inc_hl = focus_income or "Low income"
    reg_hl = focus_region or "Africa"

    
    if focus_region:
        row = under.loc[under["study_region"] == focus_region]
        rep_m = float(row["reported_deaths_m"].iloc[0]) if len(row) else None
        exc_m = float(row["excess_deaths_m"].iloc[0]) if len(row) else None
        fig1 = charts.fig_reported_vs_excess(focus_region, rep_m, exc_m)
    else:
        fig1 = charts.fig_reported_vs_excess()

   
    row1 = st.columns([1.05, 1.15, 1.15])
    with row1[0]:
        st.markdown('<div class="charttitle blue">Counted vs WHO estimates '
                    '(2020–21)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig1, width='stretch', key="chart_reported")
    with row1[1]:
        st.markdown('<div class="charttitle">Where deaths actually happened</div>',
                    unsafe_allow_html=True)
        st.markdown('<span class="sub">Share of global excess deaths by income '
                    'group — click a bar</span>', unsafe_allow_html=True)
        st.plotly_chart(
            charts.fig_excess_by_income(highlight=inc_hl), width='stretch',
            key=f"chart_income_{ss.k_income}", on_select=on_income_select,
            selection_mode="points")
    with row1[2]:
        st.markdown('<div class="charttitle">Implied undercount by region</div>',
                    unsafe_allow_html=True)
        st.markdown('<span class="sub">WHO excess estimate ÷ reported deaths — '
                    'click a bar</span>', unsafe_allow_html=True)
        st.plotly_chart(
            charts.fig_undercount_by_region(under, highlight=reg_hl),
            width='stretch', key=f"chart_region_{ss.k_region}",
            on_select=on_region_select, selection_mode="points")
