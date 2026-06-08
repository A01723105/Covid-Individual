
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import data_sources2 as ds
from income_groups import INCOME_ORDER

PAPER = "rgba(0,0,0,0)"
GRID = "rgba(148,163,184,0.16)"
TEXT = "#1f2937"
MUTED = "#9aa6b8"

GRAY_LINE = "rgba(148,163,184,0.28)"  
GRAY_SOLID = "#cbd5e1"
BLUE = "#4f8edc"                        
BLUE_DK = "#2f6fc0"
BLUE_SOFT = "#9cc2ef"
NEUTRAL_BAR = "#d2d9e3"                 
NEUTRAL_EDGE = "#b9c2cf"

MILESTONES = [
    ("2020-03-11", "WHO declares pandemic"),
    ("2020-12-08", "First vaccinations"),
    ("2021-11-26", "Omicron named"),
]


def _style(fig, height=380, legend=False, title=None):
    
    title_obj = dict(font=dict(size=15, color=TEXT), subtitle=dict(text=""))
    if title:
        title_obj["text"] = title
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif",
                  color=TEXT, size=12),
        margin=dict(l=10, r=10, t=46 if title else 16, b=10),
        height=height, hovermode="x unified",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left",
                    x=0, bgcolor=PAPER) if legend else dict(),
        title=title_obj,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def _milestones(fig):
    for date, label in MILESTONES:
        fig.add_vline(x=date, line_width=1, line_dash="dot",
                      line_color=MUTED, opacity=0.5)
        fig.add_annotation(x=date, y=1.0, yref="paper", text=label,
                           showarrow=False, textangle=-90, xshift=-7,
                           font=dict(size=9, color=MUTED), yanchor="top")
    return fig


def fig_region_lines(metrics, selected, metric="cases_smooth",
                     log_scale=False, milestones=True, height=420):
    """
    All WHO regions in gray; `selected` regions drawn in soft blue on top.
    `metrics` comes from ds.add_rolling_and_cfr() (one row per region/week).
    `selected` is a list of WHO_region codes (e.g. ["EURO", "AMRO"]).
    """
    field = metric
    is_deaths = "deaths" in metric
    unit = "deaths" if is_deaths else "cases"
    selected = [r for r in ds.REGION_ORDER if r in set(selected or [])]

    fig = go.Figure()

    for region in ds.REGION_ORDER:
        if region in selected:
            continue
        sub = metrics[metrics["WHO_region"] == region]
        if sub.empty:
            continue
        name = ds.REGION_NAMES.get(region, region)
        fig.add_trace(go.Scatter(
            x=sub["Date_reported"], y=sub[field], mode="lines",
            line=dict(width=1.4, color=GRAY_LINE), name=name,
            showlegend=False, opacity=0.85,
            hovertemplate=f"%{{y:,.0f}} {unit}<extra>{name}</extra>",
        ))

    blue_shades = [BLUE, BLUE_DK, "#1d4e8a", "#6aa3e6", "#11518f", "#83b3ee"]
    for i, region in enumerate(selected):
        sub = metrics[metrics["WHO_region"] == region]
        if sub.empty:
            continue
        name = ds.REGION_NAMES.get(region, region)
        fig.add_trace(go.Scatter(
            x=sub["Date_reported"], y=sub[field], name=name, mode="lines",
            line=dict(width=2.8, color=blue_shades[i % len(blue_shades)]),
            hovertemplate=f"%{{y:,.0f}} {unit}<extra>{name}</extra>",
        ))

    fig.update_layout(yaxis_title=f"Weekly {unit} (4-wk avg)")
    if log_scale:
        fig.update_yaxes(type="log")
    fig = _style(fig, height=height, legend=bool(selected))
    if milestones and not log_scale:
        _milestones(fig)
    return fig


def fig_reported_vs_excess(focus_region=None, reported_m=None, excess_m=None):
    """
    Global reported vs WHO-estimated excess deaths. When a region is focused
    (via cross-filtering), shows that region's reported vs excess instead.
    """
    g = ds.EXCESS_GLOBAL
    if focus_region and reported_m is not None and excess_m is not None:
        rep, exc = float(reported_m), float(excess_m)
        ratio = exc / rep if rep else float("nan")
        title_left, title_right = "Reported deaths", "WHO estimated"
        ann = (f"<b>{ratio:.1f}×</b> the reported toll" if rep
               else "no comparable reported toll")
        err = None
        subtitle = f"{focus_region}: counted vs estimated (2020–21)"
    else:
        rep, exc = g["reported_deaths_2020_2021_m"], g["excess_deaths_2020_2021_m"]
        ratio = g["undercount_ratio"]
        title_left, title_right = "Reported COVID deaths", "WHO estimated excess deaths"
        ann = f"<b>{ratio:.2f}×</b> the reported toll"
        err = dict(type="data", symmetric=False,
                   array=[g["excess_high_m"] - exc],
                   arrayminus=[exc - g["excess_low_m"]],
                   color=BLUE_DK, thickness=1.3)
        subtitle = None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[title_left], y=[rep],
        marker_color=NEUTRAL_BAR, marker_line_color=NEUTRAL_EDGE, marker_line_width=1,
        width=0.5, text=[f"{rep:.2f}M" if rep < 1 else f"{rep:.1f}M"],
        textposition="outside", textfont=dict(color=MUTED), showlegend=False,
    ))
    fig.add_trace(go.Bar(
        x=[title_right], y=[exc], marker_color=BLUE, width=0.5,
        error_y=err, text=[f"{exc:.1f}M"], textposition="outside",
        textfont=dict(color=BLUE_DK), showlegend=False,
    ))
    fig.add_annotation(
        x=title_right, y=exc, text=ann, showarrow=True, arrowhead=0,
        arrowcolor=BLUE, ax=0, ay=-46, font=dict(color=BLUE_DK, size=13),
    )
    fig.update_layout(yaxis_title="Deaths (millions)")
    fig = _style(fig, height=330)
    if subtitle:
        fig.update_layout(
            title=dict(text=subtitle, font=dict(size=12, color=MUTED),
                       subtitle=dict(text="")),
            margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fig_excess_by_income(highlight="Low income"):
    """Monochrome; `highlight` (an income-group label) is drawn in blue."""
    d = ds.EXCESS_BY_INCOME.copy()
    order = ["High income", "Upper-middle income",
             "Lower-middle income", "Low income"]
    d["income_group"] = pd.Categorical(d["income_group"], order, ordered=True)
    d = d.sort_values("income_group")
    groups = list(d["income_group"])
    colors = [BLUE if g == highlight else NEUTRAL_BAR for g in groups]
    tcol = [BLUE_DK if g == highlight else MUTED for g in groups]
    fig = go.Figure(go.Bar(
        x=d["share_of_excess_pct"], y=d["income_group"], orientation="h",
        marker_color=colors, marker_line_color=NEUTRAL_EDGE, marker_line_width=1,
        text=[f"{v}%" for v in d["share_of_excess_pct"]], textposition="outside",
        textfont=dict(color=tcol),
        customdata=[[g] for g in groups],   # canonical label for click-reading
        hovertemplate="%{customdata[0]}: %{x}% of global excess deaths<extra></extra>",
    ))
    fig.update_layout(xaxis_title="% of 14.9M excess deaths")
    fig.update_xaxes(range=[0, 60])
    return _style(fig, height=300)


def fig_undercount_by_region(table, highlight="Africa"):
    """Monochrome; `highlight` (a study-region label) is drawn in blue."""
    t = table.sort_values("undercount_factor", ascending=True)
    regions = list(t["study_region"])
    colors = [BLUE if r == highlight else NEUTRAL_BAR for r in regions]
    tcol = [BLUE_DK if r == highlight else MUTED for r in regions]
    customdata = t[["study_region", "excess_deaths_m", "reported_deaths_m"]].values
    fig = go.Figure(go.Bar(
        y=t["study_region"], x=t["undercount_factor"], orientation="h",
        marker_color=colors, marker_line_color=NEUTRAL_EDGE, marker_line_width=1,
        text=[f"{v:.1f}×" for v in t["undercount_factor"]], textposition="outside",
        textfont=dict(color=tcol),
        customdata=customdata,   # [study_region, excess_m, reported_m]
        hovertemplate=("%{customdata[0]}<br>Excess: %{customdata[1]:.2f}M"
                       "<br>Reported: %{customdata[2]:.2f}M"
                       "<br>Undercount: %{x:.1f}×<extra></extra>"),
    ))
    fig.add_vline(x=1, line_dash="dash", line_color=MUTED)
    fig.update_layout(xaxis_title="× more deaths than were reported")
    return _style(fig, height=300)


def fig_cfr_mirage(metrics, regions):
    """
    Seaborn/Matplotlib line chart. Monochrome: all regions gray (the point is
    that the lines CONVERGE, not that they differ). Returns a Matplotlib figure.
    """
    rows = []
    for region in ds.REGION_ORDER:
        if region not in regions:
            continue
        sub = metrics[metrics["WHO_region"] == region].copy()
        sub = sub[sub["cfr_pct"].notna()]
        if sub.empty:
            continue
        if sub["cases_smooth"].max() > 0:
            sub = sub[sub["cases_smooth"] > sub["cases_smooth"].max() * 0.002]
        sub["Region"] = ds.REGION_NAMES.get(region, region)
        rows.append(sub[["Date_reported", "cfr_pct", "Region"]])
    plot_df = (pd.concat(rows, ignore_index=True) if rows
               else pd.DataFrame(columns=["Date_reported", "cfr_pct", "Region"]))

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=130)
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    if not plot_df.empty:
        sns.lineplot(data=plot_df, x="Date_reported", y="cfr_pct", units="Region",
                     estimator=None, color=GRAY_SOLID, linewidth=1.4,
                     alpha=0.9, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("Apparent CFR (%)", color=TEXT, fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.margins(x=0.01)
    fig.tight_layout()
    return fig


def fig_vaccine_equity(snapshots):
    """
    Real derived data. Focal point: the Low-income group in blue; rest gray.
    snapshots: income_group, end_2021_pct, latest_pct
    """
    d = snapshots.copy()
    d["income_group"] = pd.Categorical(d["income_group"], INCOME_ORDER, ordered=True)
    d = d.sort_values("income_group")
    is_low = (d["income_group"] == "Low income")
    bar_end = [BLUE_SOFT if low else "#e4e9f0" for low in is_low]
    bar_latest = [BLUE if low else NEUTRAL_BAR for low in is_low]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="End 2021 · ≥1 dose", x=d["income_group"], y=d["end_2021_pct"],
        marker_color=bar_end, marker_line_color=NEUTRAL_EDGE, marker_line_width=1,
        text=[f"{v:.0f}%" if pd.notna(v) else "" for v in d["end_2021_pct"]],
        textposition="outside", textfont=dict(color=MUTED),
    ))
    fig.add_trace(go.Bar(
        name="Latest · ≥1 dose", x=d["income_group"], y=d["latest_pct"],
        marker_color=bar_latest, marker_line_color=NEUTRAL_EDGE, marker_line_width=1,
        text=[f"{v:.0f}%" if pd.notna(v) else "" for v in d["latest_pct"]],
        textposition="outside",
        textfont=dict(color=[BLUE_DK if low else MUTED for low in is_low]),
    ))
    fig.add_hline(y=70, line_dash="dash", line_color=MUTED,
                  annotation_text="WHO 70% target", annotation_position="top left",
                  annotation_font_color=MUTED)
    fig.update_layout(barmode="group", yaxis_title="% with ≥1 dose",
                      yaxis_ticksuffix="%")
    return _style(fig, height=320, legend=True)
