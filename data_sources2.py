
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from income_groups import INCOME_GROUP, INCOME_ORDER

UPLOAD_DIR = "/mnt/user-data/uploads"
LOCAL_DIR = "data"
WHO_CSV_URL = "https://covid19.who.int/WHO-COVID-19-global-data.csv"
CASES_FILE = "WHO-COVID-19-global-data.csv"
VAX_FILE = "COV_VAC_UPTAKE_2021_2023.csv"


def _find(filename: str):
    for base in (UPLOAD_DIR, LOCAL_DIR):
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    return None


REGION_NAMES = {
    "AFRO": "Africa", "AMRO": "Americas", "SEARO": "South-East Asia",
    "EURO": "Europe", "EMRO": "Eastern Mediterranean",
    "WPRO": "Western Pacific", "OTHER": "Other",
}
REGION_ORDER = ["AFRO", "AMRO", "SEARO", "EURO", "EMRO", "WPRO", "OTHER"]
REGION_POP = {
    "AFRO": 1_140_000_000, "AMRO": 1_020_000_000, "SEARO": 2_050_000_000,
    "EURO": 935_000_000, "EMRO": 745_000_000, "WPRO": 1_940_000_000,
    "OTHER": 3_000_000,
}

EXCESS_GLOBAL = {
    "reported_deaths_2020_2021_m": 5.4,
    "excess_deaths_2020_2021_m": 14.9,
    "excess_low_m": 13.3,
    "excess_high_m": 16.6,
    "undercount_ratio": 2.75,
}
EXCESS_BY_REGION = pd.DataFrame(
    [
        ("Africa",                  "AFRO",  1.25),
        ("Americas",                "AMRO",  3.23),
        ("Europe",                  "EURO",  3.25),
        ("South & South-East Asia", "SEARO", 5.99),
    ],
    columns=["study_region", "who_region", "excess_deaths_m"],
)
EXCESS_BY_INCOME = pd.DataFrame(
    [
        ("Low income",          4),
        ("Lower-middle income", 53),
        ("Upper-middle income", 28),
        ("High income",         15),
    ],
    columns=["income_group", "share_of_excess_pct"],
)


def _normalise_columns(df):
    return df.rename(columns={c: c.strip().lstrip("\ufeff") for c in df.columns})


def clean_timeseries(df):
    df = _normalise_columns(df).copy()
    df["Date_reported"] = pd.to_datetime(df["Date_reported"], errors="coerce")
    df = df.dropna(subset=["Date_reported"])
    for c in ["New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["New_cases", "New_deaths"]:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)
    df["WHO_region"] = df["WHO_region"].fillna("OTHER").replace("", "OTHER")
    df["Region_name"] = df["WHO_region"].map(REGION_NAMES).fillna(df["WHO_region"])
    return df.sort_values("Date_reported").reset_index(drop=True)


def load_timeseries():
    path = _find(CASES_FILE)
    if path:
        df = clean_timeseries(pd.read_csv(path))
        return df, f"WHO COVID-19 global data ({os.path.basename(path)})", False
    try:
        df = clean_timeseries(pd.read_csv(WHO_CSV_URL))
        if len(df) > 1000:
            return df, "Live WHO COVID-19 global data (covid19.who.int)", False
    except Exception:
        pass
    return generate_synthetic_timeseries(), "SYNTHETIC demo data (offline)", True


def load_vaccine_equity():
    path = _find(VAX_FILE)
    if not path:
        return None, None, "vaccine file not found"
    v = pd.read_csv(path, parse_dates=["DATE"], low_memory=False)
    v.columns = [c.strip().lstrip("\ufeff") for c in v.columns]
    v["income_group"] = v["COUNTRY"].map(INCOME_GROUP)
    v = v.dropna(subset=["income_group", "COVID_VACCINE_COV_TOT_A1D"])
    v = v[v["COVID_VACCINE_COV_TOT_A1D"].between(0, 100)]
    long_df = (v.groupby(["income_group", "DATE"], as_index=False)
                 ["COVID_VACCINE_COV_TOT_A1D"].median()
                 .rename(columns={"COVID_VACCINE_COV_TOT_A1D": "coverage_pct"}))

    def snap(cutoff):
        sub = v[v.DATE <= pd.Timestamp(cutoff)]
        latest = sub.sort_values("DATE").groupby("COUNTRY").tail(1)
        return latest.groupby("income_group")["COVID_VACCINE_COV_TOT_A1D"].median()

    end21, latest = snap("2021-12-31"), snap(v.DATE.max())
    snapshots = pd.DataFrame({
        "income_group": INCOME_ORDER,
        "end_2021_pct": [round(float(end21.get(g, np.nan)), 1) for g in INCOME_ORDER],
        "latest_pct": [round(float(latest.get(g, np.nan)), 1) for g in INCOME_ORDER],
    })
    label = f"Derived from {os.path.basename(path)} ({v.COUNTRY.nunique()} countries)"
    return long_df, snapshots, label


def region_weekly(df):
    g = (df.groupby(["Date_reported", "WHO_region", "Region_name"], as_index=False)
           [["New_cases", "New_deaths"]].sum())
    return g.sort_values("Date_reported")


def add_rolling_and_cfr(region_df, window=4):
    out = []
    for region, sub in region_df.groupby("WHO_region"):
        sub = sub.sort_values("Date_reported").copy()
        sub["cases_smooth"] = sub["New_cases"].rolling(window, min_periods=1).mean()
        sub["deaths_smooth"] = sub["New_deaths"].rolling(window, min_periods=1).mean()
        roll_c = sub["New_cases"].rolling(window, min_periods=1).sum()
        roll_d = sub["New_deaths"].rolling(window, min_periods=1).sum()
        sub["cfr_pct"] = np.where(roll_c > 0, 100 * roll_d / roll_c, np.nan)
        pop = REGION_POP.get(region, np.nan)
        sub["cases_per_100k"] = sub["cases_smooth"] / pop * 100_000
        sub["deaths_per_100k"] = sub["deaths_smooth"] / pop * 100_000
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def headline_numbers(df):
    tc, td = int(df["New_cases"].sum()), int(df["New_deaths"].sum())
    return {
        "total_cases": tc, "total_deaths": td,
        "overall_cfr_pct": 100 * td / tc if tc else 0.0,
        "first_date": df["Date_reported"].min(),
        "last_date": df["Date_reported"].max(),
        "n_countries": df["Country"].nunique(),
    }


def reported_deaths_by_region_through(df, cutoff="2021-12-31"):
    sub = df[df["Date_reported"] <= pd.Timestamp(cutoff)]
    g = sub.groupby("WHO_region", as_index=False)["New_deaths"].sum()
    return g.rename(columns={"New_deaths": "reported_deaths"})


def undercount_table(df):
    rep = reported_deaths_by_region_through(df, "2021-12-31")
    rep = rep.rename(columns={"WHO_region": "who_region"})
    merged = EXCESS_BY_REGION.merge(rep, on="who_region", how="left")
    merged["reported_deaths"] = merged["reported_deaths"].fillna(0)
    merged["reported_deaths_m"] = merged["reported_deaths"] / 1_000_000
    merged["excess_deaths"] = merged["excess_deaths_m"] * 1_000_000
    merged["undercount_factor"] = np.where(
        merged["reported_deaths"] > 0,
        merged["excess_deaths"] / merged["reported_deaths"], np.nan)
    merged["Region_name"] = merged["who_region"].map(REGION_NAMES)
    return merged


def country_weekly(df, window=4):
    out = []
    for country, sub in df.groupby("Country"):
        sub = sub.sort_values("Date_reported").copy()
        sub["cases_smooth"] = sub["New_cases"].rolling(window, min_periods=1).mean()
        sub["deaths_smooth"] = sub["New_deaths"].rolling(window, min_periods=1).mean()
        out.append(sub[["Date_reported", "Country", "WHO_region",
                        "cases_smooth", "deaths_smooth"]])
    return pd.concat(out, ignore_index=True)


def top_countries(df, by="New_cases", n=40):
    totals = df.groupby("Country")[by].sum().sort_values(ascending=False)
    return totals.head(n).index.tolist()


def generate_synthetic_timeseries(seed=42):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2020-01-06", "2023-06-26", freq="W-MON")
    t = np.arange(len(weeks), dtype=float)
    region_waves = {
        "EURO":  [(12, 6, 1.0), (45, 8, 1.4), (95, 6, 3.2), (150, 7, 1.1)],
        "AMRO":  [(16, 7, 1.2), (50, 9, 1.6), (96, 6, 2.8), (152, 8, 0.9)],
        "SEARO": [(20, 6, 0.5), (70, 9, 2.4), (98, 5, 3.6), (156, 7, 0.7)],
        "EMRO":  [(22, 7, 0.4), (60, 9, 0.9), (100, 6, 1.0), (158, 8, 0.4)],
        "AFRO":  [(24, 7, 0.20), (64, 9, 0.45), (102, 6, 0.55), (160, 8, 0.18)],
        "WPRO":  [(10, 5, 0.3), (110, 8, 1.4), (150, 6, 4.0), (175, 7, 1.0)],
    }
    region_scale = {"EURO": 1.6, "AMRO": 1.5, "SEARO": 1.4,
                    "EMRO": 0.5, "AFRO": 0.35, "WPRO": 1.7}
    sample = {
        "EURO": [("United Kingdom", "GB"), ("Italy", "IT")],
        "AMRO": [("United States of America", "US"), ("Brazil", "BR")],
        "SEARO": [("India", "IN"), ("Indonesia", "ID")],
        "EMRO": [("Iran", "IR"), ("Egypt", "EG")],
        "AFRO": [("South Africa", "ZA"), ("Nigeria", "NG")],
        "WPRO": [("China", "CN"), ("Japan", "JP")],
    }
    rows = []
    for region, waves in region_waves.items():
        cases = np.zeros_like(t)
        for peak, width, height in waves:
            cases += height * np.exp(-0.5 * ((t - peak) / width) ** 2)
        cases *= region_scale[region] * 1_000_000
        cases *= rng.normal(1.0, 0.06, size=cases.shape).clip(0.6, 1.4)
        cfr0, decay = 0.045, 60.0
        floor = {"AFRO": 0.005, "SEARO": 0.004, "EMRO": 0.004}.get(region, 0.0025)
        cfr = floor + (cfr0 - floor) * np.exp(-t / decay)
        deaths = cases * cfr * rng.normal(1.0, 0.05, size=cases.shape).clip(0.7, 1.3)
        for (cname, ccode), frac in zip(sample[region], (0.6, 0.4)):
            rows.append(pd.DataFrame({
                "Date_reported": weeks, "Country_code": ccode, "Country": cname,
                "WHO_region": region,
                "New_cases": np.round(cases * frac).astype(int),
                "New_deaths": np.round(deaths * frac).astype(int),
            }))
    df = pd.concat(rows, ignore_index=True).sort_values(["Country", "Date_reported"])
    df["Cumulative_cases"] = df.groupby("Country")["New_cases"].cumsum()
    df["Cumulative_deaths"] = df.groupby("Country")["New_deaths"].cumsum()
    return clean_timeseries(df)
