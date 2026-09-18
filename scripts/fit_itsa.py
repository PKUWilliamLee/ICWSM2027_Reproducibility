from __future__ import annotations

from pathlib import Path
import argparse
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "derived" / "empirical"
OUT = ROOT / "outputs" / "tables"

THEMES = [
    ("schoolwork_burden", "Schoolwork Burden"),
    ("shadow_education", "Shadow Education"),
    ("refund_disputes", "Refund Disputes"),
    ("after_school_services", "After-school Services"),
    ("family_pressure", "Family Pressure"),
    ("student_wellbeing", "Student Well-being"),
    ("admissions_equity", "Admissions Equity"),
]

def theme_chinese_map() -> dict[str, str]:
    expected = pd.read_csv(DATA / "all_specifications_terms_long.csv", usecols=["theme", "theme_chinese"])
    return expected.drop_duplicates().set_index("theme")["theme_chinese"].to_dict()
HAC_LAGS = 12


def add_time_terms(panel: pd.DataFrame, cutoff: pd.Timestamp, quadratic: bool = False) -> pd.DataFrame:
    out = panel.copy().sort_values("ym").reset_index(drop=True)
    out["time"] = ((out["ym"].dt.year - cutoff.year) * 12 + (out["ym"].dt.month - cutoff.month)).astype(float)
    out["post_policy"] = (out["ym"] >= cutoff).astype(int)
    out["post_time"] = np.where(out["post_policy"].eq(1), out["time"], 0.0)
    m = out["ym"].dt.month.to_numpy(float)
    out["sin12"] = np.sin(2.0 * np.pi * (m - 1.0) / 12.0)
    out["cos12"] = np.cos(2.0 * np.pi * (m - 1.0) / 12.0)
    if quadratic:
        out["time_sq"] = (out["time"] / 12.0) ** 2
    return out


def add_month_fixed_effects(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for month in range(2, 13):
        out[f"month_{month:02d}"] = out["ym"].dt.month.eq(month).astype(int)
    return out


def fit_glm(panel: pd.DataFrame, outcome: str, columns: list[str]):
    y = panel[outcome].astype(float)
    X = sm.add_constant(panel[columns].astype(float), has_constant="add")
    model = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.Logit()))
    result = model.fit(
        maxiter=200,
        tol=1e-10,
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS, "use_correction": True},
    )
    if not result.converged:
        raise RuntimeError(f"Model did not converge: {outcome}")
    return result


def coefficient_rows(specification: str, theme_zh: str, theme: str, result) -> list[dict]:
    ci = result.conf_int()
    rows = []
    for term in result.params.index:
        rows.append(
            {
                "specification": specification,
                "theme_chinese": theme_zh,
                "theme": theme,
                "term": term,
                "coefficient": float(result.params[term]),
                "std_error": float(result.bse[term]),
                "z_value": float(result.tvalues[term]),
                "p_value": float(result.pvalues[term]),
                "ci_low": float(ci.loc[term, 0]),
                "ci_high": float(ci.loc[term, 1]),
                "n_months": int(result.nobs),
                "converged": bool(result.converged),
                "q_value_bh": np.nan,
                "significant_fdr_005": False,
            }
        )
    names = list(result.params.index)
    if "time" in names and "post_time" in names:
        contrast = np.zeros(len(names), dtype=float)
        contrast[names.index("time")] = 1.0
        contrast[names.index("post_time")] = 1.0
        coef = float(contrast @ result.params.to_numpy(float))
        cov = np.asarray(result.cov_params(), dtype=float)
        se = float(np.sqrt(contrast @ cov @ contrast))
        z = coef / se
        from scipy.stats import norm
        p_value = float(2.0 * norm.sf(abs(z)))
        rows.append({
            "specification": specification,
            "theme_chinese": theme_zh,
            "theme": theme,
            "term": "post_period_slope",
            "coefficient": coef,
            "std_error": se,
            "z_value": z,
            "p_value": p_value,
            "ci_low": coef - 1.96 * se,
            "ci_high": coef + 1.96 * se,
            "n_months": int(result.nobs),
            "converged": bool(result.converged),
            "q_value_bh": np.nan,
            "significant_fdr_005": False,
        })
    return rows


def effect_row(
    specification: str,
    theme_zh: str,
    theme: str,
    result,
    panel: pd.DataFrame,
    date: pd.Timestamp,
    horizon: int,
    columns: list[str],
) -> dict:
    row = panel.loc[panel["ym"].eq(date)].iloc[0]
    x = pd.Series({"const": 1.0, **{c: float(row[c]) for c in columns}}).reindex(result.params.index).to_numpy(float)
    x0 = x.copy()
    names = list(result.params.index)
    for term in ("post_policy", "post_time"):
        if term in names:
            x0[names.index(term)] = 0.0
    beta = result.params.to_numpy(float)
    eta = float(x @ beta)
    eta0 = float(x0 @ beta)
    p = 1.0 / (1.0 + math.exp(-eta))
    p0 = 1.0 / (1.0 + math.exp(-eta0))
    grad = p * (1.0 - p) * x - p0 * (1.0 - p0) * x0
    cov = np.asarray(result.cov_params(), dtype=float)
    se = float(np.sqrt(max(0.0, grad @ cov @ grad)))
    effect = p - p0
    return {
        "specification": specification,
        "theme_chinese": theme_zh,
        "theme": theme,
        "date": date.strftime("%Y-%m-%d"),
        "horizon_months": horizon,
        "predicted_with_policy": p,
        "counterfactual_no_policy": p0,
        "effect_share": effect,
        "effect_pp": effect * 100.0,
        "se_effect_pp": se * 100.0,
        "ci_low_pp": (effect - 1.96 * se) * 100.0,
        "ci_high_pp": (effect + 1.96 * se) * 100.0,
        "n_months": int(result.nobs),
    }


def fit_spec(panel: pd.DataFrame, prefix: str, specification: str, cutoff: pd.Timestamp, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    terms: list[dict] = []
    effects: list[dict] = []
    zh_map = theme_chinese_map()
    for slug, theme in THEMES:
        theme_zh = zh_map[theme]
        outcome = f"{prefix}_share_{slug}"
        result = fit_glm(panel, outcome, columns)
        terms.extend(coefficient_rows(specification, theme_zh, theme, result))
        for horizon in (0, 12, 24, 36):
            date = cutoff + pd.DateOffset(months=horizon)
            effects.append(effect_row(specification, theme_zh, theme, result, panel, date, horizon, columns))
    terms_df = pd.DataFrame(terms)
    for term in ("post_policy", "post_time"):
        mask = terms_df["term"].eq(term)
        reject, qvals, _, _ = multipletests(terms_df.loc[mask, "p_value"].to_numpy(), alpha=0.05, method="fdr_bh")
        terms_df.loc[mask, "q_value_bh"] = qvals
        terms_df.loc[mask, "significant_fdr_005"] = reject
    return terms_df, pd.DataFrame(effects)


def run(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = pd.read_csv(DATA / "monthly_theme_panel.csv", parse_dates=["ym"])
    cutoff = pd.Timestamp("2021-08-01")
    main_cols = ["time", "post_policy", "post_time", "sin12", "cos12"]
    quad_cols = ["time", "time_sq", "post_policy", "post_time", "sin12", "cos12"]
    month_cols = ["time", "post_policy", "post_time"] + [f"month_{m:02d}" for m in range(2, 13)]

    terms_all = []
    effects_all = []

    main_panel = add_time_terms(monthly, cutoff)
    t, e = fit_spec(main_panel, "primary", "main_primary_share_aug2021", cutoff, main_cols)
    terms_all.append(t); effects_all.append(e)

    quad_panel = add_time_terms(monthly, cutoff, quadratic=True)
    t, e = fit_spec(quad_panel, "primary", "quadratic_primary_share_aug2021", cutoff, quad_cols)
    terms_all.append(t); effects_all.append(e)

    for spec, alt_cutoff in [
        ("announcement_2021_07", pd.Timestamp("2021-07-01")),
        ("delayed_2021_09", pd.Timestamp("2021-09-01")),
    ]:
        panel = add_time_terms(monthly, alt_cutoff)
        t, e = fit_spec(panel, "primary", spec, alt_cutoff, main_cols)
        terms_all.append(t); effects_all.append(e)

    any_panel = add_time_terms(monthly, cutoff)
    t, e = fit_spec(any_panel, "any", "anylabel_share_aug2021", cutoff, main_cols)
    terms_all.append(t); effects_all.append(e)

    end_2024 = monthly.loc[monthly["ym"] <= pd.Timestamp("2024-12-01")].copy()
    end_2024_panel = add_time_terms(end_2024, cutoff)
    t, e = fit_spec(end_2024_panel, "primary", "exclude_2025_primary_share", cutoff, main_cols)
    terms_all.append(t); effects_all.append(e)

    month_panel = add_month_fixed_effects(add_time_terms(monthly, cutoff))
    t, e = fit_spec(month_panel, "primary", "month_fe_primary_share_aug2021", cutoff, month_cols)
    terms_all.append(t); effects_all.append(e)

    terms = pd.concat(terms_all, ignore_index=True)
    effects = pd.concat(effects_all, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    terms.to_csv(output_dir / "itsa_all_specifications_terms.csv", index=False)
    effects.to_csv(output_dir / "itsa_all_specifications_effects.csv", index=False)
    return terms, effects


def compare_to_frozen(terms: pd.DataFrame, effects: pd.DataFrame) -> dict[str, float]:
    expected_terms = pd.read_csv(DATA / "all_specifications_terms_long.csv")
    expected_effects = pd.read_csv(DATA / "all_specifications_effects_long.csv")
    key_t = ["specification", "theme_chinese", "theme", "term"]
    merged_t = expected_terms.merge(terms, on=key_t, suffixes=("_expected", "_actual"), how="inner")
    key_e = ["specification", "theme_chinese", "theme", "date", "horizon_months"]
    merged_e = expected_effects.merge(effects, on=key_e, suffixes=("_expected", "_actual"), how="inner")
    return {
        "term_rows_expected": int(len(expected_terms)),
        "term_rows_matched": int(len(merged_t)),
        "max_abs_coefficient_diff": float((merged_t["coefficient_expected"] - merged_t["coefficient_actual"]).abs().max()),
        "max_abs_se_diff": float((merged_t["std_error_expected"] - merged_t["std_error_actual"]).abs().max()),
        "effect_rows_expected": int(len(expected_effects)),
        "effect_rows_matched": int(len(merged_e)),
        "max_abs_effect_pp_diff": float((merged_e["effect_pp_expected"] - merged_e["effect_pp_actual"]).abs().max()),
        "max_abs_effect_se_pp_diff": float((merged_e["se_effect_pp_expected"] - merged_e["se_effect_pp_actual"]).abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refit the interrupted time-series models from the released monthly aggregate panel.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    terms, effects = run(args.output_dir)
    audit = compare_to_frozen(terms, effects)
    print(pd.Series(audit).to_string())
    if audit["term_rows_matched"] != audit["term_rows_expected"] or audit["effect_rows_matched"] != audit["effect_rows_expected"]:
        raise SystemExit("Frozen-result row coverage mismatch.")
    if audit["max_abs_coefficient_diff"] > 1e-8 or audit["max_abs_se_diff"] > 1e-8:
        raise SystemExit("Coefficient reproduction check failed.")
    if audit["max_abs_effect_pp_diff"] > 1e-6 or audit["max_abs_effect_se_pp_diff"] > 1e-6:
        raise SystemExit("Effect reproduction check failed.")


if __name__ == "__main__":
    main()
