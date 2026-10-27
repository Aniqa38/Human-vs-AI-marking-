"""Fairness (differential bias) analysis for the Human-vs-AI marking study.

Links results_analysis.csv to the demographic columns of the ASAP 2.0 training file and
compares scoring bias (AI - human) between student groups, within human-score bands.

Usage:  python fairness_analysis.py
Needs:  results_analysis.csv and ASAP_2_Final_github_train.csv in the same folder
        pip install pandas numpy scipy statsmodels
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

N_PERM = 5000
rng = np.random.default_rng(42)

r = pd.read_csv("results_analysis.csv")
a = pd.read_csv("ASAP_2_Final_github_train.csv")
m = r.merge(a.drop(columns=["full_text", "score", "prompt_name"]), on="essay_id", how="left")
m["ELL"] = m.ell_status.map({"Yes": "ELL", "No": "Non-ELL"})
m["Econ"] = m.economically_disadvantaged.map({"Economically disadvantaged": "Disadvantaged",
                                              "Not economically disadvantaged": "Not disadvantaged"})
m["Disab"] = m.student_disability_status.map({"Identified as having disability": "Disability",
                                              "Not identified as having disability": "No disability"})
m["Gender"] = m.gender.map({"F": "Female", "M": "Male"})
m["Race"] = m.race_ethnicity.replace({"Asian/Pacific Islander": "Other",
                                      "Two or more races/Other": "Other",
                                      "American Indian/Alaskan Native": "Other"})
VARS = [("ELL", "ELL status", "Non-ELL"), ("Econ", "Economic disadvantage", "Not disadvantaged"),
        ("Disab", "Disability status", "No disability"), ("Gender", "Gender", "Male"),
        ("Race", "Race/ethnicity", "White")]
MODELS = [("GPT-4o", "gpt_score"), ("Claude", "claude_score")]


def within_band_diff(d, ref):
    """Mean error of each group minus the reference group, within human-score bands."""
    out = {}
    for g in d.grp.unique():
        if g == ref:
            continue
        num = w = 0
        for _, b in d.groupby("score"):
            gi, ri = b[b.grp == g].err, b[b.grp == ref].err
            if len(gi) and len(ri):
                num += len(gi) * (gi.mean() - ri.mean())
                w += len(gi)
        out[g] = num / w
    return out


rows, omni_p = [], []
for col, label, ref in VARS:
    for model, sc in MODELS:
        d = m.dropna(subset=[col, sc]).copy()
        d["err"], d["grp"] = d[sc] - d.score, d[col]
        obs = within_band_diff(d, ref)
        omni_obs = max(abs(v) for v in obs.values())
        cnt = {g: 0 for g in obs}
        omni = 0
        for _ in range(N_PERM):
            dp = d.copy()
            dp["grp"] = dp.groupby("score").grp.transform(lambda s: rng.permutation(s.values))
            st = within_band_diff(dp, ref)
            for g in obs:
                cnt[g] += abs(st[g]) >= abs(obs[g]) - 1e-12
            omni += max(abs(v) for v in st.values()) >= omni_obs - 1e-12
        # robustness check: OLS with human-score band fixed effects, HC1 errors
        fit = smf.ols(f"err ~ C(score) + C(grp, Treatment(reference='{ref}'))", data=d).fit(cov_type="HC1")
        for g in obs:
            t = f"C(grp, Treatment(reference='{ref}'))[T.{g}]"
            rows.append(dict(variable=label, model=model, group=g, reference=ref,
                             n=int((d.grp == g).sum()), diff=round(obs[g], 3),
                             p_perm=round((cnt[g] + 1) / (N_PERM + 1), 4),
                             ols_coef=round(fit.params[t], 3), ols_p=round(fit.pvalues[t], 4)))
        omni_p.append(dict(variable=label, model=model, p_omnibus=(omni + 1) / (N_PERM + 1)))

res = pd.DataFrame(rows)
om = pd.DataFrame(omni_p)
om["p_holm"] = multipletests(om.p_omnibus, method="holm")[1]
print(res.to_string(index=False))
print()
print(om.round(4).to_string(index=False))
res.to_csv("fairness_results.csv", index=False)
