"""
================================================================
Human vs AI Marking — Dissertation Pipeline
Dataset: ASAP 2.0 (github.com/scrosseye/ASAP_2.0)
Author:  Aniqa Arooj — MSc Artificial Intelligence, University of Greater Manchester
================================================================

SETUP INSTRUCTIONS:
1. Download ASAP_2_Final_github_train.csv from GitHub
2. Place it in the same folder as this script
3. Install dependencies:
      pip install pandas numpy scikit-learn openai anthropic scipy tqdm matplotlib seaborn
4. Set the OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables
5. Run:  python asap_dissertation_pipeline.py
   Or, to recompute metrics and charts from saved results without
   calling the APIs:  python asap_dissertation_pipeline.py --analyse-only

The script will:
  - Load and sample 150 essays (stratified by score band)
  - Send each essay to GPT-4o and Claude with a marking prompt
  - Save raw results to  results_raw.csv
  - Calculate QWK, Pearson, MAE, directional bias
  - Save analysis to    results_analysis.csv  and  metrics_summary.txt
  - Generate charts:    results_charts.png
================================================================
"""

import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
from tqdm import tqdm

# ----------------------------------------------------------------
# CONFIGURATION
# API keys are read from environment variables — never write them in code.
#   Windows (PowerShell):  $env:OPENAI_API_KEY="..."; $env:ANTHROPIC_API_KEY="..."
#   macOS / Linux:         export OPENAI_API_KEY="..."; export ANTHROPIC_API_KEY="..."
# ----------------------------------------------------------------
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SAMPLE_SIZE       = 150          # number of essays to score
RANDOM_SEED       = 42              # for reproducibility
DATASET_PATH      = "ASAP_2_Final_github_train.csv"
RESULTS_RAW_PATH  = "results_raw.csv"
RESULTS_ANALYSIS  = "results_analysis.csv"
METRICS_SUMMARY   = "metrics_summary.txt"

# Delay between API calls in seconds — avoids rate limit errors
API_DELAY_SECONDS = 2

# ----------------------------------------------------------------
# MARKING PROMPT
# This is what gets sent to both LLMs for each essay.
# Based on the ASAP 2.0 rubric (1–6 holistic scale).
# You can adjust the wording for your methodology chapter.
# ----------------------------------------------------------------
SYSTEM_PROMPT = """You are an experienced essay marker. 
You will be given a student essay and asked to assign a holistic score.
Respond ONLY with a JSON object in this exact format — nothing else:
{"score": <integer from 1 to 6>, "justification": "<one sentence reason>"}

Scoring rubric:
6 = Consistent mastery of writing — well-organised, persuasive, precise language
5 = Reasonably consistent mastery — minor lapses in organisation or language
4 = Adequate mastery — some lapses but generally competent
3 = Developing mastery — noticeable weaknesses in organisation or development
2 = Inconsistent mastery — significant weaknesses throughout
1 = Little or no mastery — largely incoherent or off-topic"""

def build_user_prompt(essay_text, prompt_text):
    return f"""Essay prompt: {prompt_text}

Student essay:
{essay_text}

Assign a score from 1 to 6."""


# ----------------------------------------------------------------
# STEP 1 — Load and sample the dataset
# ----------------------------------------------------------------
def load_and_sample(path, n=SAMPLE_SIZE, seed=RANDOM_SEED):
    print(f"\n[1/5] Loading dataset from {path}...")
    df = pd.read_csv(path)

    # ASAP 2.0 column names — adjust if your file differs
    # Key columns: essay_id, full_text, score, prompt_name
    required = ["essay_id", "full_text", "score", "prompt_name"]
    for col in required:
        if col not in df.columns:
            print(f"  Column '{col}' not found. Available columns:")
            print(f"  {list(df.columns)}")
            raise ValueError(f"Missing column: {col}. Check column names above and update 'required' list.")

    df = df[required].dropna()
    df["score"] = df["score"].astype(int)

    # Stratified sample — proportional across score bands 1–6
    frames = []
    for score_val, group in df.groupby("score"):
        k = min(len(group), round(n * len(group) / len(df)))
        if k > 0:
            frames.append(group.sample(k, random_state=seed))
    sampled = pd.concat(frames)

    # Top up to exactly n if rounding left us short
    if len(sampled) < n:
        remaining = df[~df["essay_id"].isin(sampled["essay_id"])]
        top_up = remaining.sample(n - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, top_up])

    sampled = sampled.head(n).reset_index(drop=True)
    print(f"  Sampled {len(sampled)} essays.")
    print(f"  Score distribution:\n{sampled['score'].value_counts().sort_index()}\n")
    return sampled


# ----------------------------------------------------------------
# STEP 2 — Query GPT-4o
# ----------------------------------------------------------------
def score_with_gpt(essay_text, prompt_text, client):
    user_msg = build_user_prompt(essay_text, prompt_text)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}
            ],
            max_tokens=150,
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)
        return int(parsed["score"]), parsed.get("justification", "")
    except Exception as e:
        print(f"    GPT error: {e}")
        return None, str(e)


# ----------------------------------------------------------------
# STEP 3 — Query Claude
# ----------------------------------------------------------------
def score_with_claude(essay_text, prompt_text, client):
    user_msg = build_user_prompt(essay_text, prompt_text)
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}]
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        # Extract JSON object if wrapped in other text
        import re
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        parsed = json.loads(raw)
        return int(parsed["score"]), parsed.get("justification", "")
    except Exception as e:
        print(f"    Claude error: {e}")
        return None, str(e)


# ----------------------------------------------------------------
# STEP 4 — Run both models over all sampled essays
# ----------------------------------------------------------------
def run_scoring(df):
    import openai, anthropic
    print("[2/5] Initialising API clients...")
    openai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    gpt_scores, gpt_justifications       = [], []
    claude_scores, claude_justifications = [], []

    print(f"[3/5] Scoring {len(df)} essays — this will take a few minutes...\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring essays"):
        essay   = str(row["full_text"])[:4000]   # truncate very long essays
        prompt  = str(row["prompt_name"])

        # GPT-4o
        g_score, g_just = score_with_gpt(essay, prompt, openai_client)
        gpt_scores.append(g_score)
        gpt_justifications.append(g_just)

        # Claude
        c_score, c_just = score_with_claude(essay, prompt, anthropic_client)
        claude_scores.append(c_score)
        claude_justifications.append(c_just)

        time.sleep(API_DELAY_SECONDS)

    df["gpt_score"]             = gpt_scores
    df["gpt_justification"]     = gpt_justifications
    df["claude_score"]          = claude_scores
    df["claude_justification"]  = claude_justifications

    df.to_csv(RESULTS_RAW_PATH, index=False)
    print(f"\n  Raw results saved to {RESULTS_RAW_PATH}")
    return df


# ----------------------------------------------------------------
# STEP 5 — Calculate all four metrics
# ----------------------------------------------------------------
def safe_qwk(y_true, y_pred, labels=None):
    """QWK with quadratic weights. Handles edge cases."""
    return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)

def calculate_metrics(df):
    print("\n[4/5] Calculating metrics...")

    df_clean = df.copy()

    score_labels = [1, 2, 3, 4, 5, 6]
    results = {}

    for model_name, score_col in [("GPT-4o", "gpt_score"), ("Claude", "claude_score")]:
        # Each model gets its own clean subset — handles partial failures independently
        subset = df_clean.dropna(subset=[score_col]).copy()
        subset[score_col] = subset[score_col].astype(int)

        if len(subset) < 2:
            print(f"  WARNING: Not enough valid scores for {model_name} — skipping metrics")
            results[model_name] = {"error": "insufficient data"}
            continue

        human      = subset["score"].values
        ai_scores  = subset[score_col].values
        diff       = ai_scores - human

        qwk         = safe_qwk(human, ai_scores, labels=score_labels)
        pearson, p  = pearsonr(human, ai_scores)
        mae         = np.mean(np.abs(diff))
        bias        = np.mean(diff)            # positive = AI marks higher, negative = AI marks lower

        results[model_name] = {
            "QWK":                round(qwk, 4),
            "Pearson_r":          round(pearson, 4),
            "Pearson_p":          round(p, 6),
            "MAE":                round(mae, 4),
            "Mean_bias":          round(bias, 4),
            "Bias_direction":     "Over-marking" if bias > 0 else "Under-marking",
            "N_valid":            len(subset),
            "N_exact_match":      int(np.sum(diff == 0)),
            "N_within_1":         int(np.sum(np.abs(diff) <= 1)),
            "Pct_exact_match":    round(100 * np.sum(diff == 0) / len(diff), 1),
            "Pct_within_1":       round(100 * np.sum(np.abs(diff) <= 1) / len(diff), 1),
        }

    # Add per-essay error columns where scores exist
    gpt_subset = df.dropna(subset=["gpt_score"]).copy()
    gpt_subset["gpt_score"] = gpt_subset["gpt_score"].astype(int)
    df["gpt_error"] = df["gpt_score"].sub(df["score"])

    claude_subset = df.dropna(subset=["claude_score"]).copy()
    claude_subset["claude_score"] = claude_subset["claude_score"].astype(int)
    df["claude_error"] = df["claude_score"].sub(df["score"])

    df_clean = df
    df_clean.to_csv(RESULTS_ANALYSIS, index=False)

    return results, df_clean


# ----------------------------------------------------------------
# STEP 6 — Print and save summary
# ----------------------------------------------------------------
def save_summary(results):
    lines = []
    lines.append("=" * 60)
    lines.append("DISSERTATION RESULTS SUMMARY")
    lines.append("Human vs AI Marking — ASAP 2.0 Dataset")
    lines.append("=" * 60)

    for model, m in results.items():
        lines.append(f"\n--- {model} vs Human Raters ---")
        lines.append(f"  N (valid essays):           {m['N_valid']}")
        lines.append(f"  Quadratic Weighted Kappa:   {m['QWK']}")
        p_txt = "< 0.001" if m['Pearson_p'] < 0.001 else f"= {m['Pearson_p']}"
        lines.append(f"  Pearson r:                  {m['Pearson_r']}  (p {p_txt})")
        lines.append(f"  Mean Absolute Error:        {m['MAE']}")
        lines.append(f"  Mean Bias:                  {m['Mean_bias']}  ({m['Bias_direction']})")
        lines.append(f"  Exact match:                {m['N_exact_match']} / {m['N_valid']}  ({m['Pct_exact_match']}%)")
        lines.append(f"  Within 1 mark:              {m['N_within_1']} / {m['N_valid']}  ({m['Pct_within_1']}%)")

    lines.append("\n" + "=" * 60)
    lines.append("QWK interpretation guide:")
    lines.append("  < 0.40  =  Poor agreement")
    lines.append("  0.40–0.59  =  Moderate agreement")
    lines.append("  0.60–0.79  =  Substantial agreement")
    lines.append("  >= 0.80  =  Almost perfect agreement")
    lines.append("=" * 60)

    summary_text = "\n".join(lines)
    print(summary_text)

    with open(METRICS_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"\n  Metrics summary saved to {METRICS_SUMMARY}")


# ----------------------------------------------------------------
# STEP 7 — Generate charts
# ----------------------------------------------------------------
def generate_charts(df_clean, results):
    print("\n[5/5] Generating charts...")
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Human vs AI Marking — ASAP 2.0 Results", fontsize=14, fontweight="bold")

    human = df_clean["score"]
    gpt   = df_clean["gpt_score"]
    claude = df_clean["claude_score"]

    # 1. Score distributions
    ax = axes[0, 0]
    x = np.arange(1, 7)
    w = 0.28
    ax.bar(x - w, [sum(human == i) for i in x],   w, label="Human",  color="#5DCAA5")
    ax.bar(x,     [sum(gpt == i) for i in x],      w, label="GPT-4o", color="#7F77DD")
    ax.bar(x + w, [sum(claude == i) for i in x],   w, label="Claude", color="#F0997B")
    ax.set_title("Score distributions")
    ax.set_xlabel("Score band (1–6)")
    ax.set_ylabel("Count")
    ax.legend()

    # 2. GPT scatter
    ax = axes[0, 1]
    ax.scatter(human, gpt, alpha=0.4, color="#7F77DD", s=20)
    ax.plot([1, 6], [1, 6], "k--", linewidth=1, label="Perfect agreement")
    ax.set_title(f"GPT-4o vs Human  (QWK={results['GPT-4o']['QWK']})")
    ax.set_xlabel("Human score")
    ax.set_ylabel("GPT-4o score")
    ax.legend(fontsize=8)

    # 3. Claude scatter
    ax = axes[0, 2]
    ax.scatter(human, claude, alpha=0.4, color="#F0997B", s=20)
    ax.plot([1, 6], [1, 6], "k--", linewidth=1, label="Perfect agreement")
    ax.set_title(f"Claude vs Human  (QWK={results['Claude']['QWK']})")
    ax.set_xlabel("Human score")
    ax.set_ylabel("Claude score")
    ax.legend(fontsize=8)

    # 4. GPT error distribution
    ax = axes[1, 0]
    ax.hist(df_clean["gpt_error"], bins=np.arange(-5.5, 6.5, 1), color="#7F77DD", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"GPT-4o error  (bias={results['GPT-4o']['Mean_bias']})")
    ax.set_xlabel("AI score − Human score")
    ax.set_ylabel("Count")

    # 5. Claude error distribution
    ax = axes[1, 1]
    ax.hist(df_clean["claude_error"], bins=np.arange(-5.5, 6.5, 1), color="#F0997B", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_title(f"Claude error  (bias={results['Claude']['Mean_bias']})")
    ax.set_xlabel("AI score − Human score")
    ax.set_ylabel("Count")

    # 6. Metrics comparison bar chart
    ax = axes[1, 2]
    metrics  = ["QWK", "Pearson_r", "MAE"]
    gpt_vals = [results["GPT-4o"][m] for m in metrics]
    cla_vals = [results["Claude"][m] for m in metrics]
    x2 = np.arange(len(metrics))
    ax.bar(x2 - 0.2, gpt_vals,  0.4, label="GPT-4o", color="#7F77DD")
    ax.bar(x2 + 0.2, cla_vals,  0.4, label="Claude",  color="#F0997B")
    ax.set_xticks(x2)
    ax.set_xticklabels(["QWK", "Pearson r", "MAE"])
    ax.set_title("Metrics comparison")
    ax.legend()
    ax.set_ylim(0, 1.2)

    plt.tight_layout()
    plt.savefig("results_charts.png", dpi=150, bbox_inches="tight")
    print("  Charts saved to results_charts.png")


# ----------------------------------------------------------------
# MAIN — runs everything in order
# ----------------------------------------------------------------
if __name__ == "__main__":
    import sys
    analyse_only = "--analyse-only" in sys.argv

    print("\n" + "=" * 60)
    print("  Human vs AI Marking — Dissertation Pipeline")
    print("=" * 60)

    if analyse_only:
        # Recompute metrics and charts from saved results (no API calls)
        if not os.path.exists(RESULTS_RAW_PATH):
            print(f"\n  ERROR: {RESULTS_RAW_PATH} not found — run the full pipeline first.")
            exit(1)
        df_scored = pd.read_csv(RESULTS_RAW_PATH)
    else:
        # Check API keys are set
        if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
            print("\n  ERROR: Set the OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables.")
            exit(1)

        # Check dataset exists
        if not os.path.exists(DATASET_PATH):
            print(f"\n  ERROR: Dataset not found at '{DATASET_PATH}'")
            print("  Download from: https://github.com/scrosseye/ASAP_2.0")
            print("  Place the CSV file in the same folder as this script.")
            exit(1)

        df_sample = load_and_sample(DATASET_PATH)
        df_scored = run_scoring(df_sample)

    results, df_clean = calculate_metrics(df_scored)
    save_summary(results)
    generate_charts(df_clean, results)

    print("\n  All done! Output files:")
    print(f"    {RESULTS_RAW_PATH}     — all essays with AI scores and justifications")
    print(f"    {RESULTS_ANALYSIS}  — results with per-essay error columns")
    print(f"    {METRICS_SUMMARY}    — metrics summary")
    print(f"    results_charts.png      — charts")
