# app.py

import json
import math
from datetime import datetime, time

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Helpers: JSON parsing & typing
# -----------------------------

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file=None, path=None, nrows=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, nrows=nrows)
    elif path is not None:
        df = pd.read_csv(path, nrows=nrows)
    else:
        raise ValueError("Either uploaded_file or path must be provided.")
    return df


def parse_result(cell):
    if pd.isna(cell):
        return {}
    if isinstance(cell, dict):
        return cell
    try:
        return json.loads(cell)
    except Exception:
        # Try to fix common issues (e.g. stray whitespace)
        try:
            return json.loads(str(cell).strip())
        except Exception:
            return {}


def extract_text_fields(obj):
    """
    Very simple recursive extractor of free text from a result JSON.
    Returns a single concatenated string.
    """
    texts = []

    def _walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(v, str):
                    # Skip obvious IDs / codes
                    if len(v.strip()) > 0 and not v.startswith("item"):
                        texts.append(v)
                else:
                    _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)

    _walk(obj)
    return " | ".join(texts)


# -----------------------------
# Report type classification
# -----------------------------

def classify_report(row, result_dict):
    score = row.get("score")
    report_id = row.get("report_id")

    # 1) Quizzes / training e-learning (score not null)
    if not pd.isna(score):
        return "quiz"

    keys = set(result_dict.keys())

    # 2) Produce audits: avocados, etc.
    if any(
        k.startswith("HASS - On Display")
        or k.startswith("SHEPARD - On Display")
        or "Avocado" in k
        for k in keys
    ):
        return "produce_audit"

    # 3) Store visits / range census
    if (
        "FULLY AUTOMATIC COFFEE" in keys
        or "HANDHELD GARMENT STEAMER" in keys
        or "Which Retailer are you in?" in keys
    ):
        return "store_visit"

    # 4) AT / conference / BA shift logs
    if "Date of shift" in keys or "Location of sampling" in keys:
        return "shift"

    # 5) Tastings / activations – inventory + comments + weather, etc.
    if (
        "Inventory Counts" in keys
        or "Samples" in keys
        or ("Weather" in keys and "Store Traffic" in keys)
    ):
        return "tasting"

    # 6) Visit-only forms (no range census)
    if "How is this visit being conducted?" in keys:
        return "store_visit"

    return "other"


def classify_all_reports(df):
    df = df.copy()
    df["result_dict"] = df["result"].apply(parse_result)
    df["report_type"] = df.apply(
        lambda r: classify_report(r, r["result_dict"]), axis=1
    )
    df["free_text"] = df["result_dict"].apply(extract_text_fields)
    return df


# -----------------------------
# Flattening per domain
# -----------------------------

def safe_get(d, key, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def to_float(x):
    try:
        if isinstance(x, str):
            x = x.replace("$", "").replace("%", "").strip()
        return float(x)
    except Exception:
        return np.nan

def to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("yes", "true", "1")
    return None

def flatten_tasting(row):
    r = row["result_dict"]
    out = {
        "id": row["id"],
        "report_id": row["report_id"],
        "user_id": row["user_id"],
        "partner_website_id": row["partner_website_id"],
        "created_at": row["created_at"],
    }

    # Interactions: many different field names – use a few heuristics
    possible_interactions = [
        safe_get(r, "question6"),
        safe_get(r, "Number of people spoken to about AT services"),
        safe_get(r, "question2", {})
        if isinstance(r.get("question2"), dict)
        else None,
    ]
    # If question2 is dict{text1,text2,text3} with counts, sum them
    q2 = r.get("question2")
    if isinstance(q2, dict):
        vals = [to_float(v) for v in q2.values()]
        possible_interactions.append(sum([v for v in vals if not math.isnan(v)]))

    out["interactions"] = next(
        (
            to_float(v)
            for v in possible_interactions
            if v not in (None, "") and not (isinstance(v, dict))
        ),
        np.nan,
    )

    # Samples: many templates use nested "Sample & Temp Records"
    samples = np.nan
    st_rec = safe_get(r, "Sample & Temp Records")
    if isinstance(st_rec, dict) and "Samples" in st_rec:
        # treat presence as 1 sample batch; in many of your forms, there is no explicit count
        samples = 1
    # Some forms have explicit "Samples" field
    if "Samples" in r and isinstance(r["Samples"], (int, float, str)):
        samples = to_float(r["Samples"])
    out["samples"] = samples

    # Units sold = sum Sold in "Inventory Counts" where present
    units_sold = np.nan
    inv = r.get("Inventory Counts")
    if isinstance(inv, list):
        sold_vals = [to_float(item.get("Sold")) for item in inv if "Sold" in item]
        sold_vals = [v for v in sold_vals if not math.isnan(v)]
        if sold_vals:
            units_sold = sum(sold_vals)
    out["units_sold"] = units_sold

    # Conversion rate
    if not math.isnan(units_sold) and not math.isnan(samples) and samples > 0:
        out["conversion_rate"] = units_sold / samples
    else:
        out["conversion_rate"] = np.nan

    out["weather"] = safe_get(r, "Weather")
    out["store_traffic"] = safe_get(r, "Store Traffic")
    out["price_flag"] = bool(safe_get(r, "Price", False))
    out["taste_flag"] = bool(safe_get(r, "Taste", False))
    out["packaging_flag"] = bool(safe_get(r, "Packaging", False))
    out["opinion_on_success"] = safe_get(r, "Opinion on Success")
    return out


def flatten_produce(row):
    r = row["result_dict"]
    out = {
        "id": row["id"],
        "report_id": row["report_id"],
        "user_id": row["user_id"],
        "partner_website_id": row["partner_website_id"],
        "created_at": row["created_at"],
        "store_position": safe_get(r, "What position in store are the loose  Avocados located?"),
        "hass_on_display": safe_get(r, "HASS - On Display"),
        "shepard_on_display": safe_get(r, "SHEPARD - On Display"),
        "hass_quality": safe_get(r, "HASS - What was the quality of the fruit?"),
        "shepard_quality": safe_get(r, "SHEPARD - What is the quality rating of the fruit?"),
        "hass_damage_pct": safe_get(r, "HASS - What percentage of all the fruit on display had external damage?"),
        "shepard_damage_pct": safe_get(r, "SHEPARD - What percentage of all the fruit on display had external damage?"),
        "hass_before_rating": safe_get(r, "HASS - BEFORE MERCHANDISING How would you rate the overall presentation of the Avocado display?"),
        "hass_after_rating": safe_get(r, "HASS - AFTER MERCHANDISING How would you rate the overall presentation of the Avocado display?"),
        "shepard_before_rating": safe_get(r, "SHEPARD - BEFORE MERCHANDISING How would you rate the overall presentation of the Avocado display?"),
        "shepard_after_rating": safe_get(r, "SHEPARD - AFTER MERCHANDISING How would you rate the overall presentation of the Avocado display?"),
        "boh_temp_hass": safe_get(r, "HASS - What temperature was the fruit stored at BOH?"),
        "boh_temp_shepard": safe_get(r, "SHEPARD - What temperature was the fruit stored at BOH?"),
        "pos_hass": ",".join(safe_get(r, "HASS - What Point of Sale/signage is on display with the Hass display?", [])),
        "pos_shepard": ",".join(safe_get(r, "SHEPARD - What Point of Sale/signage is on display with the Shepard display?", [])),
        "shoppers_engaged": to_float(safe_get(r, "How many shoppers engaged with the Avocado shelf?")),
        "shoppers_conversion_pct": safe_get(r, "What percentage of these shoppers ended up selecting for purchase?"),
    }

    # crude execution score: good quality + low damage + POS presence
    score = 0
    for field in [
        "hass_quality",
        "shepard_quality",
        "hass_before_rating",
        "hass_after_rating",
        "shepard_before_rating",
        "shepard_after_rating",
    ]:
        if isinstance(out[field], str) and out[field].startswith(("Good", "Excellent")):
            score += 1

    # low damage
    for field in ["hass_damage_pct", "shepard_damage_pct"]:
        v = out[field]
        if isinstance(v, str) and ("0%" in v or "1-10%" in v):
            score += 1

    # POS present
    if out["pos_hass"]:
        score += 1
    if out["pos_shepard"]:
        score += 1

    out["execution_score"] = score
    return out


def flatten_store_visit(row):
    r = row["result_dict"]
    out = {
        "id": row["id"],
        "report_id": row["report_id"],
        "user_id": row["user_id"],
        "partner_website_id": row["partner_website_id"],
        "created_at": row["created_at"],
        "visit_conducted": safe_get(r, "How is this visit being conducted?"),
        "extra_placement": safe_get(
            r,
            "Have you been able to secure additional product placement during this visit? EG: End cap or bulk stack",
        ),
        "pos_compliant": safe_get(
            r, "Is all POS in line with the Account Plan & POS Guidelines?"
        ),
        "comment": safe_get(
            r,
            "Comment below any insights, store feedback or competitor activity",
        ),
    }
    return out


def flatten_shift(row):
    r = row["result_dict"]
    out = {
        "id": row["id"],
        "report_id": row["report_id"],
        "user_id": row["user_id"],
        "partner_website_id": row["partner_website_id"],
        "created_at": row["created_at"],
        "date_of_shift": safe_get(r, "Date of shift"),
        "time_of_shift": safe_get(r, "Time of shift"),
        "location": safe_get(r, "Location") or safe_get(r, "Location of sampling"),
        "people_spoken": to_float(
            safe_get(r, "Number of people spoken to about AT services")
        ),
        "comments": safe_get(r, "Comments on how your shift went"),
        "recommendation": safe_get(r, "Recommendation for future shift")
        or safe_get(r, "Please state 3 recommendations for future campaigns"),
    }
    # interactions per hour (rough)
    duration_hours = np.nan
    try:
        t = str(out["time_of_shift"])
        if "-" in t:
            start, end = [x.strip() for x in t.split("-")]
            start_dt = datetime.strptime(start.replace(".", ":"), "%H:%M")
            end_dt = datetime.strptime(end.replace(".", ":"), "%H:%M")
            duration_hours = (end_dt - start_dt).seconds / 3600
    except Exception:
        pass
    out["duration_hours"] = duration_hours
    if not math.isnan(out["people_spoken"] or np.nan) and duration_hours and duration_hours > 0:
        out["interactions_per_hour"] = out["people_spoken"] / duration_hours
    else:
        out["interactions_per_hour"] = np.nan
    return out


def flatten_quiz(row):
    r = row["result_dict"]
    return {
        "id": row["id"],
        "report_id": row["report_id"],
        "user_id": row["user_id"],
        "partner_website_id": row["partner_website_id"],
        "created_at": row["created_at"],
        "score": to_float(row["score"]),
        "num_questions": sum(
            1 for k, v in r.items() if k.lower().startswith("question") or "?" in k
        ),
    }


# -----------------------------
# Simple sentiment & themes
# -----------------------------
POS_WORDS = {"great", "good", "nice", "tasty", "delicious", "smooth", "creamy", "love", "liked", "excellent"}
NEG_WORDS = {"bad", "awful", "hate", "dislike", "too sweet", "too sugary", "fake", "bitter", "expensive", "pricey"}
THEME_KEYWORDS = {
    "price": ["price", "expensive", "$", "cost"],
    "taste": ["taste", "tasty", "flavour", "flavor", "sweet", "bitter", "sugary", "creamy"],
    "stock": ["no stock", "out of stock", "sold out"],
    "space": ["no room", "no space", "plinth", "display space"],
    "weather": ["rain", "snow", "cold", "hot", "weather"],
    "traffic": ["traffic", "busy", "quiet", "slow"],
}


def analyse_text_sentiment(text):
    text_l = text.lower()
    pos = sum(w in text_l for w in POS_WORDS)
    neg = sum(w in text_l for w in NEG_WORDS)
    if pos == 0 and neg == 0:
        return "neutral"
    if pos >= neg:
        return "positive"
    return "negative"


def analyse_text_themes(text):
    text_l = text.lower()
    themes = []
    for theme, kws in THEME_KEYWORDS.items():
        if any(kw in text_l for kw in kws):
            themes.append(theme)
    return themes


def enrich_with_sentiment(df):
    df = df.copy()
    df["sentiment"] = df["free_text"].apply(analyse_text_sentiment)
    df["themes"] = df["free_text"].apply(analyse_text_themes)
    return df

# -----------------------------
# Data Quality
# -----------------------------
def qa_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Real-world anomalies: just some examples
    - Sample temps out of plausible range for chilled sampling
    - Very high reported damage % in produce audits

    """
    flags = []

    # 1) Tasting: implausibly high sample temperatures
    for _, r in df[df["report_type"] == "tasting"].iterrows():
        d = r["result_dict"]
        st_rec = d.get("Sample & Temp Records")
        if isinstance(st_rec, dict) and "Samples" in st_rec:
            temp_raw = st_rec["Samples"].get("Sample temp (if required)")
            t = to_float(temp_raw)
            # treat > 40°C as "real-world anomaly"
            if not math.isnan(t) and t > 40:
                flags.append(
                    {
                        "id": r["id"],
                        "issue": f"Sample temp unusually high for chilled product ({t}°C)",
                        "report_type": "tasting",
                    }
                )

    # 2) Produce audits: very high external damage %
    for _, r in df[df["report_type"] == "produce_audit"].iterrows():
        d = r["result_dict"]
        dmg = d.get(
            "SHEPARD - What percentage of all the fruit on display had external damage?"
        ) or d.get(
            "HASS - What percentage of all the fruit on display had external damage?"
        )
        if isinstance(dmg, str) and any(x in dmg for x in ["41-50%", ">50%", "51-60"]):
            flags.append(
                {
                    "id": r["id"],
                    "issue": f"Very high reported fruit damage ({dmg})",
                    "report_type": "produce_audit",
                }
            )

    return pd.DataFrame(flags)


def basic_null_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Column-level data quality: null counts, %, and unique values.
    """
    total = len(df)
    rows = []
    for col in df.columns:
        null_count = df[col].isna().sum()
        rows.append(
            {
                "column": col,
                "null_count": int(null_count),
                "null_pct": round((null_count / total) * 100, 2) if total > 0 else 0.0,
                "n_unique": df[col].nunique(dropna=True),
                "dtype": str(df[col].dtype),
            }
        )
    return pd.DataFrame(rows).sort_values("null_pct", ascending=False)

def data_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data-quality / formatting issues:
    - result JSON present but parsed to empty dict
    - TODO: Add others
    """
    flags = []

    # JSON parse issues: raw 'result' non-empty, but result_dict is empty
    for _, r in df.iterrows():
        raw_result = r.get("result")
        d = r.get("result_dict", {})
        if raw_result not in (None, "", np.nan) and isinstance(d, dict) and len(d) == 0:
            flags.append(
                {
                    "id": r["id"],
                    "issue": "result JSON could not be parsed (empty result_dict).",
                    "report_type": r.get("report_type"),
                }
            )

    return pd.DataFrame(flags)

# -----------------------------
# Outliers
# -----------------------------
def detect_outliers_numeric(df: pd.DataFrame, cols, z_thresh: float = 3.0, min_non_null: int = 10):
    """
    Generic numeric outlier detection using z-scores on existing columns.

    df    : flattened data (must contain 'id')
    cols  : list of existing numeric columns to check
    z_thresh : abs(z) above this is flagged
    min_non_null : skip a column if < this many non-null values
    """
    out_rows = []

    for col in cols:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.dropna()
        if len(non_null) < min_non_null:
            continue  # not enough data

        mean = non_null.mean()
        std = non_null.std(ddof=0)
        if std == 0:
            continue

        z_scores = (series - mean) / std
        mask = z_scores.abs() > z_thresh

        for idx in df.index[mask]:
            out_rows.append(
                {
                    "id": df.loc[idx, "id"],
                    "column": col,
                    "value": df.loc[idx, col],
                    "z_score": round(float(z_scores.loc[idx]), 2),
                    "mean": round(float(mean), 2),
                    "std": round(float(std), 2),
                }
            )

    return pd.DataFrame(out_rows)

# -----------------------------
# Generic report tab
# -----------------------------
def render_report_type_tab(
    df_type: pd.DataFrame,
    flatten_fn,
    numeric_cols,
    title: str,
    post_flatten=None,
    z_thresh: float = 3.0,
):
    if df_type.empty:
        st.write(f"No {title.lower()} reports in current filter.")
        return

    flat = pd.DataFrame([flatten_fn(r) for _, r in df_type.iterrows()])

    st.subheader(f"{title} – unified dataset")
    st.dataframe(flat)
    st.write(f"Total rows in this tab: {len(flat)}")

    st.subheader("Anomalies / outliers (statistical)")
    out = detect_outliers_numeric(flat, cols=numeric_cols, z_thresh=z_thresh)
    if out.empty:
        st.write("No numeric outliers found (given current thresholds).")
    else:
        st.dataframe(out)

    # --- tab-specific plots ---
    if post_flatten is not None:
        flat = post_flatten(flat)

def render_report_tab_intro():
    with st.expander("What this tab shows / where AI could go next"):
        st.markdown(
            """
            This tab focuses on one report type and turns its raw JSON into a
            **flat table** plus some very simple statistics.

            Right now, this is mostly:
            - deterministic parsing (JSON → columns),
            - basic KPIs (sums, rates, scores),
            - simple z-score outlier detection.

            There is no real AI/ML yet in these views.

            ---
            ### Where AI/ML would live for this report type

            Depending on the domain (tastings, audits, visits, shifts, quizzes), we could:

            **1. Predict key outcomes**
            - Train models to predict things like:
              - conversion rate / units sold,
              - execution quality / compliance score,
              - interactions per hour,
              - quiz performance / knowledge gaps.
            - Use these predictions to:
              - flag underperforming events/stores in advance,
              - simulate “what if” scenarios (e.g. more POS, different time of day).

            **2. Learn drivers instead of just showing raw metrics**
            - Replace simple correlations with feature importance from ML models.
            - Answer questions like:
              - “Which factors most strongly drive conversion for this client?”
              - “What differentiates top vs. bottom quartile events?”

            **3. Smarter anomaly detection**
            - Move beyond z-scores to learned anomaly detection that:
              - adapts by retailer, store, day/time, campaign,
              - distinguishes “good weird” (great performance) from “bad weird” (data errors / real issues).

            **4. Text understanding & summarisation**
            - Apply NLP/LLMs to the free-text fields linked to these reports to:
              - summarise why a score was high/low,
              - cluster similar issues or success stories,
              - generate short, actionable recommendations (“Do more of X, fix Y”).

            **5. Coaching & decision support**
            - Turn raw metrics into next-best actions, such as:
              - “For this type of store and traffic, target ≥ N interactions/hr.”
              - “Stores with low execution score often lack POS type A.”
              - “Re-train users who consistently underperform on quiz topic Z.”

            This tab is therefore a scaffold: it shows the structured data and
            basic KPIs that an AI/ML layer would build on, but the actual models,
            recommendations, and natural-language insights are intentionally missing
            to keep the demo simple and explainable.
            """
        )

# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="Field Data AI Demo", layout="wide")
    st.title("Field Data AI Demo")
    st.caption("Explore field reports by type and use cross-cutting tools for sentiment, drivers, and QA.")
    with st.expander("What this demo is:"):
        st.markdown(
            """
            This is a **lightweight playground** for the field data:
            - split reports into types,
            - flatten the JSON,
            - do some basic stats + QA,
            - add a tiny bit of “AI‑ish” logic (keywords, simple outliers).

            The idea is to see how far you get with the basics before
            bothering with real ML/AI.

            Sensible rollout: ship this simple version, see what people actually
            use, then only add heavier ML/AI where it clearly helps (predictions,
            semantic search, Q&A, etc.).
            """
        )
    st.divider()

    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Upload completed_reports.csv", type=["csv"])
    sample_n = st.sidebar.number_input(
        "Sample first N rows (0 = all)", min_value=0, value=1000, step=100
    )

    if uploaded is None:
        st.info("Upload a CSV to start (e.g. completed_reports.csv).")
        return

    df_raw = load_csv(uploaded_file=uploaded, nrows=sample_n or None)
    st.sidebar.write(f"Loaded {len(df_raw)} rows.")

    # Classify and enrich
    with st.spinner("Parsing JSON and classifying report types..."):
        df = classify_all_reports(df_raw)
        df = enrich_with_sentiment(df)

    st.sidebar.subheader("Filters")
    report_types = sorted(df["report_type"].unique())
    selected_types = st.sidebar.multiselect(
        "Report types", report_types, default=report_types
    )
    df = df[df["report_type"].isin(selected_types)]

    st.markdown("### Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Users", df["user_id"].nunique())
    with col3:
        st.metric("Report templates", df["report_id"].nunique())
    with col4:
        st.metric("Partners", df["partner_website_id"].nunique())

    st.write("Reports by type")
    st.bar_chart(df["report_type"].value_counts())

    st.divider()

    # ---------------- Tabs: cross-cutting tools / QA ----------------
    st.markdown("### Cross-cutting analysis & QA")
    analysis_tabs = st.tabs(
        [
            "Sentiment & Themes",
            "Drivers & Correlations",
            "Search & Q&A",
            "Internal QA",
        ]
    )

    # ---------------- Sentiment & Themes ----------------
    with analysis_tabs[0]:
        st.subheader("Sentiment by report type")
        with st.expander("What this tab shows / where AI could go next"):
            st.markdown(
            """
            This tab gives a first-pass text analysis over all free-text fields in the reports.

            **What it does**
            - Uses simple keyword-based rules to label each report as *positive*, *negative* or *neutral*.
            - Extracts recurring themes (e.g. price, taste, stock, space, weather, traffic) based on keyword matches.
            - Aggregates sentiment by report type so you can see where feedback is generally more positive or negative.
            - Counts how often each theme appears to highlight common pain points or topics.

            **What it does not do (yet)**
            - It does not understand context, sarcasm, or nuance.
            - It may miss important themes that don’t match the current keyword lists.
            - It’s not using embeddings or LLMs – this is not full NLP just heuristics.

            This view is meant to show how even basic rules can surface useful patterns,
            and to motivate a future upgrade to real NLP/AI (embeddings, semantic search,
            clustering, and LLM-based summarisation).
            """
            )
        sent_counts = (
            df.groupby(["report_type", "sentiment"])["id"]
            .count()
            .unstack(fill_value=0)
        )
        st.dataframe(sent_counts)

        st.subheader("Top themes")
        theme_counts = {}
        for _, r in df.iterrows():
            for t in r["themes"]:
                theme_counts[t] = theme_counts.get(t, 0) + 1
        if theme_counts:
            theme_df = (
                pd.DataFrame.from_dict(theme_counts, orient="index", columns=["count"])
                .sort_values("count", ascending=False)
            )
            st.bar_chart(theme_df)
        else:
            st.write("No themes detected with current heuristic.")

    # ---------------- Drivers & correlations ----------------
    with analysis_tabs[1]:
        st.subheader("Example driver analysis (simple correlations)")
        with st.expander("What this tab shows / where AI could go next"):
            st.markdown(
                """
                This tab is a placeholder for driver analysis. Right now it only:
                - flattens a few numeric KPIs from different report types, and
                - computes a plain Pearson correlation matrix between them.

                This is useful as a quick sanity check but it is not real AI/ML

                ---
                ### What’s missing (and where AI/ML would live)

                **1. Proper predictive models**
                - Train models to predict key KPIs, for example:
                - conversion rate (tastings),
                - execution_score (produce audits),
                - interactions_per_hour (shifts),
                - quiz scores (training).
                - Use these models to:
                - identify which features actually matter (feature importance),
                - flag underperformance relative to an expected baseline.

                **2. Richer driver analysis**
                - Go beyond simple correlations by:
                - including categorical features (store, retailer, day/time, POS types),
                - modelling non-linear relationships and interactions,
                - explaining drivers via SHAP values or similar techniques.
                - Answer questions like:
                - “What are the top 5 drivers of conversion for this client?”
                - “Which POS elements have the biggest impact on execution_score?”

                **3. Segmented and contextual insights**
                - Build separate models or segments by:
                - retailer, banner, region, store archetype,
                - campaign or product family.
                - Detect that “the drivers of performance in Retailer A may be very different from Retailer B”.

                **4. Uplift and “what if” scenarios**
                - Estimate uplift from interventions:
                - adding POS, changing sampling time, increasing staffing, etc.
                - Support “what if we changed X?” questions with model-based simulations.

                **5. Monitoring over time**
                - Track how drivers and coefficients change over time:
                - detect shifts in shopper behaviour,
                - adjust playbooks accordingly.

                In short, this tab currently shows only static, linear correlations
                A real AI/ML layer would build predictive, segmented, and explainable models
                on top of the same flattened data to give much stronger “why” and “what to do next” answers.
                """
            )
        frames = []
        if "tasting" in df["report_type"].unique():
            t_flat = pd.DataFrame(
                [flatten_tasting(r) for _, r in df[df["report_type"] == "tasting"].iterrows()]
            )
            frames.append(
                t_flat[["conversion_rate", "interactions", "samples", "units_sold"]]
            )
        if "produce_audit" in df["report_type"].unique():
            p_flat = pd.DataFrame(
                [flatten_produce(r) for _, r in df[df["report_type"] == "produce_audit"].iterrows()]
            )
            frames.append(p_flat[["execution_score", "shoppers_engaged"]])

        if not frames:
            st.write("No numeric KPIs available in current filter.")
        else:
            numeric = pd.concat(frames, axis=1)
            numeric = numeric.select_dtypes(include=[np.number])
            numeric = numeric.loc[:, numeric.columns.notnull()]
            corr = numeric.corr()
            st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

    # ---------------- Search & Q&A ----------------
    with analysis_tabs[2]:
        st.subheader("Simple search across comments/text")
        with st.expander("What this tab shows / where AI could go next"):
            st.markdown(
            """
            This tab provides a very basic search over free-text fields.

            Today, it only:
            - concatenates text into a `free_text` field, and
            - runs a simple substring match (`contains`) on that text.

            There is no semantic understanding or true “Q&A” here yet.

            ---
            ### What’s missing (and where AI/ML would live)

            **1. Semantic search instead of substring search**
            - Use embeddings (e.g. Sentence-BERT or similar) to represent each comment/report.
            - Store these in a vector ** (FAISS, Pinecone, etc.).
            - At query time, embed the user’s question and retrieve semantically ** reports:
              - “people complaining about price” should find text that mentions “too expensive”, “too pricey”, “costs too much”, etc.

            **2. Real Q&A with retrieval-augmented generation (RAG)**
            - Pipeline:
              1. User asks a natural language question (e.g. “Why is conversion low in cold weather?”).
              2. System uses embeddings to retrieve the most relevant reports
              3. An LLM reads those reports and generates a grounded answer:
                 - short explanation,
                 - key themes,
                 - supporting examples and metrics.
            - This turns search from “find rows” into “answer questions”.

            **3. Summarisation and roll-ups**
            - Let users ask for summaries instead of raw hits:
              - “Summarise key issues for Retailer X in Q1.”
              - “What are shoppers saying about flavour in this campaign?”
            - Use LLMs to produce:
              - top themes,
              - sentiment breakdown,
              - example quotes, all grounded in retrieved data.

            **4. Smarter filtering and navigation**
            - Combine semantic search with filters:
              - by report type, retailer, store, time period, score band, etc.
            - Help users quickly move from a question to:
              - relevant segments,
              - drill-down reports,
              - suggested follow-up questions.

            Right now, this tab is intentionally simple and deterministic so it’s easy to reason about.
            A real AI-powered search & Q&A experience would add embeddings, a vector index,
            and an LLM layer on top to deliver semantic, summarised, and actionable answers.
            """
            )
        q = st.text_input("Search term (e.g. 'no room', 'too sweet', 'price')").strip()
        if q:
            mask = df["free_text"].str.contains(q, case=False, na=False)
            results = df.loc[mask, ["id", "report_id", "report_type", "created_at", "free_text"]]
            st.write(f"Found {len(results)} matching rows.")
            st.dataframe(results)
        else:
            st.write("Enter a term to search.")

    # ---------------- Internal QA ----------------
    with analysis_tabs[3]:
        st.subheader("Column completeness (top-level CSV)")
        st.markdown(
            "*This tab is intentionally limited to structural/data-quality issues "
            "(nulls, parsing, bad formats). Real-world anomalies are surfaced inside "
            "each domain tab.*"
        )

        nulls = basic_null_report(df_raw)
        st.dataframe(nulls)

        st.subheader("Data quality")

        dq = data_quality_flags(df)
        if dq.empty:
            st.write("No data-quality flags raised with the current simple rules.")
        else:
            st.dataframe(dq)


    st.divider()

    # ---------------- Tabs: report-type views ----------------
    st.markdown("### Report-type views")
    report_tabs = st.tabs(
        [
            "Tastings & Activations",
            "Produce Audits",
            "Store Visits",
            "Shifts & Events",
            "Quizzes / Training",
        ]
    )

    # ---------------- Tastings ----------------
    with report_tabs[0]:
        render_report_tab_intro()
        tdf_rows = df[df["report_type"] == "tasting"]
        render_report_type_tab(
            df_type=tdf_rows,
            flatten_fn=flatten_tasting,
            numeric_cols=["conversion_rate", "samples", "units_sold", "interactions"],
            title="Tastings & activations",
            post_flatten=None,      # nothing special needed here
            z_thresh=3.0,
        )

    # ---------------- Produce ----------------
    with report_tabs[1]:
        render_report_tab_intro()
        pdf_rows = df[df["report_type"] == "produce_audit"]

        def post_flatten_produce(p_flat: pd.DataFrame) -> pd.DataFrame:
            p_flat = p_flat.copy()
            p_flat["hass_on_display"] = p_flat["hass_on_display"].apply(to_bool)
            p_flat["shepard_on_display"] = p_flat["shepard_on_display"].apply(to_bool)
            return p_flat

        def _post_produce(flat):
            flat = post_flatten_produce(flat)
            st.subheader("Execution score distribution")
            st.bar_chart(flat["execution_score"].value_counts().sort_index())
            return flat

        render_report_type_tab(
            df_type=pdf_rows,
            flatten_fn=flatten_produce,
            numeric_cols=["execution_score", "shoppers_engaged"],
            title="Produce audits",
            post_flatten=_post_produce,
            z_thresh=2.5,
        )

    # ---------------- Store visits ----------------
    with report_tabs[2]:
        render_report_tab_intro()
        sdf_rows = df[df["report_type"] == "store_visit"]
        render_report_type_tab(
            df_type=sdf_rows,
            flatten_fn=flatten_store_visit,
            numeric_cols=[],
            title="Store visits / Philips",
            post_flatten=None,
            z_thresh=3.0,
        )

    # ---------------- Shifts ----------------
    with report_tabs[3]:
        render_report_tab_intro()
        sh_rows = df[df["report_type"] == "shift"]

        def _post_shifts(flat):
            st.subheader("Interactions per hour")
            if "interactions_per_hour" in flat.columns:
                st.bar_chart(flat.set_index("id")["interactions_per_hour"].dropna())
            return flat

        render_report_type_tab(
            df_type=sh_rows,
            flatten_fn=flatten_shift,
            numeric_cols=["interactions_per_hour", "people_spoken"],
            title="Shifts & events",
            post_flatten=_post_shifts,
            z_thresh=3.0,
        )

    # ---------------- Quizzes ----------------
    with report_tabs[4]:
        render_report_tab_intro()
        q_rows = df[df["report_type"] == "quiz"]

        def _post_quiz(flat):
            st.subheader("Score distribution")
            st.bar_chart(flat["score"].value_counts().sort_index())
            return flat

        render_report_type_tab(
            df_type=q_rows,
            flatten_fn=flatten_quiz,
            numeric_cols=["score"],
            title="Quizzes / training",
            post_flatten=_post_quiz,
            z_thresh=3.0,
        )


if __name__ == "__main__":
    main()
