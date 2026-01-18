import os
import re
import random
import math
from collections import defaultdict

import pandas as pd
import streamlit as st
import plotly.express as px


# =========================
# Page config + subtle CSS
# =========================
st.set_page_config(
    page_title="Chat Survey Analytics",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1250px; }
      div[data-testid="stMetric"] { background: rgba(0,0,0,0.03); padding: 14px 16px; border-radius: 14px; }
      .small-note { color: rgba(0,0,0,0.55); font-size: 0.9rem; }
      .panel-title { font-size: 1.05rem; font-weight: 650; margin-bottom: 0.3rem; }
      hr { margin: 1.25rem 0; }
      code { font-size: 0.9em; }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Fake data generator (quant)
# =========================
@st.cache_data(show_spinner=False)
def generate_fake_scores(n=200) -> pd.DataFrame:
    rows = []
    for i in range(1, n + 1):
        cid = f"conv_{i:04d}"
        resistance = round(random.uniform(0.05, 0.85), 2)
        quality = round(random.uniform(0.2, 0.95), 2)
        trust = round(max(0.05, min(0.95, random.uniform(0.15, 0.9) * (1 - resistance + 0.2))), 2)
        cog = round(random.uniform(0.2, 0.75) * (0.7 + resistance), 2)
        momentum = round(random.uniform(-0.7, 0.7) * (0.8 - resistance / 2), 2)

        insight_yield = round(max(0.01, (quality * (1 - resistance)) * random.uniform(0.7, 1.2)), 2)
        cnps = int(round((quality - resistance) * 120 + random.uniform(-15, 15)))
        cnps = max(-100, min(100, cnps))

        completion = int(resistance < 0.55 or quality > 0.65)
        total_turns = random.randint(4, 12)
        duration_sec = round(total_turns * random.uniform(2.5, 6.0), 1)

        rows.append({
            "conversation_id": cid,
            "cNPS": cnps,
            "resistance_score": resistance,
            "quality_score": quality,
            "insight_yield": insight_yield,
            "trust_score": trust,
            "cognitive_load": cog,
            "engagement_momentum": momentum,
            "completion": completion,
            "total_turns": total_turns,
            "duration_sec": duration_sec
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_scores(path: str) -> pd.DataFrame:
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        df = generate_fake_scores(250)
        df.to_csv(path, index=False)
        return df

    df = pd.read_csv(path)
    if df.shape[1] == 0 or df.shape[0] == 0:
        df = generate_fake_scores(250)
        df.to_csv(path, index=False)
        return df

    # defensive ranges
    if "cNPS" in df.columns:
        df["cNPS"] = pd.to_numeric(df["cNPS"], errors="coerce").clip(-100, 100)
    for c in ["resistance_score", "quality_score", "trust_score", "cognitive_load"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").clip(0, 1)
    if "engagement_momentum" in df.columns:
        df["engagement_momentum"] = pd.to_numeric(df["engagement_momentum"], errors="coerce").clip(-1, 1)
    if "completion" in df.columns:
        df["completion"] = pd.to_numeric(df["completion"], errors="coerce").fillna(0).astype(int)

    return df


# =========================
# Qualitative loader
# =========================
@st.cache_data(show_spinner=False)
def load_conversations(path: str) -> pd.DataFrame:
    if (not os.path.exists(path)) or (os.path.getsize(path) == 0):
        return pd.DataFrame()

    df = pd.read_csv(path)
    needed = {"conversation_id", "turn_id", "speaker", "text"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    # ensure types
    df["turn_id"] = pd.to_numeric(df["turn_id"], errors="coerce").fillna(0).astype(int)
    df["speaker"] = df["speaker"].astype(str).str.lower().str.strip()
    df["text"] = df["text"].astype(str)
    if "sentiment" in df.columns:
        df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    else:
        df["sentiment"] = pd.NA
    if "resistance_flag" in df.columns:
        df["resistance_flag"] = pd.to_numeric(df["resistance_flag"], errors="coerce").fillna(0).astype(int)
    else:
        df["resistance_flag"] = 0
    if "trust_signal" in df.columns:
        df["trust_signal"] = pd.to_numeric(df["trust_signal"], errors="coerce").fillna(0).astype(int)
    else:
        df["trust_signal"] = 0

    df = df.sort_values(["conversation_id", "turn_id"]).reset_index(drop=True)
    return df


# =========================
# Heuristic semantic tagging
# =========================
TAG_PATTERNS = {
    "pain_point": r"\b(frustrat|annoy|disappoint|bad|broken|doesn[' ]?t work|problem|issue|hate|terrible)\b",
    "delight": r"\b(love|great|awesome|amazing|helpful|easy|perfect|worked well)\b",
    "confusion": r"\b(confus|not sure|don[' ]?t understand|what do you mean|huh)\b",
    "feature_request": r"\b(feature request|wish it had|could you add|would be nice if|please add)\b",
    "workaround": r"\b(workaround|i just|so i ended up|i decided to|i usually)\b",
    "resistance": r"\b(why do you need|don[' ]?t want to|rather not|stop asking|skip|no thanks)\b",
    "trust_signal": r"\b(i feel|honestly|to be real|personally|in my experience|i[' ]?ve been)\b",
    "dropoff_risk": r"\b(whatever|idk|fine\.?$|k\.?$|ok\.?$|sure\.?$)\b",
}

def semantic_tags(text: str) -> list[str]:
    t = (text or "").lower()
    tags = []
    for tag, pat in TAG_PATTERNS.items():
        if re.search(pat, t):
            tags.append(tag)
    return tags

def info_density(text: str) -> float:
    """Crude but useful: length + digits + nouns-ish tokens proxy."""
    if not text:
        return 0.0
    tokens = re.findall(r"[A-Za-z0-9']+", text)
    if not tokens:
        return 0.0
    digits = sum(any(ch.isdigit() for ch in tok) for tok in tokens)
    length = len(tokens)
    # normalize to ~0..1 range for typical chat lengths
    score = (min(length, 40) / 40.0) * 0.7 + (min(digits, 3) / 3.0) * 0.3
    return float(max(0.0, min(1.0, score)))

def novelty_proxy(text: str, seen_terms: set[str]) -> float:
    """Fraction of new meaningful terms not seen before in this conversation."""
    tokens = [t.lower() for t in re.findall(r"[A-Za-z']+", text or "")]
    tokens = [t for t in tokens if len(t) > 3]
    if not tokens:
        return 0.0
    new = [t for t in tokens if t not in seen_terms]
    for t in tokens:
        seen_terms.add(t)
    return float(len(new) / len(tokens))


# =========================
# Turn-level evaluation + insight events
# =========================
@st.cache_data(show_spinner=False)
def enrich_turns(conv_df: pd.DataFrame) -> pd.DataFrame:
    if conv_df.empty:
        return conv_df

    out = conv_df.copy()
    out["tags"] = out["text"].apply(semantic_tags)
    out["tag_str"] = out["tags"].apply(lambda xs: ", ".join(xs) if xs else "")

    # compute sentiment if missing using a tiny heuristic lexicon
    # (kept simple to avoid extra deps)
    pos_words = set(["love", "great", "awesome", "amazing", "helpful", "easy", "perfect"])
    neg_words = set(["hate", "bad", "broken", "frustrating", "annoying", "disappointing", "terrible", "confusing"])

    def fallback_sentiment(text: str) -> float:
        toks = [t.lower() for t in re.findall(r"[A-Za-z']+", text or "")]
        if not toks:
            return 0.0
        pos = sum(t in pos_words for t in toks)
        neg = sum(t in neg_words for t in toks)
        raw = (pos - neg) / max(1, (pos + neg))
        return float(max(-1.0, min(1.0, raw)))

    if out["sentiment"].isna().all():
        out["sentiment"] = out["text"].apply(fallback_sentiment)
    else:
        out["sentiment"] = out["sentiment"].fillna(out["text"].apply(fallback_sentiment))

    out["info_density"] = out["text"].apply(info_density)

    # novelty per conversation
    nov = []
    for cid, g in out.groupby("conversation_id", sort=False):
        seen = set()
        for txt in g["text"].tolist():
            nov.append(novelty_proxy(txt, seen))
    out["novelty"] = nov

    # sentiment delta per conversation
    out["sentiment_delta"] = out.groupby("conversation_id")["sentiment"].diff().fillna(0.0)

    # insight events on user turns
    insight_tags = {"pain_point", "feature_request", "workaround", "expectation_gap"}
    # expectation_gap heuristic
    def has_expectation_gap(text: str) -> int:
        t = (text or "").lower()
        return int(("expected" in t) or ("but" in t and ("should" in t or "supposed" in t)))
    out["expectation_gap"] = out["text"].apply(has_expectation_gap)
    out.loc[out["expectation_gap"] == 1, "tags"] = out.loc[out["expectation_gap"] == 1, "tags"].apply(lambda xs: list(set(xs + ["expectation_gap"])))

    def is_insight_event(row) -> int:
        if row["speaker"] != "user":
            return 0
        tset = set(row["tags"] or [])
        return int(len(tset.intersection({"pain_point", "feature_request", "workaround", "expectation_gap"})) > 0)

    out["insight_event"] = out.apply(is_insight_event, axis=1)

    # dropoff risk proxy: short, low novelty, neutral/negative
    def dropoff_risk(row) -> int:
        if row["speaker"] != "user":
            return 0
        short = len((row["text"] or "").strip()) <= 4
        low_nov = (row["novelty"] or 0) < 0.15
        negish = (row["sentiment"] or 0) < 0
        return int(short or (low_nov and negish))

    out["dropoff_risk"] = out.apply(dropoff_risk, axis=1)
    return out


# =========================
# Conversation archetypes + failure modes
# =========================
def slope(x: list[float]) -> float:
    # simple slope via least squares on index
    n = len(x)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(x) / n
    num = sum((xs[i]-mx)*(x[i]-my) for i in range(n))
    den = sum((xs[i]-mx)**2 for i in range(n))
    return float(num / den) if den else 0.0

@st.cache_data(show_spinner=False)
def build_conversation_analytics(turns: pd.DataFrame) -> pd.DataFrame:
    if turns.empty:
        return pd.DataFrame()

    rows = []
    for cid, g in turns.groupby("conversation_id", sort=False):
        ug = g[g["speaker"] == "user"]
        if ug.empty:
            continue

        sent = ug["sentiment"].astype(float).tolist()
        sent_slope = slope(sent)

        res = ug["resistance_flag"].astype(int).tolist()
        early_res = int(sum(res[:3]) >= 1)

        insight_events = int(ug["insight_event"].sum())
        user_turns = int(len(ug))
        insight_yield = insight_events / max(1, user_turns)

        trust = int(ug["trust_signal"].sum())
        avg_len = float(ug["text"].astype(str).apply(lambda s: len(s.strip())).mean())

        # archetypes (simple and explainable)
        if user_turns <= 2 and insight_events == 0:
            archetype = "Dead-End"
        elif trust >= 2 and insight_yield >= 0.35 and user_turns >= 5:
            archetype = "High-Trust Deep Dive"
        elif early_res and sent_slope < 0:
            archetype = "Early Friction"
        elif sent_slope < -0.05 and user_turns >= 4:
            archetype = "Slow Bleed"
        elif sent_slope > 0.05 and user_turns >= 4:
            archetype = "Warm-Up"
        else:
            archetype = "Mixed"

        # failure modes
        # Rephrasing proxy: user asks "what" or repeats question marks often
        rephrase = int(ug["text"].str.count(r"\?").sum() >= 3)
        big_sent_drop = int((ug["sentiment_delta"] < -0.4).any())
        short_answers = int((ug["text"].astype(str).apply(lambda s: len(s.split())) <= 2).sum() >= 2)

        rows.append({
            "conversation_id": cid,
            "insight_events": insight_events,
            "insight_yield_calc": round(insight_yield, 3),
            "sentiment_slope_user": round(sent_slope, 3),
            "early_resistance_spike": early_res,
            "trust_signals_user": trust,
            "avg_user_msg_len": round(avg_len, 1),
            "archetype": archetype,
            "fail_rephrase_proxy": rephrase,
            "fail_big_sentiment_drop": big_sent_drop,
            "fail_short_answers": short_answers,
        })

    return pd.DataFrame(rows)


# =========================
# Question effectiveness (bot question -> next user reply outcomes)
# =========================
@st.cache_data(show_spinner=False)
def question_effectiveness(turns: pd.DataFrame) -> pd.DataFrame:
    if turns.empty:
        return pd.DataFrame()

    rows = []
    for cid, g in turns.groupby("conversation_id", sort=False):
        g = g.sort_values("turn_id")
        # pair each bot turn with the next user turn
        for i in range(len(g) - 1):
            if g.iloc[i]["speaker"] != "bot":
                continue
            # find next user
            j = i + 1
            while j < len(g) and g.iloc[j]["speaker"] != "user":
                j += 1
            if j >= len(g):
                continue

            bot_q = str(g.iloc[i]["text"]).strip()
            user_a = g.iloc[j]

            # heuristics: answer "quality gained"
            q_gained = 0.45 * float(user_a["info_density"]) + 0.35 * float(user_a["novelty"]) + 0.20 * (1.0 if user_a["insight_event"] else 0.0)
            r_induced = float(user_a["resistance_flag"])
            insight = int(user_a["insight_event"])
            trust = int(user_a["trust_signal"])

            rows.append({
                "conversation_id": cid,
                "question": bot_q,
                "answer_quality_gained": q_gained,
                "resistance_induced": r_induced,
                "insight_unlocked": insight,
                "trust_unlocked": trust,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # group by question text (in real systems you'd normalize / cluster questions)
    agg = df.groupby("question", as_index=False).agg(
        asks=("question", "count"),
        avg_quality_gained=("answer_quality_gained", "mean"),
        avg_resistance=("resistance_induced", "mean"),
        insight_rate=("insight_unlocked", "mean"),
        trust_rate=("trust_unlocked", "mean"),
    )

    # effectiveness score
    # Quality + insight + trust minus resistance
    agg["effectiveness"] = (
        0.45 * agg["avg_quality_gained"] +
        0.35 * agg["insight_rate"] +
        0.20 * agg["trust_rate"] -
        0.50 * agg["avg_resistance"]
    )

    # clean formatting
    for c in ["avg_quality_gained", "avg_resistance", "insight_rate", "trust_rate", "effectiveness"]:
        agg[c] = agg[c].round(3)

    agg = agg.sort_values("effectiveness", ascending=False).reset_index(drop=True)
    return agg


# =========================
# Sidebar controls
# =========================
st.sidebar.title("Controls")
scores_path = st.sidebar.text_input("Scores CSV path", value="conversation_scores_fake.csv")
convos_path = st.sidebar.text_input("Conversations CSV path", value="conversation_fake.csv")

with st.sidebar:
    with st.expander("Debug", expanded=False):
        st.write("Working directory:", os.getcwd())
        try:
            st.write("Files here:", os.listdir(os.getcwd()))
        except Exception:
            st.write("Could not list files in cwd.")

df_scores = load_scores(scores_path)
df_convos = load_conversations(convos_path)
df_turns = enrich_turns(df_convos) if not df_convos.empty else pd.DataFrame()
df_conv_analytics = build_conversation_analytics(df_turns) if not df_turns.empty else pd.DataFrame()
df_qeff = question_effectiveness(df_turns) if not df_turns.empty else pd.DataFrame()

# merge computed insight_yield onto score table (if missing or to compare)
if not df_conv_analytics.empty:
    df_scores = df_scores.merge(
        df_conv_analytics[["conversation_id", "insight_yield_calc", "archetype",
                           "early_resistance_spike", "fail_big_sentiment_drop", "fail_short_answers"]],
        on="conversation_id",
        how="left"
    )

# =========================
# Header
# =========================
st.title("💬 Conversational Survey Dashboard")
st.caption("Quantitative outcomes plus conversation diagnostics derived from turn-level signals.")


# =========================
# Global filters (quant)
# =========================
with st.sidebar:
    st.markdown("---")
    st.subheader("Filters")

    # Completion filter
    if "completion" in df_scores.columns:
        completion_filter = st.multiselect(
            "Completion",
            options=["Completed", "Not completed"],
            default=["Completed", "Not completed"],
        )
        mask_completion = pd.Series(True, index=df_scores.index)
        if completion_filter != ["Completed", "Not completed"]:
            allowed = []
            if "Completed" in completion_filter:
                allowed.append(1)
            if "Not completed" in completion_filter:
                allowed.append(0)
            mask_completion = df_scores["completion"].isin(allowed)
    else:
        mask_completion = pd.Series(True, index=df_scores.index)

    # cNPS range
    cnps_min, cnps_max = st.slider("cNPS range", -100, 100, (-100, 100))
    mask_cnps = df_scores["cNPS"].between(cnps_min, cnps_max) if "cNPS" in df_scores.columns else pd.Series(True, index=df_scores.index)

    # Resistance range
    r_min, r_max = st.slider("Resistance range", 0.0, 1.0, (0.0, 1.0))
    mask_res = df_scores["resistance_score"].between(r_min, r_max) if "resistance_score" in df_scores.columns else pd.Series(True, index=df_scores.index)

filtered_scores = df_scores[mask_completion & mask_cnps & mask_res].copy()


# =========================
# Tabs (clean UI)
# =========================
tab_overview, tab_diagnostics, tab_questions, tab_conversations = st.tabs(
    ["Overview", "Diagnostics", "Question Effectiveness", "Conversation Explorer"]
)


# =========================
# Helpers
# =========================
def safe_mean(df, col):
    if col not in df.columns or len(df) == 0:
        return float("nan")
    return float(pd.to_numeric(df[col], errors="coerce").mean())

def pct(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x*100:.1f}%"

def segment_cnps(x):
    if pd.isna(x):
        return "Unknown"
    if x >= 50:
        return "Promoter"
    if x <= 0:
        return "Detractor"
    return "Passive"


# =========================
# TAB 1: OVERVIEW
# =========================
with tab_overview:
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric("cNPS", f"{safe_mean(filtered_scores,'cNPS'):.1f}")
    with k2:
        st.metric("Resistance", pct(safe_mean(filtered_scores,'resistance_score')))
    with k3:
        st.metric("Quality", pct(safe_mean(filtered_scores,'quality_score')))
    with k4:
        # show computed insight yield if available, else quantitative
        if "insight_yield_calc" in filtered_scores.columns and filtered_scores["insight_yield_calc"].notna().any():
            st.metric("Insight Yield", f"{safe_mean(filtered_scores,'insight_yield_calc'):.2f}")
        else:
            st.metric("Insight Yield", f"{safe_mean(filtered_scores,'insight_yield'):.2f}")
    with k5:
        if "completion" in filtered_scores.columns:
            st.metric("Completion", pct(float(filtered_scores["completion"].mean()) if len(filtered_scores) else float("nan")))
        else:
            st.metric("Completion", "—")

    st.markdown("<p class='small-note'>Overview is for outcomes. Diagnostics is for why. Question Effectiveness is for what to change.</p>", unsafe_allow_html=True)
    st.markdown("---")

    left, right = st.columns([1.35, 0.85], gap="large")

    with left:
        st.markdown("<div class='panel-title'>Score Distributions</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col, title, slot in [
            ("cNPS", "cNPS", c1),
            ("resistance_score", "Resistance", c2),
            ("quality_score", "Quality", c3),
        ]:
            with slot:
                if col in filtered_scores.columns:
                    fig = px.histogram(filtered_scores, x=col, nbins=22, title=title)
                    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=290, title_font_size=14)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Missing column: {col}")

    with right:
        st.markdown("<div class='panel-title'>Conversation Segments</div>", unsafe_allow_html=True)
        seg_df = filtered_scores.copy()
        if "cNPS" in seg_df.columns:
            seg_df["segment"] = seg_df["cNPS"].apply(segment_cnps)
            counts = seg_df["segment"].value_counts().reset_index()
            counts.columns = ["segment", "count"]
            fig = px.bar(counts, x="segment", y="count")
            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=290)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cNPS column found.")

    st.markdown("---")

    # Executive Summary (auto-generated narrative)
    st.markdown("<div class='panel-title'>Executive Summary</div>", unsafe_allow_html=True)

    summary_bits = []
    if len(filtered_scores) > 0:
        # early resistance impact (if computed)
        if "early_resistance_spike" in filtered_scores.columns and "insight_yield_calc" in filtered_scores.columns:
            a = filtered_scores[filtered_scores["early_resistance_spike"] == 1]["insight_yield_calc"].dropna()
            b = filtered_scores[filtered_scores["early_resistance_spike"] == 0]["insight_yield_calc"].dropna()
            if len(a) >= 5 and len(b) >= 5:
                delta = (b.mean() - a.mean())
                summary_bits.append(f"Early resistance spikes are associated with lower insight yield (gap ≈ {delta:.2f}).")

        # correlations
        num = filtered_scores.select_dtypes(include="number")
        if {"resistance_score", "quality_score", "cNPS"}.issubset(num.columns):
            corr_r_cnps = num["resistance_score"].corr(num["cNPS"])
            corr_q_cnps = num["quality_score"].corr(num["cNPS"])
            if not pd.isna(corr_r_cnps):
                summary_bits.append(f"Resistance correlates with cNPS at r = {corr_r_cnps:.2f}.")
            if not pd.isna(corr_q_cnps):
                summary_bits.append(f"Quality correlates with cNPS at r = {corr_q_cnps:.2f}.")

        # failure mode rates
        if "fail_big_sentiment_drop" in filtered_scores.columns:
            rate = filtered_scores["fail_big_sentiment_drop"].fillna(0).mean()
            summary_bits.append(f"Big sentiment drops appear in ~{rate*100:.1f}% of conversations (turn-level signal).")

    if not summary_bits:
        summary_bits = [
            "Load conversations to unlock insight events, archetypes, and question effectiveness.",
            "Right now you are seeing quantitative outcomes only."
        ]

    st.write("• " + "\n• ".join(summary_bits))


# =========================
# TAB 2: DIAGNOSTICS
# =========================
with tab_diagnostics:
    st.markdown("<div class='panel-title'>Diagnostics: What drives good conversations?</div>", unsafe_allow_html=True)

    d1, d2 = st.columns([1.2, 0.8], gap="large")

    with d1:
        if {"resistance_score", "quality_score"}.issubset(filtered_scores.columns):
            color_col = "completion" if "completion" in filtered_scores.columns else "cNPS"
            fig = px.scatter(
                filtered_scores,
                x="resistance_score",
                y="quality_score",
                color=color_col if color_col in filtered_scores.columns else None,
                hover_data=[c for c in ["conversation_id", "cNPS", "insight_yield_calc", "total_turns"] if c in filtered_scores.columns],
                title="Resistance vs Quality"
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need resistance_score and quality_score to show this plot.")

    with d2:
        numeric = filtered_scores.select_dtypes(include="number")
        if numeric.shape[1] >= 2:
            corr = numeric.corr(numeric_only=True).round(2)
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420, title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

    st.markdown("---")

    # Archetypes (if available)
    if "archetype" in filtered_scores.columns:
        a = filtered_scores["archetype"].value_counts().reset_index()
        a.columns = ["archetype", "count"]
        fig = px.bar(a, x="archetype", y="count", title="Conversation Archetypes")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=330, title_font_size=14)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Archetypes are derived from user-turn sentiment slope, early resistance, and insight density.")

    # Failure modes table
    fail_cols = [c for c in ["fail_big_sentiment_drop", "fail_short_answers", "early_resistance_spike"] if c in filtered_scores.columns]
    if fail_cols:
        rates = {c: float(filtered_scores[c].fillna(0).mean()) for c in fail_cols}
        fm = pd.DataFrame({"failure_mode": list(rates.keys()), "rate": [round(v, 3) for v in rates.values()]})
        fm["rate_pct"] = fm["rate"].apply(lambda x: f"{x*100:.1f}%")
        st.markdown("<div class='panel-title'>Failure Mode Rates</div>", unsafe_allow_html=True)
        st.dataframe(fm[["failure_mode", "rate_pct"]], use_container_width=True, hide_index=True)


# =========================
# TAB 3: QUESTION EFFECTIVENESS
# =========================
with tab_questions:
    st.markdown("<div class='panel-title'>Which bot questions help vs hurt?</div>", unsafe_allow_html=True)

    if df_qeff.empty:
        st.info("Load a valid conversations CSV to enable question effectiveness. This requires bot turn text + next user reply.")
    else:
        st.dataframe(df_qeff, use_container_width=True, hide_index=True)
        st.caption("Effectiveness = (quality gained + insight + trust) minus resistance. This is heuristic and designed to be explainable.")

        # Small visual: top/bottom questions
        topn = min(8, len(df_qeff))
        if topn >= 2:
            fig = px.bar(df_qeff.head(topn), x="effectiveness", y="question", orientation="h", title="Top Questions")
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=340, title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)


# =========================
# TAB 4: CONVERSATION EXPLORER
# =========================
with tab_conversations:
    st.markdown("<div class='panel-title'>Conversation Explorer (turn-level signals + semantic tags)</div>", unsafe_allow_html=True)

    if df_turns.empty:
        st.info("No conversations file loaded. Add conversations_fake.csv to unlock turn-level evaluation and semantic tags.")
    else:
        # pick conversation
        options = sorted(df_turns["conversation_id"].unique().tolist())
        cid = st.selectbox("Select a conversation", options=options, index=0)

        t = df_turns[df_turns["conversation_id"] == cid].sort_values("turn_id").copy()

        # top-line mini summary
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("User turns", int((t["speaker"] == "user").sum()))
        with s2:
            st.metric("Insight events", int(t["insight_event"].sum()))
        with s3:
            st.metric("Avg info density (user)", f"{t[t['speaker']=='user']['info_density'].mean():.2f}")
        with s4:
            st.metric("Dropoff risks (user)", int(t[t["speaker"] == "user"]["dropoff_risk"].sum()))

        st.markdown("---")

        # Trajectory chart (sentiment over user turns)
        ug = t[t["speaker"] == "user"].reset_index(drop=True)
        if len(ug) >= 2:
            fig = px.line(ug, x=ug.index, y="sentiment", title="User Sentiment Trajectory", markers=True)
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=300, title_font_size=14)
            st.plotly_chart(fig, use_container_width=True)

        # Show conversation with tags
        show_cols = ["turn_id", "speaker", "text", "sentiment", "sentiment_delta", "resistance_flag", "trust_signal", "info_density", "novelty", "insight_event", "tag_str"]
        show_cols = [c for c in show_cols if c in t.columns]
        st.dataframe(t[show_cols], use_container_width=True, hide_index=True)

        st.caption("Semantic tags are rule-based right now. You can later swap the tagger for embeddings/LLM without changing the dashboard structure.")
