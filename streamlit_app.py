"""
Typology — Editorial MBTI Personality Inference Web App (Streamlit Edition).

Designed for Streamlit Community Cloud and local Streamlit runs with a
quiet, paper-and-ink editorial aesthetic and zero emojis.
"""

from __future__ import annotations

import html
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from model_engine import MBTIEngine, MBTI_PROFILES

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "mbti_predictor.joblib"

st.set_page_config(
    page_title="Typology — MBTI Inference from Written Language",
    layout="centered",
    initial_sidebar_state="collapsed",
)


PROMPTS: List[Dict[str, str]] = [
    {
        "num": "01",
        "question": "Describe your ideal weekend from morning to night.",
        "placeholder": "How do you choose to spend unstructured hours when nobody expects anything from you?",
    },
    {
        "num": "02",
        "question": "How do you usually approach a large project or tight deadline?",
        "placeholder": "Do you map milestones upfront, adapt as you go, or wait for momentum?",
    },
    {
        "num": "03",
        "question": "When making a difficult decision, what weighs most heavily for you?",
        "placeholder": "Think of a recent hard choice and how you arrived at your conclusion.",
    },
    {
        "num": "04",
        "question": "How do you recharge after a long, demanding week?",
        "placeholder": "What reliably restores your energy, and what drains it fastest?",
    },
    {
        "num": "05",
        "question": "What subjects, hobbies, or conversations absorb your attention completely?",
        "placeholder": "Write about the ideas, crafts, or problems you naturally lose track of time exploring.",
    },
    {
        "num": "06",
        "question": "How do you handle disagreement or tension with someone close to you?",
        "placeholder": "What is your instinct when a conversation gets uncomfortable?",
    },
]

SAMPLES: Dict[str, List[str]] = {
    "architect": [
        "I usually spend my weekends quietly working through a long-term side project or reading books on systems design, history, and philosophy. Crowded, noisy rooms drain my energy quickly, and I would rather have one focused conversation than circulate through small talk.",
        "Before starting any major project, I map out the full architecture, list each milestone, and set firm internal deadlines so nothing depends on last-minute improvisation. Having a clear logical plan frees my mind to focus on the hard technical questions.",
        "When a friend asks for advice on a difficult decision, my first instinct is to lay out the objective trade-offs, second-order consequences, and measurable criteria rather than reacting purely to how everyone feels in the moment.",
        "What excites me most is uncovering the underlying framework behind complex phenomena. Memorizing routine step-by-step procedures without understanding the theory behind them quickly loses my interest.",
    ],
    "campaigner": [
        "Last weekend my friends and I spontaneously drove out toward the coast with zero itinerary, stopped at three tiny roadside cafes, and ended up talking for hours with a local ceramicist we met at a market!",
        "I get genuinely energized bouncing fresh ideas around with people. When someone I care about is going through a rough week, I want to sit down, listen with my whole heart, and help them feel truly understood and hopeful again.",
        "Honestly, rigid schedules and repetitive routines make me feel boxed in. I prefer leaving space in my days for curiosity, unexpected invitations, and creative detours that turn an ordinary afternoon into a story worth remembering.",
        "I love imagining what could be possible in the future and connecting people's personal stories, values, and passions. Even in disagreements, keeping warmth, sincerity, and mutual respect intact matters more to me than scoring a logical point.",
    ],
    "logistician": [
        "My ideal weekend starts early with coffee, a walk around the neighborhood, and getting the house and errands squared away before noon. Once the practical tasks are finished, I enjoy cooking a familiar recipe, working on a tangible repair project, or catching up with family.",
        "On any work assignment I review the exact specifications first, verify the facts, and build a realistic step-by-step checklist. I do not like vague speculation when concrete numbers and proven methods are right in front of us.",
        "When making major life choices I look closely at reliability, budget, past experience, and long-term obligations. Keeping commitments and following through consistently matters far more to me than chasing untested trends.",
        "In a disagreement I prefer calm, matter-of-fact honesty. Laying out the specific facts clearly and agreeing on practical next steps resolves tension much faster than guessing at unspoken motives.",
    ],
}

GROUP_ORDER = ["Analysts", "Diplomats", "Sentinels", "Explorers"]


def inject_editorial_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');

        :root {
          --bg: #F6F4EF;
          --surface: #FFFFFF;
          --surface-muted: #F0EDE6;
          --border: #DFDBD2;
          --border-strong: #C6C0B4;
          --ink: #171715;
          --ink-secondary: #4E4B45;
          --ink-muted: #7E7A72;
          --accent: #1D4338;
          --accent-soft: #EAF1EE;
          --pole-pos: #1D4338;
          --pole-neg: #964B32;
          --warn-bg: #FAF3E8;
          --warn-border: #E5CFA8;
          --warn-ink: #7D4E18;
        }

        /* Hide default Streamlit chrome */
        #MainMenu, footer, header[data-testid="stHeader"], div[data-testid="stToolbar"], div[data-testid="stDecoration"] {
          display: none !important;
          visibility: hidden !important;
          height: 0 !important;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
          background-color: var(--bg) !important;
          color: var(--ink) !important;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        .block-container {
          max-width: 900px !important;
          padding-top: 2.2rem !important;
          padding-bottom: 4.5rem !important;
          padding-left: 1.5rem !important;
          padding-right: 1.5rem !important;
        }

        /* Masthead */
        .masthead {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          padding-bottom: 16px;
          margin-bottom: 24px;
          border-bottom: 1px solid var(--border);
          flex-wrap: wrap;
        }
        .masthead-left {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .wordmark {
          font-family: 'Newsreader', Georgia, serif;
          font-size: 1.65rem;
          font-weight: 500;
          letter-spacing: -0.02em;
          color: var(--ink);
          line-height: 1;
        }
        .brand-sep {
          width: 1px;
          height: 16px;
          background-color: var(--border-strong);
        }
        .brand-sub {
          font-size: 0.82rem;
          color: var(--ink-muted);
        }
        .engine-pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 5px 11px;
          background-color: var(--surface);
          border: 1px solid var(--border);
          border-radius: 4px;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.74rem;
          color: var(--ink-secondary);
        }
        .engine-dot {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background-color: var(--warn-ink);
          display: inline-block;
        }
        .engine-dot.is-ml {
          background-color: var(--accent);
        }

        /* Page intros */
        .eyebrow {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--ink-muted);
          margin: 0 0 8px 0;
        }
        .editorial-h1 {
          font-family: 'Newsreader', Georgia, serif;
          font-size: 2.25rem;
          font-weight: 400;
          line-height: 1.15;
          letter-spacing: -0.02em;
          color: var(--ink);
          margin: 0 0 10px 0;
        }
        .editorial-lead {
          font-size: 0.95rem;
          line-height: 1.65;
          color: var(--ink-secondary);
          max-width: 660px;
          margin: 0 0 22px 0;
        }

        /* Tabs styled as clean editorial navigation */
        .stTabs [data-baseweb="tab-list"] {
          gap: 28px;
          border-bottom: 1px solid var(--border);
          margin-bottom: 28px;
          background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
          height: auto;
          padding: 0 0 12px 0;
          background-color: transparent !important;
          border: none !important;
          color: var(--ink-muted) !important;
          font-family: 'Inter', sans-serif !important;
          font-size: 0.88rem !important;
          font-weight: 500 !important;
        }
        .stTabs [aria-selected="true"] {
          color: var(--ink) !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
          background-color: var(--ink) !important;
          height: 2px !important;
        }

        /* Text areas like archival writing sheets */
        .stTextArea textarea {
          background-color: var(--surface) !important;
          border: 1px solid var(--border) !important;
          border-radius: 6px !important;
          padding: 16px 18px !important;
          font-family: 'Newsreader', Georgia, serif !important;
          font-size: 1.06rem !important;
          line-height: 1.7 !important;
          color: var(--ink) !important;
          box-shadow: none !important;
        }
        .stTextArea textarea:focus {
          border-color: var(--ink) !important;
          box-shadow: none !important;
        }
        .stTextArea label {
          font-family: 'Inter', sans-serif !important;
          font-size: 0.9rem !important;
          font-weight: 500 !important;
          color: var(--ink) !important;
        }

        /* Radio buttons */
        .stRadio > div {
          gap: 18px !important;
        }
        .stRadio label {
          font-size: 0.85rem !important;
          color: var(--ink-secondary) !important;
        }

        /* Buttons */
        .stButton > button {
          border-radius: 4px !important;
          font-family: 'Inter', sans-serif !important;
          font-size: 0.84rem !important;
          font-weight: 500 !important;
          padding: 0.48rem 1rem !important;
          border: 1px solid var(--border-strong) !important;
          background-color: var(--surface) !important;
          color: var(--ink-secondary) !important;
          transition: border-color 0.15s ease, color 0.15s ease, background-color 0.15s ease !important;
          box-shadow: none !important;
        }
        .stButton > button:hover {
          border-color: var(--ink) !important;
          color: var(--ink) !important;
          background-color: var(--surface) !important;
        }
        .stButton > button[kind="primary"] {
          background-color: var(--ink) !important;
          border-color: var(--ink) !important;
          color: #FFFFFF !important;
        }
        .stButton > button[kind="primary"]:hover {
          background-color: #2C2C28 !important;
          border-color: #2C2C28 !important;
          color: #FFFFFF !important;
        }

        /* Signal meter */
        .meter-strip {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          padding: 10px 14px;
          margin: 6px 0 14px 0;
          background-color: #FBFAF7;
          border: 1px solid var(--border);
          border-radius: 6px;
          flex-wrap: wrap;
        }
        .meter-left {
          display: flex;
          align-items: center;
          gap: 12px;
        }
        .mono-stat {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.78rem;
          color: var(--ink);
          font-variant-numeric: tabular-nums;
        }
        .signal-track {
          width: 96px;
          height: 5px;
          background-color: var(--surface-muted);
          border-radius: 999px;
          overflow: hidden;
        }
        .signal-fill {
          height: 100%;
          border-radius: 999px;
        }
        .meter-note {
          font-size: 0.78rem;
          color: var(--ink-muted);
        }

        /* Prompt headers */
        .prompt-head {
          display: flex;
          align-items: baseline;
          justify-content: space-between;
          gap: 12px;
          margin: 14px 0 4px 0;
        }
        .prompt-q {
          font-size: 0.93rem;
          font-weight: 500;
          color: var(--ink);
        }
        .prompt-num {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.76rem;
          color: var(--ink-muted);
          margin-right: 8px;
        }
        .prompt-wc {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          color: var(--ink-muted);
        }

        /* Results Dossier */
        .dossier-wrap {
          margin-top: 32px;
        }
        .meta-bar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
          margin-bottom: 10px;
          flex-wrap: wrap;
        }
        .meta-pills {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
        }
        .mono-pill {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          padding: 3px 9px;
          border-radius: 4px;
          border: 1px solid var(--border);
          background-color: var(--surface);
          color: var(--ink-secondary);
        }
        .dossier-card {
          background-color: var(--surface);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 30px;
        }
        .dossier-header {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 20px;
          flex-wrap: wrap;
          margin-bottom: 16px;
        }
        .type-id {
          display: flex;
          align-items: center;
          gap: 18px;
        }
        .type-code {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 2.2rem;
          font-weight: 500;
          letter-spacing: 0.08em;
          line-height: 1;
          padding: 13px 16px;
          background-color: var(--accent-soft);
          color: var(--accent);
          border: 1px solid #C9DCD5;
          border-radius: 6px;
        }
        .type-group {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--ink-muted);
          margin: 0 0 2px 0;
        }
        .type-title {
          font-family: 'Newsreader', Georgia, serif;
          font-size: 1.85rem;
          font-weight: 400;
          line-height: 1.15;
          color: var(--ink);
          margin: 0;
        }
        .stack-box {
          display: flex;
          flex-direction: column;
          align-items: flex-end;
          gap: 6px;
        }
        .stack-label {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.68rem;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--ink-muted);
        }
        .stack-row {
          display: flex;
          gap: 6px;
        }
        .fn-pill {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.78rem;
          padding: 3px 9px;
          border-radius: 4px;
          border: 1px solid var(--border);
          background-color: #FBFAF7;
          color: var(--ink);
        }
        .fn-pill.dom {
          border-color: var(--ink);
          font-weight: 500;
        }
        .type-summary {
          font-family: 'Newsreader', Georgia, serif;
          font-size: 1.1rem;
          line-height: 1.65;
          color: var(--ink-secondary);
          margin: 0;
          max-width: 72ch;
        }
        .rule {
          height: 1px;
          background-color: var(--border);
          margin: 24px 0;
        }
        .sec-head {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          margin-bottom: 16px;
          flex-wrap: wrap;
          gap: 8px;
        }
        .sec-title {
          font-size: 0.82rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: var(--ink);
          margin: 0;
        }
        .sec-cap {
          font-size: 0.78rem;
          color: var(--ink-muted);
        }

        /* Diverging spectrum bars */
        .axis-row {
          display: grid;
          grid-template-columns: 148px 1fr 148px;
          align-items: center;
          gap: 14px;
          margin-bottom: 15px;
        }
        .axis-row:last-child {
          margin-bottom: 0;
        }
        .pole {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        .pole.right {
          text-align: right;
        }
        .pole-name {
          font-size: 0.84rem;
          color: var(--ink-muted);
        }
        .pole-name.win {
          color: var(--ink);
          font-weight: 600;
        }
        .pole-pct {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.78rem;
          color: var(--ink-secondary);
          font-variant-numeric: tabular-nums;
        }
        .spec-track {
          position: relative;
          height: 10px;
          background-color: var(--surface-muted);
          border-radius: 2px;
          overflow: hidden;
        }
        .spec-mid {
          position: absolute;
          top: 0;
          bottom: 0;
          left: 50%;
          width: 1px;
          background-color: var(--border-strong);
          z-index: 2;
        }
        .spec-fill {
          position: absolute;
          top: 0;
          bottom: 0;
        }
        .spec-fill.left {
          right: 50%;
          background-color: var(--pole-pos);
        }
        .spec-fill.right {
          left: 50%;
          background-color: var(--pole-neg);
        }

        /* Two column dossier detail */
        .dossier-grid {
          display: grid;
          grid-template-columns: 1.12fr 0.88fr;
          gap: 28px;
        }
        .dossier-col-right {
          border-left: 1px solid var(--border);
          padding-left: 28px;
        }
        .top3-row {
          display: grid;
          grid-template-columns: 54px 1fr 56px;
          align-items: center;
          gap: 10px;
          padding: 9px 12px;
          border: 1px solid var(--border);
          border-radius: 4px;
          background-color: #FCFBF9;
          margin-bottom: 8px;
        }
        .top3-row.first {
          border-color: var(--ink);
          background-color: var(--surface);
        }
        .top3-code {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.88rem;
          font-weight: 500;
          color: var(--ink);
        }
        .top3-name {
          font-size: 0.84rem;
          color: var(--ink-secondary);
        }
        .top3-pct {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.8rem;
          text-align: right;
          color: var(--ink);
        }
        .sub-h4 {
          font-size: 0.8rem;
          font-weight: 600;
          color: var(--ink);
          margin: 18px 0 6px 0;
        }
        .body-p {
          font-size: 0.86rem;
          color: var(--ink-secondary);
          margin: 0;
          line-height: 1.55;
        }
        .two-lists {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 16px;
          margin-top: 6px;
        }
        .clean-ul {
          margin: 0;
          padding-left: 16px;
          font-size: 0.84rem;
          color: var(--ink-secondary);
        }
        .clean-ul li {
          margin-bottom: 4px;
        }
        .sig-row {
          display: flex;
          justify-content: space-between;
          gap: 10px;
          padding: 7px 0;
          border-bottom: 1px solid var(--border);
          font-size: 0.83rem;
        }
        .sig-row:last-child {
          border-bottom: none;
        }
        .sig-k {
          color: var(--ink-secondary);
        }
        .sig-v {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.78rem;
          color: var(--ink);
        }
        .note-box {
          margin-top: 16px;
          padding: 11px 13px;
          border-radius: 4px;
          border: 1px solid var(--border);
          background-color: #FBFAF7;
          font-size: 0.79rem;
          line-height: 1.55;
          color: var(--ink-secondary);
        }
        .note-box.warn {
          background-color: var(--warn-bg);
          border-color: var(--warn-border);
          color: var(--warn-ink);
        }

        /* Reference detail & cards */
        .ref-detail-card {
          background-color: var(--surface);
          border: 1px solid var(--ink);
          border-radius: 8px;
          padding: 24px 26px;
          margin-bottom: 24px;
        }
        .ref-card {
          background-color: var(--surface);
          border: 1px solid var(--border);
          border-radius: 6px;
          padding: 15px 16px;
          height: 100%;
          margin-bottom: 10px;
        }
        .ref-code {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 1rem;
          font-weight: 500;
          color: var(--ink);
        }
        .ref-title {
          font-family: 'Newsreader', Georgia, serif;
          font-size: 1.05rem;
          color: var(--ink);
          margin: 2px 0 4px 0;
        }
        .ref-fn {
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          color: var(--ink-muted);
          margin-bottom: 8px;
        }
        .ref-sum {
          font-size: 0.82rem;
          color: var(--ink-secondary);
          line-height: 1.5;
          margin: 0;
        }

        .status-card {
          background-color: var(--surface);
          border: 1px solid var(--border);
          border-radius: 8px;
          padding: 22px 24px;
          margin-bottom: 18px;
        }

        .footer-strip {
          margin-top: 48px;
          padding-top: 18px;
          border-top: 1px solid var(--border);
          display: flex;
          justify-content: space-between;
          gap: 12px;
          flex-wrap: wrap;
          font-size: 0.76rem;
          color: var(--ink-muted);
        }

        @media (max-width: 720px) {
          .dossier-card { padding: 20px; }
          .dossier-grid { grid-template-columns: 1fr; }
          .dossier-col-right { border-left: none; border-top: 1px solid var(--border); padding-left: 0; padding-top: 22px; }
          .axis-row { grid-template-columns: 1fr; gap: 6px; }
          .pole.right { text-align: left; }
          .stack-box { align-items: flex-start; }
          .two-lists { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or ""))


def render_meter_html(word_count: int, target_note: str = "Add 150+ words for best reliability") -> str:
    pct = min(100, int(round((word_count / 220.0) * 100)))
    if word_count == 0:
        color = "#C6C0B4"
        label = target_note
    elif word_count < 80:
        color = "#964B32"
        label = "Low signal — keep writing"
    elif word_count < 200:
        color = "#B07D32"
        label = "Moderate signal — good"
    else:
        color = "#1D4338"
        label = "High signal — ready"

    unit = "word" if word_count == 1 else "words"
    return (
        f'<div class="meter-strip">'
        f'  <div class="meter-left">'
        f'    <span class="mono-stat">{word_count} {unit}</span>'
        f'    <div class="signal-track"><div class="signal-fill" style="width:{pct}%;background:{color};"></div></div>'
        f'  </div>'
        f'  <span class="meter-note">{html.escape(label)}</span>'
        f'</div>'
    )


def build_plain_summary(res: Dict[str, Any]) -> str:
    lines = [
        "Typology Assessment Summary",
        f"Predicted Type: {res['predicted_type']} ({res['profile']['title']})",
        f"Cognitive Stack: {' - '.join(res['profile']['functions'])}",
        f"Input: {res['word_count']} words ({res['reliability_label']})",
        f"Engine: {res['engine']['mode_label']}",
        "",
        "Axis Breakdown:",
    ]
    for ax in res["axes"]:
        lines.append(
            f"  {ax['pos_letter']} {ax['pos_prob']*100:5.1f}% / {ax['neg_prob']*100:5.1f}% {ax['neg_letter']}  -> {ax['predicted_name']}"
        )
    top3 = ", ".join(f"{t['type']} ({t['probability']*100:.1f}%)" for t in res["top_3_types"])
    lines.extend(["", f"Top 3 Probable Types: {top3}"])
    return "\n".join(lines)


def render_result_dossier(res: Dict[str, Any]) -> None:
    prof = res["profile"]
    eng = res["engine"]
    eng_short = "ML Ensemble" if eng["mode"] == "ml_ensemble" else "Built-in Estimator"

    fn_pills = "".join(
        f'<span class="fn-pill{" dom" if i == 0 else ""}">{html.escape(fn)}</span>'
        for i, fn in enumerate(prof["functions"])
    )

    axes_html_parts = []
    for ax in res["axes"]:
        pos_pct = ax["pos_prob"] * 100.0
        neg_pct = ax["neg_prob"] * 100.0
        is_pos = ax["predicted_letter"] == ax["pos_letter"]
        offset = abs(ax["pos_prob"] - 0.5) * 100.0
        side = "left" if is_pos else "right"
        axes_html_parts.append(
            f'<div class="axis-row">'
            f'  <div class="pole left">'
            f'    <span class="pole-name{" win" if is_pos else ""}">{ax["pos_letter"]} &middot; {html.escape(ax["pos_name"])}</span>'
            f'    <span class="pole-pct">{pos_pct:.1f}%</span>'
            f'  </div>'
            f'  <div class="spec-track">'
            f'    <div class="spec-mid"></div>'
            f'    <div class="spec-fill {side}" style="width:{offset:.1f}%;"></div>'
            f'  </div>'
            f'  <div class="pole right">'
            f'    <span class="pole-name{" win" if not is_pos else ""}">{html.escape(ax["neg_name"])} &middot; {ax["neg_letter"]}</span>'
            f'    <span class="pole-pct">{neg_pct:.1f}%</span>'
            f'  </div>'
            f'</div>'
        )
    axes_html = "".join(axes_html_parts)

    top3_html = "".join(
        f'<div class="top3-row{" first" if i == 0 else ""}">'
        f'  <span class="top3-code">{html.escape(item["type"])}</span>'
        f'  <span class="top3-name">{html.escape(item["title"])}</span>'
        f'  <span class="top3-pct">{item["probability"]*100:.1f}%</span>'
        f'</div>'
        for i, item in enumerate(res["top_3_types"])
    )

    strengths_li = "".join(f"<li>{html.escape(s)}</li>" for s in prof["strengths"])
    blindspots_li = "".join(f"<li>{html.escape(s)}</li>" for s in prof["blind_spots"])

    s = res["signals"]
    sig_pairs = [
        ("Average word length", f"{s['avg_word_length']} chars"),
        ("Words per sentence", f"{s['words_per_sentence']}"),
        ("Lexical diversity (MATTR)", f"{s['lexical_diversity'] * 100:.1f}%"),
        ("Exclamation rate", f"{s['exclamation_rate']} / 100w"),
        ("Question rate", f"{s['question_rate']} / 100w"),
        ("First-person focus (I/my)", f"{s['first_person_rate']} / 100w"),
        ("Social pronouns (we/you)", f"{s['social_pronoun_rate']} / 100w"),
        ("Analytical vs. feeling index", f"{s['analytical_vs_feeling']:+.3f}"),
        ("Planning vs. spontaneous index", f"{s['planning_vs_spontaneous']:+.3f}"),
    ]
    sig_html = "".join(
        f'<div class="sig-row"><span class="sig-k">{html.escape(k)}</span><span class="sig-v">{html.escape(v)}</span></div>'
        for k, v in sig_pairs
    )

    if eng["mode"] == "ml_ensemble":
        note_cls = "note-box"
        note_txt = (
            "Predicted with your trained 4-axis balanced ML ensemble "
            "(Word and Character TF-IDF + 16 stylometric signals + SVD/GBDT). "
            "Explicit MBTI labels are masked so results generalize to everyday writing."
        )
    else:
        note_cls = "note-box warn"
        note_txt = (
            "Running on the built-in psycholinguistic estimator. Open the Model Setup tab "
            "to upload mbti_predictor.joblib for the full 8,675-user trained ML ensemble."
        )

    dossier_html = (
        f'<div class="dossier-wrap">'
        f'  <div class="meta-bar">'
        f'    <span class="eyebrow" style="margin:0;">Assessment Report</span>'
        f'    <div class="meta-pills">'
        f'      <span class="mono-pill">{res["word_count"]} words</span>'
        f'      <span class="mono-pill">{html.escape(res["reliability_label"])}</span>'
        f'      <span class="mono-pill">{html.escape(eng_short)}</span>'
        f'    </div>'
        f'  </div>'
        f'  <div class="dossier-card">'
        f'    <div class="dossier-header">'
        f'      <div class="type-id">'
        f'        <div class="type-code">{html.escape(res["predicted_type"])}</div>'
        f'        <div>'
        f'          <p class="type-group">{html.escape(prof["group"])}</p>'
        f'          <h2 class="type-title">{html.escape(prof["title"])}</h2>'
        f'        </div>'
        f'      </div>'
        f'      <div class="stack-box">'
        f'        <span class="stack-label">Cognitive Stack</span>'
        f'        <div class="stack-row">{fn_pills}</div>'
        f'      </div>'
        f'    </div>'
        f'    <p class="type-summary">{html.escape(prof["summary"])}</p>'
        f'    <div class="rule"></div>'
        f'    <div class="sec-head">'
        f'      <h3 class="sec-title">Four Dichotomy Spectrums</h3>'
        f'      <span class="sec-cap">Center line marks 50% decision threshold</span>'
        f'    </div>'
        f'    {axes_html}'
        f'    <div class="rule"></div>'
        f'    <div class="dossier-grid">'
        f'      <div>'
        f'        <h3 class="sec-title" style="margin-bottom:12px;">Most Probable Types</h3>'
        f'        {top3_html}'
        f'        <h4 class="sub-h4">Communication Style</h4>'
        f'        <p class="body-p">{html.escape(prof["communication"])}</p>'
        f'        <div class="two-lists">'
        f'          <div>'
        f'            <h4 class="sub-h4">Core Strengths</h4>'
        f'            <ul class="clean-ul">{strengths_li}</ul>'
        f'          </div>'
        f'          <div>'
        f'            <h4 class="sub-h4">Potential Blind Spots</h4>'
        f'            <ul class="clean-ul">{blindspots_li}</ul>'
        f'          </div>'
        f'        </div>'
        f'      </div>'
        f'      <div class="dossier-col-right">'
        f'        <h3 class="sec-title" style="margin-bottom:4px;">Observed Linguistic Signals</h3>'
        f'        <p class="sec-cap" style="margin:0 0 10px 0;">Length-normalized measurements extracted from your writing.</p>'
        f'        {sig_html}'
        f'        <div class="{note_cls}">{html.escape(note_txt)}</div>'
        f'      </div>'
        f'    </div>'
        f'  </div>'
        f'</div>'
    )
    st.markdown(dossier_html, unsafe_allow_html=True)

    with st.expander("Copyable Plain-Text Summary"):
        st.code(build_plain_summary(res), language=None)


# --- Session State Initialization ---
def init_state() -> None:
    if "engine" not in st.session_state:
        st.session_state.engine = MBTIEngine(
            DEFAULT_MODEL_PATH if DEFAULT_MODEL_PATH.exists() else None
        )
    if "free_text" not in st.session_state:
        st.session_state.free_text = ""
    if "multi_count" not in st.session_state:
        st.session_state.multi_count = 3
    for i in range(8):
        if f"entry_{i}" not in st.session_state:
            st.session_state[f"entry_{i}"] = ""
    for i in range(len(PROMPTS)):
        if f"prompt_{i}" not in st.session_state:
            st.session_state[f"prompt_{i}"] = ""
    if "last_result_analyze" not in st.session_state:
        st.session_state.last_result_analyze = None
    if "last_result_prompts" not in st.session_state:
        st.session_state.last_result_prompts = None
    if "selected_ref_type" not in st.session_state:
        st.session_state.selected_ref_type = "INTJ"
    if "upload_notice" not in st.session_state:
        st.session_state.upload_notice = None


def load_sample_cb(sample_key: str, mode: str) -> None:
    posts = SAMPLES[sample_key]
    if mode == "Single Passage":
        st.session_state.free_text = "\n\n".join(posts)
    else:
        st.session_state.multi_count = len(posts)
        for i in range(8):
            st.session_state[f"entry_{i}"] = posts[i] if i < len(posts) else ""


def clear_analyze_cb(mode: str) -> None:
    st.session_state.last_result_analyze = None
    if mode == "Single Passage":
        st.session_state.free_text = ""
    else:
        for i in range(8):
            st.session_state[f"entry_{i}"] = ""


def clear_prompts_cb() -> None:
    st.session_state.last_result_prompts = None
    for i in range(len(PROMPTS)):
        st.session_state[f"prompt_{i}"] = ""


def add_entry_cb() -> None:
    if st.session_state.multi_count < 8:
        st.session_state.multi_count += 1


def main() -> None:
    init_state()
    inject_editorial_css()

    engine: MBTIEngine = st.session_state.engine
    status = engine.get_status()
    is_ml = status["mode"] == "ml_ensemble"
    dot_cls = "engine-dot is-ml" if is_ml else "engine-dot"
    pill_text = "ML Ensemble" if is_ml else "Estimator Mode"

    # Masthead
    st.markdown(
        f'<div class="masthead">'
        f'  <div class="masthead-left">'
        f'    <span class="wordmark">Typology</span>'
        f'    <span class="brand-sep"></span>'
        f'    <span class="brand-sub">Written Language Assessment</span>'
        f'  </div>'
        f'  <span class="engine-pill"><span class="{dot_cls}"></span>{html.escape(pill_text)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    tab_analyze, tab_prompts, tab_ref, tab_model = st.tabs(
        ["Analyze", "Guided Prompts", "16 Types", "Model Setup"]
    )

    # ------------------------------------------------------------------
    # TAB 1: FREE-FORM WRITING ANALYSIS
    # ------------------------------------------------------------------
    with tab_analyze:
        st.markdown(
            '<p class="eyebrow">Four-Axis Psycholinguistic Inference</p>'
            '<h1 class="editorial-h1">Read the structure behind how someone writes.</h1>'
            '<p class="editorial-lead">'
            'Paste essays, journal notes, messages, or multiple short posts. The model evaluates '
            'vocabulary patterns, phrasing cadence, and sixteen length-normalized linguistic signals '
            'across Introversion/Extraversion, Intuition/Sensing, Thinking/Feeling, and Judging/Perceiving.'
            '</p>',
            unsafe_allow_html=True,
        )

        input_mode = st.radio(
            "Input format",
            ["Single Passage", "Multiple Entries"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # Sample loader row
        s_cols = st.columns([1.1, 1.35, 1.45, 1.45])
        with s_cols[0]:
            st.markdown(
                '<div style="padding-top:7px;font-size:0.8rem;color:#7E7A72;">Try a sample:</div>',
                unsafe_allow_html=True,
            )
        with s_cols[1]:
            st.button(
                "Systems Thinker",
                key="sample_arch",
                on_click=load_sample_cb,
                args=("architect", input_mode),
            )
        with s_cols[2]:
            st.button(
                "Creative Connector",
                key="sample_camp",
                on_click=load_sample_cb,
                args=("campaigner", input_mode),
            )
        with s_cols[3]:
            st.button(
                "Practical Organizer",
                key="sample_logi",
                on_click=load_sample_cb,
                args=("logistician", input_mode),
            )

        if input_mode == "Single Passage":
            text_val = st.text_area(
                "Writing sample",
                key="free_text",
                height=240,
                label_visibility="collapsed",
                placeholder=(
                    "Write naturally in your own voice, or paste several paragraphs you have already written. "
                    "Avoid typing explicit MBTI labels; focus on ordinary thoughts, stories, opinions, or plans."
                ),
            )
            entries = [text_val.strip()] if text_val.strip() else []
        else:
            st.markdown(
                '<p style="font-size:0.84rem;color:#7E7A72;margin:4px 0 8px 0;">'
                'Enter separate posts, emails, or short reflections to preserve natural cadence across topics.'
                '</p>',
                unsafe_allow_html=True,
            )
            entries = []
            for i in range(st.session_state.multi_count):
                val = st.text_area(
                    f"Entry {i + 1:02d}",
                    key=f"entry_{i}",
                    height=95,
                    placeholder="Write or paste an entry...",
                )
                if val.strip():
                    entries.append(val.strip())
            if st.session_state.multi_count < 8:
                st.button("+ Add another entry", key="add_entry", on_click=add_entry_cb)

        total_words = sum(count_words(e) for e in entries)
        st.markdown(render_meter_html(total_words), unsafe_allow_html=True)

        a_left, a_mid, a_right = st.columns([3.5, 1, 1.4])
        with a_mid:
            st.button(
                "Clear",
                key="clear_analyze",
                on_click=clear_analyze_cb,
                args=(input_mode,),
                use_container_width=True,
            )
        with a_right:
            run_analyze = st.button(
                "Analyze Writing",
                key="run_analyze",
                type="primary",
                use_container_width=True,
            )

        if run_analyze:
            if total_words < 8:
                st.error(
                    "Input is too short. Write at least 8 words (150+ words recommended for reliable results)."
                )
            else:
                payload = entries if len(entries) > 1 else entries[0]
                st.session_state.last_result_analyze = engine.predict(payload)

        if st.session_state.last_result_analyze is not None:
            render_result_dossier(st.session_state.last_result_analyze)

    # ------------------------------------------------------------------
    # TAB 2: GUIDED 6-QUESTION INTERVIEW
    # ------------------------------------------------------------------
    with tab_prompts:
        st.markdown(
            '<p class="eyebrow">Structured Writing Interview</p>'
            '<h1 class="editorial-h1">Six open-ended questions, answered in your normal voice.</h1>'
            '<p class="editorial-lead">'
            'Short single sentences rarely carry enough signal to separate personality traits reliably. '
            'Answer two to four honest sentences per prompt to assemble a balanced 180-350 word profile.'
            '</p>',
            unsafe_allow_html=True,
        )

        prompt_answers: List[str] = []
        for idx, item in enumerate(PROMPTS):
            current_wc = count_words(st.session_state[f"prompt_{idx}"])
            wc_unit = "word" if current_wc == 1 else "words"
            st.markdown(
                f'<div class="prompt-head">'
                f'  <span class="prompt-q"><span class="prompt-num">{item["num"]}</span>{html.escape(item["question"])}</span>'
                f'  <span class="prompt-wc">{current_wc} {wc_unit}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            ans = st.text_area(
                f"Prompt {item['num']}",
                key=f"prompt_{idx}",
                height=96,
                label_visibility="collapsed",
                placeholder=item["placeholder"],
            )
            if ans.strip():
                prompt_answers.append(ans.strip())

        prompt_words = sum(count_words(a) for a in prompt_answers)
        st.markdown(
            render_meter_html(prompt_words, target_note="Target: 180+ total words across prompts"),
            unsafe_allow_html=True,
        )

        p_left, p_mid, p_right = st.columns([3.3, 1, 1.6])
        with p_mid:
            st.button(
                "Reset",
                key="clear_prompts",
                on_click=clear_prompts_cb,
                use_container_width=True,
            )
        with p_right:
            run_prompts = st.button(
                "Analyze Responses",
                key="run_prompts",
                type="primary",
                use_container_width=True,
            )

        if run_prompts:
            if prompt_words < 8:
                st.error("Please answer at least one or two prompts (8+ words minimum; 180+ recommended).")
            else:
                st.session_state.last_result_prompts = engine.predict(prompt_answers)

        if st.session_state.last_result_prompts is not None:
            render_result_dossier(st.session_state.last_result_prompts)

    # ------------------------------------------------------------------
    # TAB 3: 16 TYPES REFERENCE
    # ------------------------------------------------------------------
    with tab_ref:
        st.markdown(
            '<p class="eyebrow">Typology Index</p>'
            '<h1 class="editorial-h1">The sixteen personality archetypes.</h1>'
            '<p class="editorial-lead">'
            'Inspect any personality type to review its cognitive function stack, communication pattern, '
            'core strengths, and characteristic blind spots.'
            '</p>',
            unsafe_allow_html=True,
        )

        all_codes = list(MBTI_PROFILES.keys())
        sel_col1, sel_col2 = st.columns([1.2, 1.8])
        with sel_col1:
            selected_code = st.selectbox(
                "Inspect a specific type",
                options=all_codes,
                index=all_codes.index(st.session_state.selected_ref_type),
                format_func=lambda c: f"{c} — {MBTI_PROFILES[c]['title']}",
            )
            st.session_state.selected_ref_type = selected_code
        with sel_col2:
            group_filter = st.radio(
                "Filter grid by temperament",
                ["All"] + GROUP_ORDER,
                horizontal=True,
            )

        info = MBTI_PROFILES[selected_code]
        st.markdown(
            f'<div class="ref-detail-card">'
            f'  <p class="eyebrow">{html.escape(info["group"])} &middot; {" - ".join(info["functions"])}</p>'
            f'  <h2 class="type-title" style="margin-bottom:10px;">'
            f'    <span style="font-family:\'IBM Plex Mono\',monospace;color:#1D4338;margin-right:10px;">{selected_code}</span>'
            f'    {html.escape(info["title"])}'
            f'  </h2>'
            f'  <p class="type-summary">{html.escape(info["summary"])}</p>'
            f'  <div class="rule"></div>'
            f'  <h4 class="sub-h4" style="margin-top:0;">Communication Style</h4>'
            f'  <p class="body-p">{html.escape(info["communication"])}</p>'
            f'  <div class="two-lists">'
            f'    <div>'
            f'      <h4 class="sub-h4">Core Strengths</h4>'
            f'      <ul class="clean-ul">{"".join(f"<li>{html.escape(x)}</li>" for x in info["strengths"])}</ul>'
            f'    </div>'
            f'    <div>'
            f'      <h4 class="sub-h4">Potential Blind Spots</h4>'
            f'      <ul class="clean-ul">{"".join(f"<li>{html.escape(x)}</li>" for x in info["blind_spots"])}</ul>'
            f'    </div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        groups_to_show = GROUP_ORDER if group_filter == "All" else [group_filter]
        for grp in groups_to_show:
            st.markdown(f'<p class="eyebrow" style="margin-top:18px;">{html.escape(grp)}</p>', unsafe_allow_html=True)
            grp_items = [(c, d) for c, d in MBTI_PROFILES.items() if d["group"] == grp]
            cols = st.columns(2)
            for i, (code, data) in enumerate(grp_items):
                with cols[i % 2]:
                    st.markdown(
                        f'<div class="ref-card">'
                        f'  <div class="ref-code">{code}</div>'
                        f'  <div class="ref-title">{html.escape(data["title"])}</div>'
                        f'  <div class="ref-fn">{" · ".join(data["functions"])}</div>'
                        f'  <p class="ref-sum">{html.escape(data["summary"])}</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ------------------------------------------------------------------
    # TAB 4: MODEL SETUP & HOT-UPLOAD
    # ------------------------------------------------------------------
    with tab_model:
        st.markdown(
            '<p class="eyebrow">Inference Backend</p>'
            '<h1 class="editorial-h1">Model configuration.</h1>'
            '<p class="editorial-lead">'
            'The app runs out-of-the-box with a calibrated psycholinguistic estimator. '
            'Upload a trained <code>mbti_predictor.joblib</code> bundle from the notebook or <code>train.py</code> '
            'to switch live to the full 8,675-user four-axis ML ensemble.'
            '</p>',
            unsafe_allow_html=True,
        )

        cur = engine.get_status()
        file_label = cur["model_file"] if cur["model_file"] else "None loaded (built-in estimator active)"
        mask_label = "Enabled (anti-leakage)" if cur["remove_mbti_words"] else "Disabled"
        st.markdown(
            f'<div class="status-card">'
            f'  <div class="sig-row"><span class="sig-k">Active Engine</span><span class="sig-v">{html.escape(cur["mode_label"])}</span></div>'
            f'  <div class="sig-row"><span class="sig-k">Loaded Artifact</span><span class="sig-v">{html.escape(file_label)}</span></div>'
            f'  <div class="sig-row"><span class="sig-k">MBTI Keyword Masking</span><span class="sig-v">{html.escape(mask_label)}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "Upload trained mbti_predictor.joblib",
            type=["joblib"],
            help="Export mbti_predictor.joblib from Cell 16 of MBTI_prediction_improved.ipynb or run python train.py.",
        )

        if uploaded is not None:
            raw_bytes = uploaded.getvalue()
            if st.session_state.get("last_uploaded_sig") != (uploaded.name, len(raw_bytes)):
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tf:
                        tf.write(raw_bytes)
                        tmp_path = Path(tf.name)
                    ok, msg = engine.load_model(tmp_path)
                    st.session_state.last_uploaded_sig = (uploaded.name, len(raw_bytes))
                    st.session_state.upload_notice = (ok, msg)
                    if ok:
                        engine.model_path = Path(uploaded.name)
                        st.rerun()
                except Exception as exc:
                    st.session_state.upload_notice = (False, f"Could not load uploaded model: {exc}")

        if st.session_state.upload_notice is not None:
            ok, msg = st.session_state.upload_notice
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        st.markdown(
            '<div class="note-box" style="margin-top:20px;">'
            '<strong>How to train the full ML ensemble:</strong> Run <code>MBTI_prediction_improved.ipynb</code> '
            'in Google Colab (Cell 16 writes a portable <code>mbti_predictor.joblib</code>) or run '
            '<code>python train.py --data mbti_1.csv.zip</code> locally, then either commit '
            '<code>mbti_predictor.joblib</code> next to <code>streamlit_app.py</code> or upload it above.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        '<div class="footer-strip">'
        '  <span>Typology — Four-axis MBTI text inference.</span>'
        '  <span>Predictions reflect statistical writing patterns, not clinical assessment.</span>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
