"""
Core linguistic feature extraction, ML ensemble inference, and fallback
psycholinguistic estimator for the Typology MBTI web application.
"""

from __future__ import annotations

import itertools
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import joblib
    from scipy import sparse
    HAS_ML_DEPS = HAS_NUMPY
except ImportError:
    joblib = None
    sparse = None
    HAS_ML_DEPS = False


DIMENSIONS: List[Tuple[str, str, str, str]] = [
    ("IE", "I", "E", "Introversion vs. Extraversion"),
    ("NS", "N", "S", "Intuition vs. Sensing"),
    ("TF", "T", "F", "Thinking vs. Feeling"),
    ("JP", "J", "P", "Judging vs. Perceiving"),
]

AXIS_LABELS: Dict[str, Dict[str, str]] = {
    "IE": {"pos_name": "Introversion", "neg_name": "Extraversion"},
    "NS": {"pos_name": "Intuition", "neg_name": "Sensing"},
    "TF": {"pos_name": "Thinking", "neg_name": "Feeling"},
    "JP": {"pos_name": "Judging", "neg_name": "Perceiving"},
}

MBTI_TYPES = [
    "infj", "infp", "intj", "intp",
    "isfj", "isfp", "istj", "istp",
    "enfj", "enfp", "entj", "entp",
    "esfj", "esfp", "estj", "estp",
]

MBTI_PATTERN = re.compile(
    r"\b(?:" + "|".join(MBTI_TYPES) + r")s?\b|\bmbti\b|\bpersonalitycafe\b",
    flags=re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMOTICON_PATTERN = re.compile(
    r"(?::|;|=)(?:-)?(?:\)|\(|D|P|p|/|\\|O|o|\*)|<3|\b[xX]D\b|\^_\^"
)

FIRST_PERSON_SG = {"i", "me", "my", "mine", "myself", "im", "ive", "id"}
FIRST_PERSON_PL = {"we", "us", "our", "ours", "ourselves", "were", "weve"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves", "youre", "youve"}
THIRD_PERSON = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}
NEGATIONS = {
    "not", "no", "never", "none", "nothing", "neither", "nor",
    "dont", "cant", "wont", "isnt", "arent", "wasnt", "werent", "didnt",
}

ANALYTICAL_WORDS = {
    "think", "thought", "logic", "logical", "reason", "rational", "analyze", "analysis",
    "theory", "system", "objective", "data", "fact", "facts", "evidence", "efficient",
    "because", "therefore", "however", "actually", "concept", "argument", "debate",
    "strategy", "framework", "proof", "criteria", "metrics", "evaluate", "tradeoff",
    "tradeoffs", "efficiency", "accurate", "precision", "principle", "principles",
}
FEELING_WORDS = {
    "feel", "feeling", "feelings", "love", "heart", "care", "caring", "empathy",
    "compassion", "happy", "sad", "hurt", "hope", "sorry", "warm", "kind", "beautiful",
    "emotional", "harmony", "values", "passion", "grateful", "appreciate", "relationship",
    "meaningful", "trust", "support", "supportive", "gentle", "connect", "connection",
    "kindness", "sincere", "authentic", "soul",
}
PLANNING_WORDS = {
    "plan", "plans", "planning", "schedule", "organize", "organized", "structure",
    "structured", "deadline", "deadlines", "goal", "goals", "finish", "complete",
    "list", "order", "routine", "decide", "decided", "always", "milestone", "milestones",
    "checklist", "prepare", "prepared", "discipline", "systematic", "methodical",
    "commit", "commitment", "punctual", "priority", "priorities", "strict",
}
SPONTANEOUS_WORDS = {
    "maybe", "might", "whatever", "spontaneous", "random", "flexible", "explore",
    "options", "depends", "later", "procrastinate", "sometimes", "guess", "wonder",
    "improvise", "curious", "wing", "open", "surprises", "adventure", "adventures",
    "wander", "flow", "unplanned", "impulse", "shift", "adapt", "experiment",
}

# Additional domain lexicons for the built-in estimator fallback
INTROVERSION_WORDS = {
    "quiet", "alone", "solitude", "reading", "read", "book", "books", "reflect",
    "reflection", "introspect", "introspective", "private", "drain", "drained",
    "draining", "recharge", "home", "indoors", "journal", "journaling", "deep",
    "silence", "peaceful", "calm", "independent", "solo", "headphones", "listening",
    "observe", "observer", "reserved", "overwhelming", "crowds", "noisy",
}
EXTRAVERSION_WORDS = {
    "party", "parties", "friends", "people", "social", "energized", "energy", "chat",
    "talking", "talk", "crowd", "crowds", "meet", "meeting", "team", "group",
    "hangout", "lively", "outgoing", "exciting", "excited", "fun", "everyone",
    "gathering", "gatherings", "conversation", "conversations", "network", "networking",
    "bouncing", "share", "sharing", "together", "celebrate", "hosted", "hosting",
}
INTUITION_WORDS = {
    "abstract", "pattern", "patterns", "theory", "theoretical", "concept", "conceptual",
    "philosophy", "philosophical", "future", "possibilities", "possibility", "imagine",
    "imagining", "imagination", "metaphor", "meaning", "vision", "framework", "frameworks",
    "ideas", "idea", "hypothetical", "underlying", "symbolism", "architecture", "paradigm",
    "insight", "insights", "innovative", "speculate", "complexity", "systems", "dream",
    "dreams", "curiosity", "why",
}
SENSING_WORDS = {
    "practical", "hands", "concrete", "detail", "details", "specific", "specifics",
    "realistic", "reality", "facts", "fact", "proven", "step", "steps", "routine",
    "experience", "experienced", "tangible", "physical", "everyday", "present", "tool",
    "tools", "build", "fix", "repair", "cooking", "sports", "budget", "sensible",
    "tradition", "traditional", "reliable", "procedure", "procedures", "directly",
    "material", "observable",
}


MBTI_PROFILES: Dict[str, Dict[str, Any]] = {
    "INTJ": {
        "title": "The Architect",
        "group": "Analysts",
        "functions": ["Ni", "Te", "Fi", "Se"],
        "summary": "Strategic, independent, and systems-minded. Prefers long-range synthesis, clear logical standards, and self-directed execution.",
        "communication": "Concise, structured, and direct. Favors underlying principles over small talk.",
        "strengths": ["Long-term planning", "Independent critical thinking", "Pattern synthesis"],
        "blind_spots": ["Impatience with inefficiency", "Over-intellectualizing interpersonal friction"],
    },
    "INTP": {
        "title": "The Logician",
        "group": "Analysts",
        "functions": ["Ti", "Ne", "Si", "Fe"],
        "summary": "Curious, analytical, and conceptually driven. Enjoys dissecting how ideas fit together and testing assumptions from multiple angles.",
        "communication": "Precise, exploratory, and qualification-rich. Often questions premises before agreeing.",
        "strengths": ["First-principles reasoning", "Conceptual originality", "Intellectual openness"],
        "blind_spots": ["Endless refinement over closure", "Understating emotional subtext"],
    },
    "ENTJ": {
        "title": "The Commander",
        "group": "Analysts",
        "functions": ["Te", "Ni", "Se", "Fi"],
        "summary": "Decisive, ambitious, and structurally focused. Naturally organizes people, constraints, and timelines around clear strategic outcomes.",
        "communication": "Assertive, action-oriented, and transparent. Prioritizes clarity and momentum.",
        "strengths": ["Executive alignment", "Decisive problem-solving", "Strategic execution"],
        "blind_spots": ["Coming across as overly forceful", "Overlooking slower consensus needs"],
    },
    "ENTP": {
        "title": "The Debater",
        "group": "Analysts",
        "functions": ["Ne", "Ti", "Fe", "Si"],
        "summary": "Quick-witted, inventive, and energized by intellectual sparring. Connects disparate concepts and challenges entrenched conventions.",
        "communication": "Lively, associative, and playful. Uses analogies, hypotheticals, and counter-arguments.",
        "strengths": ["Lateral ideation", "Adaptive persuasion", "Spotting hidden possibilities"],
        "blind_spots": ["Losing interest during routine maintenance", "Debating when empathy is needed"],
    },
    "INFJ": {
        "title": "The Advocate",
        "group": "Diplomats",
        "functions": ["Ni", "Fe", "Ti", "Se"],
        "summary": "Insightful, reflective, and values-centered. Combines quiet long-term vision with deep sensitivity to human motivation and meaning.",
        "communication": "Thoughtful, nuanced, and empathetic. Frequently frames observations around purpose.",
        "strengths": ["Reading interpersonal nuance", "Quiet conviction", "Meaningful mentorship"],
        "blind_spots": ["Absorbing others' stress", "Setting unrealistically high internal standards"],
    },
    "INFP": {
        "title": "The Mediator",
        "group": "Diplomats",
        "functions": ["Fi", "Ne", "Si", "Te"],
        "summary": "Imaginative, introspective, and deeply anchored in personal ethics. Seeks authenticity, creative expression, and emotional resonance.",
        "communication": "Reflective, expressive, and individualized. Attuned to tone, sincerity, and metaphor.",
        "strengths": ["Moral clarity", "Creative empathy", "Open-minded listening"],
        "blind_spots": ["Avoiding necessary confrontation", "Struggling with rigid bureaucracy"],
    },
    "ENFJ": {
        "title": "The Protagonist",
        "group": "Diplomats",
        "functions": ["Fe", "Ni", "Se", "Ti"],
        "summary": "Warm, articulate, and people-galvanizing. Excels at aligning groups around shared ideals and bringing out others' potential.",
        "communication": "Encouraging, cohesive, and socially attuned. Speaks in terms of shared growth.",
        "strengths": ["Facilitating collaboration", "Clear empathetic leadership", "Reading group morale"],
        "blind_spots": ["Over-committing to help others", "Taking disagreement personally"],
    },
    "ENFP": {
        "title": "The Campaigner",
        "group": "Diplomats",
        "functions": ["Ne", "Fi", "Te", "Si"],
        "summary": "Enthusiastic, curious, and human-centered. Draws energy from fresh possibilities, expressive storytelling, and authentic connection.",
        "communication": "Expressive, warm, and associative. Jumps readily between ideas and personal stories.",
        "strengths": ["Creative energy", "Relational warmth", "Reframing stale problems"],
        "blind_spots": ["Spreading attention across too many threads", "Resisting repetitive structure"],
    },
    "ISTJ": {
        "title": "The Logistician",
        "group": "Sentinels",
        "functions": ["Si", "Te", "Fi", "Ne"],
        "summary": "Methodical, dependable, and grounded in verifiable facts. Values thoroughness, clear commitments, and proven systems that last.",
        "communication": "Measured, factual, and orderly. Prefers concrete evidence over speculation.",
        "strengths": ["Operational reliability", "Detail accuracy", "Steady follow-through"],
        "blind_spots": ["Skepticism toward untested shifts", "Under-expressing appreciation aloud"],
    },
    "ISFJ": {
        "title": "The Defender",
        "group": "Sentinels",
        "functions": ["Si", "Fe", "Ti", "Ne"],
        "summary": "Attentive, loyal, and quietly supportive. Remembers practical details that matter to others and sustains stability behind the scenes.",
        "communication": "Considerate, grounded, and diplomatic. Focused on practical care and harmony.",
        "strengths": ["Consistent dependability", "Thoughtful stewardship", "Practical empathy"],
        "blind_spots": ["Difficulty saying no", "Hesitating to voice personal needs early"],
    },
    "ESTJ": {
        "title": "The Executive",
        "group": "Sentinels",
        "functions": ["Te", "Si", "Ne", "Fi"],
        "summary": "Direct, organized, and results-minded. Brings structure, clear accountability, and pragmatic standards to teams and projects.",
        "communication": "Crisp, unambiguous, and pragmatic. Focuses on responsibilities and next steps.",
        "strengths": ["Project coordination", "Clear accountability", "Pragmatic decision-making"],
        "blind_spots": ["Impatience with ambiguity", "Focusing on rules before context"],
    },
    "ESFJ": {
        "title": "The Consul",
        "group": "Sentinels",
        "functions": ["Fe", "Si", "Ne", "Ti"],
        "summary": "Sociable, conscientious, and community-oriented. Builds warm, well-organized environments where people feel welcomed and supported.",
        "communication": "Friendly, attentive, and practical. Keeps conversations inclusive and grounded.",
        "strengths": ["Community building", "Reliable coordination", "Attentive hospitality"],
        "blind_spots": ["Sensitivity to criticism", "Discomfort with unstructured conflict"],
    },
    "ISTP": {
        "title": "The Virtuoso",
        "group": "Explorers",
        "functions": ["Ti", "Se", "Ni", "Fe"],
        "summary": "Calm, pragmatic, and hands-on. Understands mechanisms through direct experimentation and stays composed under immediate pressure.",
        "communication": "Sparse, matter-of-fact, and economical. Values action and demonstration over theory.",
        "strengths": ["Real-time troubleshooting", "Mechanical and technical intuition", "Calm autonomy"],
        "blind_spots": ["Disengaging from long-term planning", "Keeping feelings guarded"],
    },
    "ISFP": {
        "title": "The Adventurer",
        "group": "Explorers",
        "functions": ["Fi", "Se", "Ni", "Te"],
        "summary": "Quietly expressive, aesthetic, and present-focused. Lives by deeply felt personal taste and responds authentically to the immediate moment.",
        "communication": "Gentle, understated, and personal. Often communicates values through craft or choices.",
        "strengths": ["Aesthetic sensibility", "Quiet authenticity", "Adaptable presence"],
        "blind_spots": ["Disliking rigid schedules", "Avoiding difficult direct confrontation"],
    },
    "ESTP": {
        "title": "The Entrepreneur",
        "group": "Explorers",
        "functions": ["Se", "Ti", "Fe", "Ni"],
        "summary": "Energetic, observant, and action-biased. Reads immediate dynamics rapidly and prefers tangible momentum over prolonged deliberation.",
        "communication": "Direct, brisk, and engaging. Anchored in real examples and immediate outcomes.",
        "strengths": ["Fast tactical judgment", "High energy under pressure", "Persuasive presence"],
        "blind_spots": ["Overlooking long-range consequences", "Restlessness with slow routines"],
    },
    "ESFP": {
        "title": "The Entertainer",
        "group": "Explorers",
        "functions": ["Se", "Fi", "Te", "Ni"],
        "summary": "Spontaneous, warm, and vividly engaged with the present. Brings levity, practical generosity, and vitality to everyday experiences.",
        "communication": "Lively, candid, and story-driven. Naturally responsive to the room's energy.",
        "strengths": ["Creating momentum and warmth", "Observing real-time cues", "Flexible optimism"],
        "blind_spots": ["Putting off dry administrative tasks", "Avoiding heavy long-term constraints"],
    },
}


def clean_text_single(text: str, remove_mbti_words: bool = True) -> str:
    text = str(text)
    text = URL_PATTERN.sub(" ", text)
    text = text.replace("|||", " ")
    text = text.lower()
    text = re.sub(r"['’]", "", text)
    if remove_mbti_words:
        text = MBTI_PATTERN.sub(" ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_stylometry_single(raw_text: str) -> Tuple[List[float], Dict[str, float]]:
    """Extracts 16 length-invariant stylometric features matching the notebook pipeline."""
    raw = str(raw_text)
    n_urls = len(URL_PATTERN.findall(raw))
    n_emoticons = len(EMOTICON_PATTERN.findall(raw))
    no_url = URL_PATTERN.sub(" ", raw)

    n_exclam = no_url.count("!")
    n_question = no_url.count("?")
    n_ellipsis = len(re.findall(r"\.{3,}", no_url))

    segments = [s.strip() for s in re.split(r"\|\|\||[.!?]+", no_url) if s.strip()]
    n_segments = max(len(segments), 1)

    raw_words = re.findall(r"[A-Za-z]+", no_url)
    n_words = max(len(raw_words), 1)
    scale = 100.0 / n_words

    n_allcaps = sum(1 for w in raw_words if len(w) >= 2 and w.isupper())
    avg_word_len = sum(len(w) for w in raw_words) / n_words
    words_per_sent = min(n_words / n_segments, 60.0)

    norm_text = re.sub(r"['’]", "", no_url.lower())
    words = re.findall(r"[a-z]+", norm_text)
    n_norm = max(len(words), 1)
    norm_scale = 100.0 / n_norm

    win = 50
    if len(words) <= win:
        mattr = len(set(words)) / n_norm
    else:
        step = max(1, (len(words) - win) // 20)
        ttrs = [len(set(words[i : i + win])) / win for i in range(0, len(words) - win + 1, step)]
        mattr = float(sum(ttrs) / len(ttrs))

    c_1sg = sum(1 for w in words if w in FIRST_PERSON_SG)
    c_1pl = sum(1 for w in words if w in FIRST_PERSON_PL)
    c_2p = sum(1 for w in words if w in SECOND_PERSON)
    c_3p = sum(1 for w in words if w in THIRD_PERSON)
    c_neg = sum(1 for w in words if w in NEGATIONS)
    c_ana = sum(1 for w in words if w in ANALYTICAL_WORDS)
    c_feel = sum(1 for w in words if w in FEELING_WORDS)
    c_plan = sum(1 for w in words if w in PLANNING_WORDS)
    c_spon = sum(1 for w in words if w in SPONTANEOUS_WORDS)

    ana_vs_feel = (c_ana - c_feel) / (c_ana + c_feel + 1.0)
    plan_vs_spon = (c_plan - c_spon) / (c_plan + c_spon + 1.0)

    vec = [
        n_exclam * scale,
        n_question * scale,
        n_ellipsis * scale,
        n_allcaps * scale,
        n_emoticons * scale,
        n_urls * scale,
        avg_word_len,
        words_per_sent,
        mattr,
        c_1sg * norm_scale,
        c_1pl * norm_scale,
        c_2p * norm_scale,
        c_3p * norm_scale,
        c_neg * norm_scale,
        ana_vs_feel,
        plan_vs_spon,
    ]

    readable = {
        "word_count": len(raw_words),
        "avg_word_length": round(avg_word_len, 2),
        "words_per_sentence": round(words_per_sent, 1),
        "lexical_diversity": round(mattr, 3),
        "exclamation_rate": round(n_exclam * scale, 2),
        "question_rate": round(n_question * scale, 2),
        "first_person_rate": round(c_1sg * norm_scale, 2),
        "social_pronoun_rate": round((c_1pl + c_2p) * norm_scale, 2),
        "analytical_vs_feeling": round(ana_vs_feel, 3),
        "planning_vs_spontaneous": round(plan_vs_spon, 3),
    }
    return vec, readable


def align_proba_to_threshold(p: float, thr: float) -> float:
    if p >= thr:
        return 0.5 + 0.5 * (p - thr) / max(1.0 - thr, 1e-6)
    return 0.5 * p / max(thr, 1e-6)


def _sigmoid(x: float) -> float:
    x = max(-8.0, min(8.0, x))
    return 1.0 / (1.0 + math.exp(-x))


class MBTIEngine:
    """Unified prediction engine supporting trained ML bundles and built-in fallback."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path: Optional[Path] = None
        self.bundle: Optional[Dict[str, Any]] = None
        self.mode: str = "heuristic_estimator"
        self.load_error: Optional[str] = None
        self.all_16_combos = list(itertools.product(
            [("I", 1), ("E", 0)],
            [("N", 1), ("S", 0)],
            [("T", 1), ("F", 0)],
            [("J", 1), ("P", 0)],
        ))

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, path: Union[str, Path]) -> Tuple[bool, str]:
        p = Path(path)
        if not p.exists():
            self.mode = "heuristic_estimator"
            self.bundle = None
            self.model_path = None
            return False, f"Model file not found at {p}"

        if not HAS_ML_DEPS:
            self.mode = "heuristic_estimator"
            self.load_error = "joblib or scipy not installed"
            return False, "Cannot load ML model: install scikit-learn, scipy, and joblib."

        try:
            loaded = joblib.load(p)
            if isinstance(loaded, dict) and "axis_bundles" in loaded and "word_tfidf" in loaded:
                self.bundle = loaded
                self.model_path = p
                self.mode = "ml_ensemble"
                self.load_error = None
                return True, f"Loaded trained 4-axis ML ensemble from {p.name}."
            return (
                False,
                "Unrecognized joblib format. Re-export 'mbti_predictor.joblib' using the updated Cell 16 of MBTI_prediction_improved.ipynb or train.py.",
            )
        except Exception as exc:
            self.load_error = str(exc)
            return False, f"Failed to load model artifact: {exc}"

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "mode_label": (
                "Trained 4-Axis ML Ensemble"
                if self.mode == "ml_ensemble"
                else "Linguistic Estimator (Built-in)"
            ),
            "model_file": self.model_path.name if self.model_path else None,
            "remove_mbti_words": (
                bool(self.bundle.get("remove_mbti_words", True))
                if self.bundle
                else True
            ),
            "load_error": self.load_error,
        }

    def _predict_axes_ml(self, combined_text: str, stylo_vec: List[float]) -> Dict[str, float]:
        b = self.bundle
        assert b is not None and sparse is not None

        cleaned = clean_text_single(
            combined_text, remove_mbti_words=b.get("remove_mbti_words", True)
        )
        X_word = b["word_tfidf"].transform([cleaned])
        X_char = b["char_tfidf"].transform([cleaned])

        X_stylo_raw = np.array([stylo_vec], dtype=np.float32)
        X_stylo_scaled = b["stylo_scaler"].transform(X_stylo_raw)
        X_stylo_sparse = sparse.csr_matrix(b["stylo_norm"].transform(X_stylo_scaled) * 0.35)

        X_tfidf = sparse.hstack([X_word, X_char], format="csr")
        X_svd = b["svd"].transform(X_tfidf).astype(np.float32)

        X_sp = sparse.hstack([X_word, X_char, X_stylo_sparse], format="csr")
        X_dn = np.hstack([X_svd, X_stylo_scaled]).astype(np.float32)

        aligned: Dict[str, float] = {}
        for col, _, _, _ in DIMENSIONS:
            ab = b["axis_bundles"][col]
            m = ab["models"]
            w = ab["weights"]
            p_vec = np.array([
                m["lr"].predict_proba(X_sp)[0, 1],
                m["svc"].predict_proba(X_sp)[0, 1],
                m["sgd"].predict_proba(X_sp)[0, 1],
                m["gbdt"].predict_proba(X_dn)[0, 1],
            ])
            p_raw = float(np.dot(p_vec, w))
            aligned[col] = float(align_proba_to_threshold(p_raw, float(ab["threshold"])))
        return aligned

    def _predict_axes_heuristic(self, combined_text: str, stylo_vec: List[float]) -> Dict[str, float]:
        """Calibrated psycholinguistic estimator used before a trained joblib bundle is loaded."""
        cleaned = clean_text_single(combined_text, remove_mbti_words=True)
        words = cleaned.split()
        n = max(len(words), 1)

        def rate(lexicon: set[str]) -> float:
            return sum(1 for w in words if w in lexicon) / n * 100.0

        r_intro = rate(INTROVERSION_WORDS)
        r_extra = rate(EXTRAVERSION_WORDS)
        r_intuit = rate(INTUITION_WORDS)
        r_sens = rate(SENSING_WORDS)
        r_think = rate(ANALYTICAL_WORDS)
        r_feel = rate(FEELING_WORDS)
        r_judge = rate(PLANNING_WORDS)
        r_perc = rate(SPONTANEOUS_WORDS)

        exclam_rate = stylo_vec[0]
        allcaps_rate = stylo_vec[3]
        emoticon_rate = stylo_vec[4]
        avg_word_len = stylo_vec[6]
        mattr = stylo_vec[8]
        first_sg = stylo_vec[9]
        first_pl = stylo_vec[10]
        second_p = stylo_vec[11]
        ana_vs_feel = stylo_vec[14]
        plan_vs_spon = stylo_vec[15]

        # Damp confidence smoothly when the text is short
        length_factor = min(1.0, math.sqrt(n / 120.0))

        # I vs E: introversion vocabulary + longer words vs social pronouns, exclamations, extraversion cues
        ie_logit = (
            0.55 * (r_intro - r_extra)
            - 0.22 * exclam_rate
            - 0.14 * emoticon_rate
            - 0.10 * allcaps_rate
            + 0.06 * (first_sg - (first_pl + second_p))
            + 0.18 * (avg_word_len - 4.55)
        ) * length_factor

        # N vs S: abstract/conceptual lexicon + lexical diversity & word length vs concrete/practical lexicon
        ns_logit = (
            0.60 * (r_intuit - r_sens)
            + 0.45 * (avg_word_len - 4.45)
            + 1.20 * (mattr - 0.74)
        ) * length_factor

        # T vs F: analytical vs affective lexicon & punctuation warmth
        tf_logit = (
            0.62 * (r_think - r_feel)
            + 1.35 * ana_vs_feel
            - 0.16 * exclam_rate
            - 0.14 * emoticon_rate
        ) * length_factor

        # J vs P: planning/structure vs spontaneity/flexibility
        jp_logit = (
            0.65 * (r_judge - r_perc)
            + 1.45 * plan_vs_spon
            - 0.08 * stylo_vec[2]  # ellipses rate
        ) * length_factor

        return {
            "IE": max(0.06, min(0.94, _sigmoid(ie_logit))),
            "NS": max(0.06, min(0.94, _sigmoid(ns_logit))),
            "TF": max(0.06, min(0.94, _sigmoid(tf_logit))),
            "JP": max(0.06, min(0.94, _sigmoid(jp_logit))),
        }

    def predict(self, text_or_posts: Union[str, Sequence[str]]) -> Dict[str, Any]:
        if isinstance(text_or_posts, (list, tuple)):
            combined_text = " ||| ".join(str(p).strip() for p in text_or_posts if str(p).strip())
        else:
            combined_text = str(text_or_posts).strip()

        stylo_vec, readable_signals = extract_stylometry_single(combined_text)
        word_count = int(readable_signals["word_count"])

        if self.mode == "ml_ensemble" and self.bundle is not None:
            aligned_probs = self._predict_axes_ml(combined_text, stylo_vec)
        else:
            aligned_probs = self._predict_axes_heuristic(combined_text, stylo_vec)

        axes_detail: List[Dict[str, Any]] = []
        for col, pos_l, neg_l, desc in DIMENSIONS:
            p_pos = float(aligned_probs[col])
            p_neg = 1.0 - p_pos
            predicted_letter = pos_l if p_pos >= 0.5 else neg_l
            confidence = max(p_pos, p_neg)
            axes_detail.append({
                "key": col,
                "label": desc,
                "pos_letter": pos_l,
                "neg_letter": neg_l,
                "pos_name": AXIS_LABELS[col]["pos_name"],
                "neg_name": AXIS_LABELS[col]["neg_name"],
                "pos_prob": round(p_pos, 4),
                "neg_prob": round(p_neg, 4),
                "predicted_letter": predicted_letter,
                "predicted_name": (
                    AXIS_LABELS[col]["pos_name"]
                    if predicted_letter == pos_l
                    else AXIS_LABELS[col]["neg_name"]
                ),
                "confidence": round(confidence, 4),
            })

        p_i, p_n, p_t, p_j = (
            aligned_probs["IE"],
            aligned_probs["NS"],
            aligned_probs["TF"],
            aligned_probs["JP"],
        )

        type_scores: List[Tuple[str, float]] = []
        for (l0, b0), (l1, b1), (l2, b2), (l3, b3) in self.all_16_combos:
            prob = (
                (p_i if b0 else 1.0 - p_i)
                * (p_n if b1 else 1.0 - p_n)
                * (p_t if b2 else 1.0 - p_t)
                * (p_j if b3 else 1.0 - p_j)
            )
            type_scores.append((l0 + l1 + l2 + l3, prob))
        type_scores.sort(key=lambda item: item[1], reverse=True)

        predicted_type = type_scores[0][0]
        if word_count >= 200:
            reliability_tier = "high"
            reliability_label = "High signal (200+ words)"
        elif word_count >= 80:
            reliability_tier = "moderate"
            reliability_label = "Moderate signal (80-199 words)"
        else:
            reliability_tier = "low"
            reliability_label = "Low signal (under 80 words; add more writing for accuracy)"

        return {
            "predicted_type": predicted_type,
            "profile": MBTI_PROFILES[predicted_type],
            "word_count": word_count,
            "reliability_tier": reliability_tier,
            "reliability_label": reliability_label,
            "axes": axes_detail,
            "top_3_types": [
                {
                    "type": t,
                    "title": MBTI_PROFILES[t]["title"],
                    "probability": round(float(prob), 4),
                }
                for t, prob in type_scores[:3]
            ],
            "signals": readable_signals,
            "engine": self.get_status(),
        }
