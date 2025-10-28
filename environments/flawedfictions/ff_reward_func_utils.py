"""
Reward function utils for FlawedFictions.

High-level structure:
- Stage 1 (binary): predicting whether there is a plot-hole in the story
  - binary_format_reward: 1.0 if <response> and <decision> exist (or \\boxed{yes|no}); else 0.0
  - binary_accuracy_reward: 1.0 if parsed decision equals gold (0/1); else 0.0
- Stage 2 (localization): predicting the specific lines that contain the plot-hole and the contradicted lines
  - localization_format_reward: 1.0 if <error_lines> and <contradicted_lines> exist; else 0.0
  - localization_full_reward: 1.0 if overlap holds for both error sets and contradicted sets; else 0.0
"""

import re
from typing import Any, Dict, List, Tuple

from verifiers.parsers.parser import Parser
from verifiers.types import Messages
from verifiers.utils.data_utils import extract_boxed_answer


#########################################################################
# Text-handling functions
#########################################################################


def _strip_bracketed(text: str) -> str:
    return re.sub(r"\[.*?\]", "", text)


def _remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _preprocess_sent(text: str) -> str:
    # Matches original preprocessing closely
    text = text.lower()
    text = text.replace("•", "")
    text = _strip_bracketed(text)
    text = _remove_html_tags(text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> List[str]:
    # Lightweight splitter for this dataset
    text = _remove_html_tags(_strip_bracketed(text))
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p]


def precompute_story_sentences(story: str) -> Tuple[List[str], List[str]]:
    """Return (raw_sentences, processed_sentences) for a story.

    - raw_sentences: human-readable sentences with markup removed
    - processed_sentences: normalized with _preprocess_sent for matching
    """
    assert isinstance(story, str), "story must be a string"
    raw_sentences = _split_sentences(story)
    processed_sentences = [_preprocess_sent(s) for s in raw_sentences]
    return raw_sentences, processed_sentences


#########################################################################
# Dataset line utilities - converts a string containing <li>...</li>
# or newline-separated lines into a list of strings. Logic taken
# from the original paper's scripts.
#########################################################################


def listify_lines(lines: str) -> List[str]:
    """Normalize a single string field of lines into a list[str].

    Expected input is a string that either contains <li>…</li> items or
    newline-separated lines. We intentionally do not accept lists to keep
    types explicit and predictable.
    """
    assert isinstance(lines, str), "lines must be a string"
    s = lines.strip()
    if not s:
        return []
    if "<li>" in s:
        parts = s.split("</li>")
        parts = [p.strip() for p in parts if p.strip()]
        parts = [p.replace("-", "").replace("*", "").strip() for p in parts]
        return [p for p in parts if p]
    parts = s.split("\n")
    parts = [p.strip() for p in parts]
    parts = [p.replace("-", "").replace("*", "").strip() for p in parts]
    return [p for p in parts if p]


#########################################################################
# Original paper's parsing functions
#########################################################################


def _extract_tag_lines(text: str, tag: str) -> List[str]:
    """Extract and lightly clean the content inside <tag>…</tag> lines."""
    open_tag, close_tag = f"<{tag}>", f"</{tag}>"
    if open_tag not in text or close_tag not in text:
        return []
    inner = text.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()
    parts = inner.split("\n")
    parts = [p.strip().replace("-", "").replace("*", "").strip() for p in parts]
    return [p for p in parts if p]


def extract_tag_lines(text: str, tag: str) -> List[str]:
    """Public helper: extract cleaned lines inside <tag>…</tag>."""
    return _extract_tag_lines(text, tag)


def _parse_first_decision(text: str) -> int | None:
    """Return 0/1 from FIRST <decision>…</decision>; None if missing/malformed."""
    if "<response>" not in text or "</response>" not in text:
        return None
    if "<decision>" not in text or "</decision>" not in text:
        return None
    inner = text.split("<decision>", 1)[1].split("</decision>", 1)[0].strip()
    return 0 if "no continuity error found" in inner.lower() else 1


def parse_paper_binary_label(text: str) -> int | None:
    """Parse a 0/1 label from paper-style <decision>...</decision>."""
    return _parse_first_decision(text)


def is_paper_binary_formatted(text: str) -> bool:
    return (
        "<response>" in text
        and "</response>" in text
        and "<decision>" in text
        and "</decision>" in text
    )


def _line_tags_ok(text: str) -> bool:
    return (
        "<error_lines>" in text
        and "</error_lines>" in text
        and "<contradicted_lines>" in text
        and "</contradicted_lines>" in text
    )


def map_candidates_to_sentence_indices(
    candidates: List[str], story_sents_proc: List[str]
) -> set[int]:
    """Map candidate snippets to indices of matching story sentences.

    - Normalizes each candidate with _preprocess_sent and checks substring
      containment in every processed sentence. All matches are collected.
    """
    matches: set[int] = set()
    for cand in candidates:
        q = _preprocess_sent(cand)
        if not q:
            continue
        for idx, proc in enumerate(story_sents_proc):
            if q in proc:
                matches.add(idx)
    return matches


#########################################################################
# Boxed (simple) parsing helpers
#########################################################################


def parse_boxed_binary_label(text: str) -> int | None:
    """Parse 0/1 from a \\boxed{yes|no} segment."""
    val = extract_boxed_answer(text)
    val = (val or text or "").strip().lower()
    if val == "yes":
        return 1
    if val == "no":
        return 0
    return None


def is_boxed_binary_formatted(text: str) -> bool:
    return parse_boxed_binary_label(text) is not None


#########################################################################
# Reward function factories
#########################################################################


def make_localization_rewards(parse_binary_label_fn, is_binary_formatted_fn):
    """Factory to create (format_reward, full_reward) for localization task.

    Because the localization task requires on formatting from the
    binary task (which depends on prompt style), we use this factory so we can reuse the logic.

    parse_binary_label_fn: (text) -> int | None; whet
    is_binary_formatted_fn: (text) -> bool
    """

    def loc_format(
        parser: Parser, completion: Messages, answer: Dict[str, Any], **kwargs
    ) -> float:
        text = parser.parse_answer(completion) or ""
        if not is_binary_formatted_fn(text):
            return 0.0
        pred = parse_binary_label_fn(text)
        if pred is None:
            return 0.0
        if pred == 0:
            return 1.0
        return 1.0 if _line_tags_ok(text) else 0.0

    def loc_full(
        parser: Parser, completion: Messages, answer: Dict[str, Any], **kwargs
    ) -> float:
        text = parser.parse_answer(completion) or ""
        if not is_binary_formatted_fn(text):
            return 0.0
        pred = parse_binary_label_fn(text)
        if pred is None:
            return 0.0
        gold = int(answer["cont_error"])  # 0/1
        if pred != gold:
            return 0.0
        if pred == 0:
            return 1.0
        if not _line_tags_ok(text):
            return 0.0
        s_proc = answer.get("story_sents_proc")
        if not isinstance(s_proc, list):
            return 0.0
        gt_err = answer.get("cont_error_lines", [])
        gt_contra = answer.get("contradicted_lines", [])
        if not isinstance(gt_err, list) or not isinstance(gt_contra, list):
            return 0.0
        pred_err = extract_tag_lines(text, "error_lines")
        pred_contra = extract_tag_lines(text, "contradicted_lines")
        pred_err_idx = map_candidates_to_sentence_indices(pred_err, s_proc)
        pred_contra_idx = map_candidates_to_sentence_indices(pred_contra, s_proc)
        gt_err_idx = map_candidates_to_sentence_indices(gt_err, s_proc)
        gt_contra_idx = map_candidates_to_sentence_indices(gt_contra, s_proc)
        err_ok = bool(pred_err_idx.intersection(gt_err_idx))
        contra_ok = bool(pred_contra_idx.intersection(gt_contra_idx))
        return 1.0 if (err_ok and contra_ok) else 0.0

    return loc_format, loc_full


def make_binary_rewards(parse_binary_label_fn, is_binary_formatted_fn):
    """Factory to create (binary_format_reward, binary_accuracy_reward).

    parse_label_from_text: (text) -> 0/1 | None
    binary_format_ok_from_text: (text) -> bool
    """

    def bin_format(parser: Parser, completion: Messages, **kwargs) -> float:
        text = parser.parse_answer(completion) or ""
        return 1.0 if is_binary_formatted_fn(text) else 0.0

    def bin_accuracy(
        parser: Parser, completion: Messages, answer: Dict[str, Any], **kwargs
    ) -> float:
        text = parser.parse_answer(completion) or ""
        pred = parse_binary_label_fn(text)
        if pred is None:
            return 0.0
        gold = int(answer["cont_error"])  # 0/1
        return 1.0 if pred == gold else 0.0

    return bin_format, bin_accuracy
