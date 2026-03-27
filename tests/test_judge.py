"""
Tests for the judge service.

Covers:
- JudgeUnavailableError raised on no API key (no fallback)
- Empty candidates returns empty JudgeResult (not an error)
- JudgeResult helper methods (exact_matches, close_alternatives, accepted, rejected)
- JSON parsing: valid JSON, invalid JSON fallback, markdown stripping, unknown ID filtering
"""

import pytest
from app.services.retriever import RetrievalResult
from app.services.judge import (
    judge_text_candidates,
    judge_image_candidates,
    JudgeResult,
    JudgeUnavailableError,
    _parse_judge_json,
    CandidateJudgment,
)


def _make_product(
    pid: int = 1,
    category: str = "shirt",
    color: str = "white",
    price: float = 50.0,
    tfidf: float = 0.30,
    description: str = "",
) -> RetrievalResult:
    return RetrievalResult(
        id=pid,
        product_name=f"Product {pid}",
        brand="TestBrand",
        category=category,
        base_color=color,
        usage="Casual",
        season="Summer",
        gender="Men",
        gender_normalized="Men",
        price=price,
        image_path=f"images/{pid}.jpg",
        description=description,
        tfidf_score=tfidf,
    )


# ── No-API-key raises JudgeUnavailableError ────────────────────────────────────

def test_judge_text_no_api_key_raises():
    products = [_make_product(pid=i) for i in range(3)]
    with pytest.raises(JudgeUnavailableError):
        judge_text_candidates(
            user_ask="white shirt",
            search_state_summary="shirt, white color",
            candidates=products,
            api_key="",
        )


def test_judge_image_no_api_key_raises():
    products = [_make_product(pid=i) for i in range(3)]
    with pytest.raises(JudgeUnavailableError):
        judge_image_candidates(
            user_text="",
            query_image_b64="fake_b64",
            candidates=products,
            api_key="",
        )


# ── Empty candidates returns empty JudgeResult (no error) ─────────────────────

def test_judge_text_empty_candidates_returns_empty_result():
    result = judge_text_candidates("white shirt", "shirt, white", [], api_key="")
    assert isinstance(result, JudgeResult)
    assert result.judgments == []
    assert result.overall_summary != ""


def test_judge_image_empty_candidates_returns_empty_result():
    result = judge_image_candidates(
        user_text="",
        query_image_b64="fake_b64",
        candidates=[],
        api_key="",
    )
    assert isinstance(result, JudgeResult)
    assert result.judgments == []


# ── JudgeResult helper methods ─────────────────────────────────────────────────

def test_judge_result_helpers():
    result = JudgeResult(
        judgments=[
            CandidateJudgment(1, "exact_match", "fits"),
            CandidateJudgment(2, "close_alternative", "similar"),
            CandidateJudgment(3, "reject", "wrong category"),
        ]
    )
    assert len(result.exact_matches()) == 1
    assert len(result.close_alternatives()) == 1
    assert len(result.accepted()) == 2
    assert len(result.rejected()) == 1


# ── JSON parsing ───────────────────────────────────────────────────────────────

def test_parse_judge_json_valid():
    candidates = [_make_product(pid=1), _make_product(pid=2)]
    raw = (
        '{"judgments": ['
        '{"product_id": 1, "verdict": "exact_match", "reason": "perfect"}, '
        '{"product_id": 2, "verdict": "reject", "reason": "wrong color"}'
        '], "overall_summary": "one match"}'
    )
    result = _parse_judge_json(raw, candidates)
    assert len(result.judgments) == 2
    assert result.judgments[0].verdict == "exact_match"
    assert result.judgments[1].verdict == "reject"
    assert result.overall_summary == "one match"


def test_parse_judge_json_invalid_falls_back():
    candidates = [_make_product(pid=1), _make_product(pid=2)]
    result = _parse_judge_json("not valid json {{{{", candidates)
    # Fallback: all close_alternative
    assert len(result.judgments) == 2
    assert all(j.verdict == "close_alternative" for j in result.judgments)


def test_parse_judge_json_strips_markdown():
    candidates = [_make_product(pid=1)]
    raw = (
        '```json\n'
        '{"judgments": [{"product_id": 1, "verdict": "close_alternative", "reason": "ok"}], '
        '"overall_summary": "found one"}\n```'
    )
    result = _parse_judge_json(raw, candidates)
    assert result.judgments[0].verdict == "close_alternative"


def test_parse_judge_json_ignores_unknown_ids():
    candidates = [_make_product(pid=1)]
    raw = '{"judgments": [{"product_id": 999, "verdict": "exact_match", "reason": "x"}], "overall_summary": ""}'
    result = _parse_judge_json(raw, candidates)
    # ID 999 not in candidates → ignored
    assert result.judgments == []
