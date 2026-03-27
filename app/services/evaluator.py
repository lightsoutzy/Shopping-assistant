"""
Lightweight guardrail check on recommended products.
Rule-based only; no LLM.
"""

from dataclasses import dataclass, field

from app.services.query_parser import ParsedQuery
from app.services.retriever import RetrievalResult


@dataclass
class EvalResult:
    passed: bool
    violations: list[str] = field(default_factory=list)


def evaluate(
    products: list[RetrievalResult],
    constraints: ParsedQuery,
    valid_ids: set[int],
) -> EvalResult:
    """
    Check that:
    1. All returned product IDs exist in the catalog.
    2. Category constraint is not obviously violated (if specified).
    3. Price constraint is not violated (if specified).
    """
    violations: list[str] = []

    for p in products:
        if p.id not in valid_ids:
            violations.append(f"Product id {p.id} not found in catalog.")

        if constraints.category and p.category != constraints.category:
            violations.append(
                f"Product '{p.product_name}' has category '{p.category}', "
                f"expected '{constraints.category}'."
            )

        if constraints.max_price and p.price > constraints.max_price:
            violations.append(
                f"Product '{p.product_name}' costs ${p.price:.2f}, "
                f"exceeds max ${constraints.max_price:.2f}."
            )

    return EvalResult(passed=len(violations) == 0, violations=violations)
