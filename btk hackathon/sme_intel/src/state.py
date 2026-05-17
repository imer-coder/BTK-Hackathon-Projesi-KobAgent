"""
state.py — LangGraph graph state and shared data models for SME-Intel.

All Pydantic models used as structured LLM output schemas are defined here
so that agents.py and graph.py share one canonical type hierarchy.

Design notes:
- ``GraphState`` uses TypedDict (LangGraph's native contract).
- List fields use ``Annotated[list, operator.add]`` so LangGraph can
  MERGE updates from parallel branches instead of overwriting them.
- The dataframe is stored as ``Any`` (pd.DataFrame at runtime) because
  LangGraph operates in-memory for this MVP; no checkpointing is used.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, List, Literal, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────── #
#  Structured output models (used as LLM response schemas)                     #
# ──────────────────────────────────────────────────────────────────────────── #


class RiskFinding(BaseModel):
    """
    A single risk signal detected by the DataAnalyzer_Agent.

    Attributes:
        customer_id: Masked token (e.g. ``"CUSTOMER_1"``). NEVER a real name.
        risk_type: Category of the detected risk.
        severity: Qualitative severity level for prioritisation.
        evidence: Factual, data-driven explanation referencing only masked IDs.
        metric_value: The numeric value that triggered this finding
                      (e.g. margin ratio, order-drop percentage).
    """

    customer_id: str = Field(
        description=(
            "Masked customer token such as 'CUSTOMER_1'. "
            "Must never contain real business names."
        )
    )
    risk_type: Literal["CHURN_RISK", "MARGIN_RISK", "BOTH"] = Field(
        description="Type of risk detected for this customer."
    )
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Qualitative severity for prioritisation."
    )
    evidence: str = Field(
        description=(
            "Concise, data-driven explanation of why this risk was flagged. "
            "Reference only masked IDs. No real names allowed."
        )
    )
    metric_value: float = Field(
        description=(
            "Key numeric metric that triggered the alert "
            "(e.g. margin_ratio=0.18, order_drop_pct=42.5)."
        )
    )


class AnalysisReport(BaseModel):
    """Wrapper returned by DataAnalyzer_Agent containing all findings."""

    findings: List[RiskFinding] = Field(
        default_factory=list,
        description="List of all risk findings. Empty list if no risks detected.",
    )
    summary: str = Field(
        description=(
            "One-paragraph executive summary of the analysis. "
            "Use only masked customer IDs."
        )
    )


class StrategyRecommendation(BaseModel):
    """
    A concrete, actionable B2B strategy for a single at-risk customer.

    Attributes:
        customer_id: Masked token this strategy targets.
        risk_type: The risk type this strategy addresses.
        strategy_title: Short title (e.g. ``"Shorten payment terms to Net 30"``).
        rationale: Why this strategy counters the specific risk.
        action_steps: Ordered list of concrete steps the sales rep must take.
        priority: Execution priority relative to other recommendations.
    """

    customer_id: str = Field(
        description="Masked customer token. Must never contain real names."
    )
    risk_type: Literal["CHURN_RISK", "MARGIN_RISK", "BOTH"] = Field(
        description="The risk type this strategy addresses."
    )
    strategy_title: str = Field(
        description="Short, descriptive title for the strategy (max 10 words)."
    )
    rationale: str = Field(
        description=(
            "Why this specific strategy counters the identified risk. "
            "Reference evidence from the analysis. Use only masked IDs."
        )
    )
    action_steps: List[str] = Field(
        description=(
            "Ordered, concrete steps the sales representative must execute. "
            "Each step must be actionable and measurable."
        )
    )
    priority: Literal["LOW", "MEDIUM", "HIGH", "URGENT"] = Field(
        description="Execution priority relative to other recommendations."
    )


class StrategyReport(BaseModel):
    """Wrapper returned by Strategy_Agent containing all strategy recommendations."""

    recommendations: List[StrategyRecommendation] = Field(
        default_factory=list,
        description="One recommendation per at-risk customer.",
    )
    strategic_summary: str = Field(
        description=(
            "One-paragraph strategic overview for the business owner. "
            "Use only masked customer IDs."
        )
    )


class DraftMessage(BaseModel):
    """
    A professionally drafted outreach message for an at-risk customer.

    Attributes:
        customer_id: Masked token identifying the target. NEVER real name.
        channel: Communication channel this message is optimised for.
        subject: Email subject line (leave empty for WhatsApp drafts).
        body: Full message body in Turkish. Must use masked customer ID
              as a placeholder that the user will personalise before sending.
        tone: Detected tone for the user's review.
    """

    customer_id: str = Field(
        description=(
            "Masked customer token (e.g. 'CUSTOMER_2'). "
            "Must never contain real business names."
        )
    )
    channel: Literal["EMAIL", "WHATSAPP"] = Field(
        description="Communication channel this draft is optimised for."
    )
    subject: str = Field(
        default="",
        description="Email subject line. Leave empty for WhatsApp.",
    )
    body: str = Field(
        description=(
            "Full message body written in professional Turkish. "
            "Where the real customer name would appear, use the masked ID "
            "as a placeholder (e.g. '[CUSTOMER_2 yöneticisi]'). "
            "The business owner will personalise it before sending."
        )
    )
    tone: Literal["FORMAL", "FRIENDLY_PROFESSIONAL", "URGENT"] = Field(
        description="Tone of the drafted message."
    )


class MessageBundle(BaseModel):
    """Wrapper returned by Action_Agent containing all drafted messages."""

    messages: List[DraftMessage] = Field(
        default_factory=list,
        description="One or two draft messages per customer (Email + WhatsApp).",
    )


# ──────────────────────────────────────────────────────────────────────────── #
#  LangGraph State                                                              #
# ──────────────────────────────────────────────────────────────────────────── #


class GraphState(TypedDict):
    """
    Shared mutable state passed between every node in the SME-Intel graph.

    Fields with ``Annotated[list, operator.add]`` are *accumulated*:
    each node appends to them rather than overwriting. Plain fields are
    overwritten on each update.

    Attributes:
        masked_dataframe:   The anonymised pandas DataFrame produced by
                            data_loader.py.  Held in memory; never serialised.
        analysis_results:   Accumulated list of RiskFinding objects from
                            DataAnalyzer_Agent.
        proposed_strategy:  Accumulated list of StrategyRecommendation objects
                            from Strategy_Agent.
        draft_messages:     Accumulated list of DraftMessage objects from
                            Action_Agent.
        analysis_summary:   Free-text executive summary from the analyzer.
        strategy_summary:   Free-text strategic overview from the strategist.
        error_log:          Fault-tolerance log: any node may append an error
                            message here instead of raising, keeping the graph
                            running.
    """

    # ── Input ───────────────────────────────────────────────────────────── #
    masked_dataframe: Any  # pd.DataFrame; Any keeps TypedDict serialisable

    # ── Accumulated outputs (additive merge across nodes) ───────────────── #
    analysis_results: Annotated[List[RiskFinding], operator.add]
    proposed_strategy: Annotated[List[StrategyRecommendation], operator.add]
    draft_messages: Annotated[List[DraftMessage], operator.add]
    error_log: Annotated[List[str], operator.add]

    # ── Scalar summaries (last-write-wins) ──────────────────────────────── #
    analysis_summary: Optional[str]
    strategy_summary: Optional[str]
