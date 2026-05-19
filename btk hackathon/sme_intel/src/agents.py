import os
import logging
from typing import Dict, List

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from config.settings import settings
from src.state import (
    AnalysisReport,
    DraftMessage,
    GraphState,
    MessageBundle,
    RiskFinding,
    StrategyRecommendation,
    StrategyReport,
)

logger = logging.getLogger(__name__)

# --- ÇEVRE DEĞİŞKENLERİNİ YÜKLE VE GEMINI MODELİNİ BAŞLAT ---
# .env dosyasını yükle (uygulama genelinde API anahtarının görünmesini sağlar)
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
    raise ValueError(
        "Kritik Hata: GOOGLE_API_KEY bulunamadı! Lütfen uygulamanın kök dizinindeki "
        ".env dosyasına geçerli bir API anahtarı eklediğinizden emin olun."
    )

# 404 hatasını önlemek için "gemini-1.5-flash-latest" kullanıyoruz ve API key'i settings'den açıkça veriyoruz.
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest", 
    temperature=0,
    google_api_key=settings.GOOGLE_API_KEY.get_secret_value()
)


def _df_to_prompt_context(df: pd.DataFrame) -> str:
    lines: List[str] = []
    for customer_id, group in df.groupby("Customer Name"):
        group_sorted = group.sort_values("transaction_date")
        lines.append(f"\n=== {customer_id} ===")
        for _, row in group_sorted.iterrows():
            margin_pct = (
                (row["unit_sales_price"] - row["unit_cost"]) / row["unit_sales_price"]
            ) * 100
            lines.append(
                f"  {row['transaction_date']} | Category: {row['category']} | "
                f"Qty: {row['quantity']} | "
                f"Price: {row['unit_sales_price']:.2f} TRY | "
                f"Cost: {row['unit_cost']:.2f} TRY | "
                f"Margin: {margin_pct:.1f}%"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node 1 — DataAnalyzer_Agent 
# ---------------------------------------------------------------------------

_ANALYZER_SYSTEM = """\
You are DataAnalyzer_Agent, a senior B2B sales data analyst for an SME consulting firm.

SECURITY RULE — CRITICAL:
- The customer names in the data have already been anonymised to tokens like CUSTOMER_1.
- You MUST use ONLY these tokens in every field of your output. NEVER invent or guess real names.

LANGUAGE RULE — CRITICAL:
- You MUST output the analysis and executive summary STRICTLY in Turkish.
- No English sentences or words should appear in the final JSON output (except keys defined by the schema).

YOUR TASK:
Analyse the masked B2B sales data below and identify customers at risk.

DETECTION CRITERIA:
1. CHURN_RISK   → Order quantity declining ≥ 30 % across the last two consecutive periods,
                  OR a sudden single-period drop ≥ 50 %.
2. MARGIN_RISK  → Current gross margin < 15 %, OR cost growth rate outpaces price growth
                  rate by ≥ 10 percentage points over the observed period.
3. BOTH         → Customer meets criteria for both risks simultaneously.

OUTPUT FORMAT:
Return a structured AnalysisReport with:
- findings: one RiskFinding per at-risk customer (skip healthy customers).
- summary: one concise executive paragraph using only masked IDs.

Only flag genuine risks. Do not flag healthy customers.
"""

_ANALYZER_HUMAN = """\
Masked Sales Data:
{data_context}

Analyse all customers and return the AnalysisReport.
"""

def data_analyzer_node(state: GraphState) -> Dict:
    logger.info("[DataAnalyzer] Node started.")
    df: pd.DataFrame = state["masked_dataframe"]

    try:
        data_context = _df_to_prompt_context(df)

        prompt = ChatPromptTemplate.from_messages([
            ("system", _ANALYZER_SYSTEM),
            ("human", _ANALYZER_HUMAN),
        ])

        chain = prompt | llm.with_structured_output(AnalysisReport)
        report: AnalysisReport = chain.invoke({"data_context": data_context})

        return {
            "analysis_results": report.findings,
            "analysis_summary": report.summary,
            "error_log": [],
        }

    except Exception as exc:
        msg = f"[DataAnalyzer] Failed: {exc}"
        logger.exception(msg)
        return {
            "analysis_results": [],
            "analysis_summary": None,
            "error_log": [msg],
        }


# ---------------------------------------------------------------------------
# Node 2 — Strategy_Agent 
# ---------------------------------------------------------------------------

_STRATEGY_SYSTEM = """\
You are Strategy_Agent, a senior B2B account management strategist.

SECURITY RULE — CRITICAL:
- All customer identifiers are anonymised tokens (e.g. CUSTOMER_1).
- Use ONLY these tokens in every output field. NEVER use real names.

LANGUAGE RULE — CRITICAL:
- You MUST output the strategies and strategic summary STRICTLY in Turkish.
- No English sentences or words should appear in the final JSON output (except keys defined by the schema).

YOUR TASK:
For every RiskFinding provided, produce one concrete StrategyRecommendation.

STRATEGY PRINCIPLES:
- CHURN_RISK   → Focus on re-engagement: loyalty offers, dedicated account manager,
                 win-back meetings, flexible payment terms.
- MARGIN_RISK  → Focus on margin recovery: price renegotiation, cost reduction talks,
                 shifting to higher-margin product lines, tighter payment terms.
- BOTH         → Address both dimensions; prioritise margin stabilisation first.

OUTPUT FORMAT:
Return a StrategyReport with:
- recommendations: one StrategyRecommendation per finding, with 3-5 action_steps each.
- strategic_summary: one concise strategic overview paragraph using only masked IDs.
"""

_STRATEGY_HUMAN = """\
Risk Findings from DataAnalyzer_Agent:
{findings_context}

Produce the StrategyReport.
"""

def _findings_to_text(findings: List[RiskFinding]) -> str:
    if not findings:
        return "No risk findings provided."
    return "\n".join(
        f"- {f.customer_id} | {f.risk_type} | Severity: {f.severity} | "
        f"Metric: {f.metric_value:.3f} | Evidence: {f.evidence}"
        for f in findings
    )

def strategy_node(state: GraphState) -> Dict:
    logger.info("[Strategy] Node started.")
    findings: List[RiskFinding] = state.get("analysis_results", [])

    if not findings:
        return {
            "proposed_strategy": [],
            "strategy_summary": "No risks detected; no strategies required.",
            "error_log": [],
        }

    try:
        findings_context = _findings_to_text(findings)

        prompt = ChatPromptTemplate.from_messages([
            ("system", _STRATEGY_SYSTEM),
            ("human", _STRATEGY_HUMAN),
        ])

        chain = prompt | llm.with_structured_output(StrategyReport)
        report: StrategyReport = chain.invoke({"findings_context": findings_context})

        return {
            "proposed_strategy": report.recommendations,
            "strategy_summary": report.strategic_summary,
            "error_log": [],
        }

    except Exception as exc:
        msg = f"[Strategy] Failed: {exc}"
        logger.exception(msg)
        return {
            "proposed_strategy": [],
            "strategy_summary": None,
            "error_log": [msg],
        }


# ---------------------------------------------------------------------------
# Node 3 — Action_Agent 
# ---------------------------------------------------------------------------

_ACTION_SYSTEM = """\
You are Action_Agent, a professional B2B communications specialist who writes
outreach messages in fluent, formal Turkish.

SECURITY RULE — CRITICAL:
- Customer names are anonymised tokens (e.g. CUSTOMER_1).
- Where a real name would appear in the message, write the token in brackets
  as a placeholder, e.g. "[CUSTOMER_1 yöneticisi]".
- NEVER invent or use real company names.

LANGUAGE RULE — CRITICAL:
- You MUST output the draft messages STRICTLY in Turkish.
- No English sentences or words should appear in the final JSON output (except keys defined by the schema).

YOUR TASK:
For every StrategyRecommendation, draft TWO outreach messages:
1. EMAIL  — formal subject line + professional body (tone: FORMAL or FRIENDLY_PROFESSIONAL).
2. WHATSAPP — concise, warm, mobile-friendly version of the same message (no subject line).

LANGUAGE: Turkish (professional business register).
LENGTH: Email body 150-250 words. WhatsApp body 60-100 words.

OUTPUT FORMAT:
Return a MessageBundle containing all DraftMessage objects.
"""

_ACTION_HUMAN = """\
Strategy Recommendations:
{strategies_context}

Draft the complete MessageBundle in Turkish.
"""

def _strategies_to_text(strategies: List[StrategyRecommendation]) -> str:
    if not strategies:
        return "No recommendations provided."
    parts: List[str] = []
    for s in strategies:
        steps = "\n".join(f"    {i+1}. {step}" for i, step in enumerate(s.action_steps))
        parts.append(
            f"Customer: {s.customer_id}\n"
            f"Risk: {s.risk_type} | Priority: {s.priority}\n"
            f"Strategy: {s.strategy_title}\n"
            f"Rationale: {s.rationale}\n"
            f"Action Steps:\n{steps}"
        )
    return "\n\n---\n\n".join(parts)

def action_node(state: GraphState) -> Dict:
    logger.info("[Action] Node started.")
    strategies: List[StrategyRecommendation] = state.get("proposed_strategy", [])

    if not strategies:
        return {"draft_messages": [], "error_log": []}

    try:
        strategies_context = _strategies_to_text(strategies)

        prompt = ChatPromptTemplate.from_messages([
            ("system", _ACTION_SYSTEM),
            ("human", _ACTION_HUMAN),
        ])

        chain = prompt | llm.with_structured_output(MessageBundle)
        bundle: MessageBundle = chain.invoke({"strategies_context": strategies_context})

        return {"draft_messages": bundle.messages, "error_log": []}

    except Exception as exc:
        msg = f"[Action] Failed: {exc}"
        logger.exception(msg)
        return {"draft_messages": [], "error_log": [msg]}
