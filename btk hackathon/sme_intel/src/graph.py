"""
graph.py — LangGraph orchestrator for SME-Intel.

Wires the three agent nodes into a linear pipeline:

    START
      |
      v
  [data_analyzer]  ← DataAnalyzer_Agent: detect churn & margin risks
      |
      v
  [strategy]       ← Strategy_Agent: formulate concrete B2B actions
      |
      v
  [action]         ← Action_Agent: draft Turkish outreach messages
      |
      v
    END

The graph is compiled once at module load time (``_COMPILED_GRAPH``) and
reused across Streamlit re-runs for efficiency.  The public entry-point is
``run_workflow()``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.agents import action_node, data_analyzer_node, strategy_node
from src.state import GraphState

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────── #
#  Node name constants (single source of truth — avoids magic strings)         #
# ──────────────────────────────────────────────────────────────────────────── #

NODE_ANALYZER = "data_analyzer"
NODE_STRATEGY = "strategy"
NODE_ACTION = "action"


# ──────────────────────────────────────────────────────────────────────────── #
#  Graph builder                                                                #
# ──────────────────────────────────────────────────────────────────────────── #


def _build_graph() -> Any:
    """
    Construct and compile the SME-Intel LangGraph StateGraph.

    This function is called once at module import.  It:
    1. Creates a ``StateGraph`` typed to ``GraphState``.
    2. Registers the three agent nodes.
    3. Wires the linear edge chain.
    4. Compiles and returns the runnable graph object.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to ``.invoke()``.
    """
    builder: StateGraph = StateGraph(GraphState)

    # ── Register nodes ───────────────────────────────────────────────── #
    builder.add_node(NODE_ANALYZER, data_analyzer_node)
    builder.add_node(NODE_STRATEGY, strategy_node)
    builder.add_node(NODE_ACTION, action_node)

    # ── Define edges (linear pipeline) ──────────────────────────────── #
    builder.add_edge(START, NODE_ANALYZER)
    builder.add_edge(NODE_ANALYZER, NODE_STRATEGY)
    builder.add_edge(NODE_STRATEGY, NODE_ACTION)
    builder.add_edge(NODE_ACTION, END)

    compiled = builder.compile()
    logger.info("SME-Intel LangGraph compiled successfully.")
    return compiled


# Module-level singleton — compiled once, reused for every workflow run.
_COMPILED_GRAPH: Any = _build_graph()


# ──────────────────────────────────────────────────────────────────────────── #
#  Public API                                                                   #
# ──────────────────────────────────────────────────────────────────────────── #


def run_workflow(masked_df: pd.DataFrame) -> GraphState:
    """
    Execute the full SME-Intel agentic workflow on the provided masked DataFrame.

    This is the **single entry-point** for the Streamlit UI (Step 3).
    It initialises a fresh ``GraphState``, invokes the compiled graph, and
    returns the final state containing analysis results, strategies, and
    drafted messages — all using masked customer IDs only.

    Args:
        masked_df: The anonymised pandas DataFrame produced by
                   ``data_loader.load_and_anonymise()``.
                   MUST already have customer names replaced with tokens.

    Returns:
        The final ``GraphState`` after all three nodes have executed.
        Key fields of interest for the caller:
        - ``state["analysis_results"]``: List of ``RiskFinding`` objects.
        - ``state["proposed_strategy"]``: List of ``StrategyRecommendation`` objects.
        - ``state["draft_messages"]``: List of ``DraftMessage`` objects.
        - ``state["error_log"]``: Any errors accumulated during the run.

    Raises:
        ValueError: If *masked_df* is empty or not a DataFrame.
        RuntimeError: If the graph fails catastrophically (non-recoverable).

    Example::

        from src.security import DataMasker
        from src.data_loader import load_and_anonymise
        from src.graph import run_workflow

        masker = DataMasker()
        df = load_and_anonymise("data/raw/sales_data.xlsx", masker)
        final_state = run_workflow(df)

        for msg in final_state["draft_messages"]:
            # Unmask before showing to the user
            real_body = masker.unmask_text(msg.body)
            print(real_body)
    """
    # ── Input validation ────────────────────────────────────────────── #
    if not isinstance(masked_df, pd.DataFrame):
        raise ValueError(
            f"run_workflow expects a pandas DataFrame, got {type(masked_df).__name__}."
        )
    if masked_df.empty:
        raise ValueError("run_workflow received an empty DataFrame. Cannot proceed.")

    logger.info(
        "Starting SME-Intel workflow. Rows: %d, Customers: %d.",
        len(masked_df),
        masked_df["Customer Name"].nunique(),
    )

    # ── Build initial state ─────────────────────────────────────────── #
    initial_state: GraphState = {
        "masked_dataframe": masked_df,
        "analysis_results": [],
        "proposed_strategy": [],
        "draft_messages": [],
        "error_log": [],
        "analysis_summary": None,
        "strategy_summary": None,
    }

    # ── Invoke the graph ────────────────────────────────────────────── #
    try:
        final_state: GraphState = _COMPILED_GRAPH.invoke(initial_state)
    except Exception as exc:
        logger.exception("Graph execution failed with an unrecoverable error.")
        raise RuntimeError(
            f"SME-Intel workflow failed during graph execution: {exc}"
        ) from exc

    # ── Log summary ─────────────────────────────────────────────────── #
    error_count = len(final_state.get("error_log", []))
    if error_count:
        logger.warning(
            "Workflow completed with %d error(s): %s",
            error_count,
            final_state["error_log"],
        )
    else:
        logger.info(
            "Workflow completed successfully. "
            "Findings: %d | Strategies: %d | Messages: %d.",
            len(final_state.get("analysis_results", [])),
            len(final_state.get("proposed_strategy", [])),
            len(final_state.get("draft_messages", [])),
        )

    return final_state


def get_graph_png() -> bytes | None:
    """
    Render the compiled graph as a PNG image for debugging / UI display.

    Returns:
        Raw PNG bytes if Mermaid/graphviz is available, else ``None``.
    """
    try:
        return _COMPILED_GRAPH.get_graph().draw_mermaid_png()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render graph image: %s", exc)
        return None
