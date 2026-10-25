# Isolates logical graph switches and routing conditions.
from typing import Literal
from app.core.config import settings
from app.rag.state import AgentState


def edge_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"


def edge_after_kb_grade(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"


# Updated, robust implementation for your app/rag/edges.py file:
def edge_after_web_grade(
    state: AgentState,
) -> Literal["generate_from_web", "answer_insufficient"]:
    """
    Evaluates web fallback data payload states. If data exists, it routes
    straight to the text synthesizer node to generate a grounded answer.
    """
    # If our Tavily tool successfully retrieved search snippets, process them natively
    if (
        "web_results" in state
        and state["web_results"].strip()
        and "Error" not in state["web_results"]
    ):
        return "generate_from_web"

    return "answer_insufficient"
