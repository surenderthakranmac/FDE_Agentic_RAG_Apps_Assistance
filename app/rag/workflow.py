# Acts as the central entry point layout. It maps component tasks and compiles the workflow pipeline into a clean, lightweight module.
from langgraph.graph import StateGraph, START, END
from app.core.logger_setup import logger
from app.rag.state import AgentState
from app.rag.nodes import (
    node_route_question,
    node_retrieve_kb,
    node_grade_kb_evidence,
    node_search_web_fallback,
    node_rewrite_query,
    node_generate_from_kb,
    node_generate_from_web,
    node_direct_answer,
    node_answer_insufficient,
)
from app.rag.edges import edge_after_router, edge_after_kb_grade, edge_after_web_grade

# Initialize Graph Configuration Structure using the localized State ledger
workflow = StateGraph(AgentState)

# 1. Map modular functional processing node connections
workflow.add_node("route_question", node_route_question)
workflow.add_node("retrieve_kb", node_retrieve_kb)
workflow.add_node("grade_kb_evidence", node_grade_kb_evidence)
workflow.add_node("search_web", node_search_web_fallback)
workflow.add_node("rewrite_query", node_rewrite_query)
workflow.add_node("generate_from_kb", node_generate_from_kb)
workflow.add_node("generate_from_web", node_generate_from_web)
workflow.add_node("direct_answer", node_direct_answer)
workflow.add_node("answer_insufficient", node_answer_insufficient)

# 2. Build directional graph execution rules
workflow.add_edge(START, "route_question")

workflow.add_conditional_edges(
    "route_question",
    edge_after_router,
    {"retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"},
)

workflow.add_edge("retrieve_kb", "grade_kb_evidence")

workflow.add_conditional_edges(
    "grade_kb_evidence",
    edge_after_kb_grade,
    {"generate_from_kb": "generate_from_kb", "search_web": "search_web"},
)

# 🟢 FIXED MAPPING BLOCK: Replaced static edge with conditional edge routing rules
workflow.add_conditional_edges(
    "search_web",
    edge_after_web_grade,
    {
        "generate_from_web": "generate_from_web",
        "rewrite_query": "rewrite_query",
        "answer_insufficient": "answer_insufficient",
    },
)

workflow.add_edge("rewrite_query", "retrieve_kb")
workflow.add_edge("generate_from_kb", END)
workflow.add_edge("generate_from_web", END)
workflow.add_edge("direct_answer", END)
workflow.add_edge("answer_insufficient", END)

# 3. Compile full application workflow execution graph control matrix
app = workflow.compile()


# ──► ADD THIS WRAPPER AT THE VERY BOTTOM OF THE FILE:
def ask(question: str) -> dict:
    """
    Synchronous execution wrapper for the compiled LangGraph.
    Invokes the graph with the initial state and formats the output matrix.
    """
    logger.info(f"Graph execution initiated for question: '{question}'")
    
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    return app.invoke(initial)

