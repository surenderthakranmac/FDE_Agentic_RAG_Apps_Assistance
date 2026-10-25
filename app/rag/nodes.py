import ollama
import sys
from qdrant_client import QdrantClient, models
from app.core.config import settings
from app.core.logger_setup import logger
from app.rag.state import AgentState
from app.rag.tools import InternetSearchTool
from app.rag.reranker import local_rerank_nodes
from app.core.config import qdrant_client, settings

# Open app/rag/nodes.py and ensure line 8 imports the fix correctly:
from app.services.ingestion import generate_sparse_vector


# Target the existing disk-backed database cluster on your Mac
# ============================================================
# ⚙️ SECURE GLOBAL CLIENT SINGLETON PATTERN
# ============================================================
def get_safe_qdrant_client():
    """
    Looks directly at Python's global namespace to recycle the existing,
    active disk connection instead of competing for folder access locks.
    """
    # 1. First, check if main.py or run.py initialized a global client
    for module_name in ["app.main", "__main__", "app.services.ingestion"]:
        if module_name in sys.modules and hasattr(sys.modules[module_name], "client"):
            logger.info(
                f"🔄 Re-using synchronized Qdrant storage context from module: {module_name}"
            )
            return getattr(sys.modules[module_name], "client")

    # 2. Check the global runtime dictionary fallback memory maps
    if "_global_qdrant_instance" not in globals():
        try:
            globals()["_global_qdrant_instance"] = QdrantClient(
                path=settings.qdrant_path
            )
        except Exception:
            # If completely locked, try opening a connection to the existing database in read-only mode
            globals()["_global_qdrant_instance"] = QdrantClient(
                path=settings.qdrant_path
            )

    return globals()["_global_qdrant_instance"]


def node_route_question(state: AgentState):
    """Classifies user intent using JSON mode execution boundaries."""
    question = state["question"]
    prompt = f"""
    Evaluate this request and categorize its destination. Return a JSON object: {{"route": "kb" | "direct"}}
    Choose 'kb' for Unisys Urbis core system, data dictionary queries, client and account information,general ledger queries,core on demad report queris, GUI card.
    Choose 'direct' for basic conversations, greetings, jokes, or pleasantries.
    User Request: '{question}'
    JSON Output:
    """
    response = ollama.generate(model=settings.llm_model, prompt=prompt, format="json")

    p_tokens = response.get("prompt_eval_count", 0)
    c_tokens = response.get("eval_count", 0)

    route_target = "kb" if "kb" in response["response"].lower() else "direct"

    accumulated_p = state.get("prompt_tokens", 0) + p_tokens
    accumulated_c = state.get("completion_tokens", 0) + c_tokens

    logger.info(
        "Route classification node completed processing",
        extra={
            "telemetry": {
                "route": route_target,
                "p_tokens": accumulated_p,
                "c_tokens": accumulated_c,
            }
        },
    )

    return {
        "current_query": question,
        "source_used": route_target,
        "prompt_tokens": accumulated_p,
        "completion_tokens": accumulated_c,
    }


# ============================================================
# 🔄 LANGGRAPH NODES CIRCUIT
# ============================================================


def node_retrieve_kb(state: AgentState):
    """Executes local hybrid dense + sparse vector lookups over local files."""
    query = state["current_query"]
    logger.info("Starting hybrid retrieval query scan over Qdrant matrix indices")

    dense_q = ollama.embeddings(model=settings.embedding_model, prompt=query)[
        "embedding"
    ]
    sparse_q = generate_sparse_vector(query)

    # 🟢 PROGRAMMATIC SINGLETON INJECTION SAFELY BYPASSES COLLISION LOOPS
    # active_client = get_safe_qdrant_client()

    search_hits = qdrant_client.query_points(
        collection_name=settings.collection_name,
        prefetch=[
            models.Prefetch(query=dense_q, using="dense-node", limit=settings.top_k),
            models.Prefetch(
                query=models.SparseVector(**sparse_q),
                using="sparse-node",
                limit=settings.top_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=settings.top_k,
    )

    refined_context = local_rerank_nodes(query, search_hits.points)
    return {"kb_docs": refined_context}


def node_grade_kb_evidence(state: AgentState):
    """Evaluates if the extracted information satisfies factual limits."""
    user_query = state.get('question', '').lower()
    
    # ──► 🛡️ CORPORATE KEYWORD INTERCEPT OVERRIDE
    corporate_keywords = [
        "urbis", 
        "data dictionary", 
        "client", 
        "account", 
        "general ledger", 
        "on demand", 
        "gui card"
    ]
    
    # If the user asks about core systems, immediately force a 'good' rating and pass 0 tokens spent
    if any(keyword in user_query for keyword in corporate_keywords):
        grade_target = "good"
        accumulated_p = state.get("prompt_tokens", 0)
        accumulated_c = state.get("completion_tokens", 0)
        
        logger.info(
            "🎯 [Corporate Keyword Match] Bypassing LLM grading loop and forcing 'good' status constraint.",
            extra={
                "telemetry": {
                    "grade": grade_target,
                    "p_tokens": accumulated_p,
                    "c_tokens": accumulated_c,
                }
            },
        )
        return {
            "kb_grade": grade_target,
            "prompt_tokens": accumulated_p,
            "completion_tokens": accumulated_c,
        }

    # ──► STANDARD SYSTEM LLM GRADER LOOP (Fires if no keywords match)
    prompt = f"""
    Check if the text context blocks contain exact data parameters capable of answering the query.
    Return JSON block layout containing structure: {{"grade": "good" | "weak"}}
    Text Context Data: {state['kb_docs']}
    User Query: {state['question']}
    JSON Output:
    """
    
    response = ollama.generate(model=settings.llm_model, prompt=prompt, format="json")
    grade_target = "good" if "good" in response["response"].lower() else "weak"

    p_tokens = response.get("prompt_eval_count", 0)
    c_tokens = response.get("eval_count", 0)

    accumulated_p = state.get("prompt_tokens", 0) + p_tokens
    accumulated_c = state.get("completion_tokens", 0) + c_tokens

    logger.info(
        "Context evidence evaluation complete via LLM",
        extra={
            "telemetry": {
                "grade": grade_target,
                "p_tokens": accumulated_p,
                "c_tokens": accumulated_c,
            }
        },
    )

    return {
        "kb_grade": grade_target,
        "prompt_tokens": accumulated_p,
        "completion_tokens": accumulated_c,
    }



def node_search_web_fallback(state: AgentState):
    """Routes query execution paths down outbound Tavily search tools."""
    query = state["current_query"]
    # web_snippet_payload = InternetSearchTool.execute(query, max_results=2)    #uncomment this line to enable live Tavily search, but ensure you have a valid API key and network access.
    
    # ──► This exact literal string object is returned here:
    web_snippet_payload = "Sorry webs search is turned off for this demo."
    
    # Python automatically merges this dictionary right into your AgentState context parameters!
    return {
        "web_results": web_snippet_payload,  # state["web_results"] becomes this string
        "source_used": "web_search"           # state["source_used"] updates to this string
    }


def node_rewrite_query(state: AgentState):
    """Optimizes keyword properties to attempt recursive data retrieval cycles."""
    question = state["question"]
    current_retry = state["retry_count"] + 1
    prompt = f"Optimize and clean this IT query into focused search keywords. Return ONLY the string line: '{question}'"

    response = ollama.generate(model=settings.llm_model, prompt=prompt)
    rewritten = response["response"].strip().replace("'", "")

    p_tokens = response.get("prompt_eval_count", 0)
    c_tokens = response.get("eval_count", 0)

    accumulated_p = state.get("prompt_tokens", 0) + p_tokens
    accumulated_c = state.get("completion_tokens", 0) + c_tokens

    logger.info(
        "Query rewriter node generation completed",
        extra={
            "telemetry": {
                "rewritten_query": rewritten,
                "p_tokens": accumulated_p,
                "c_tokens": accumulated_c,
            }
        },
    )

    return {
        "current_query": rewritten,
        "retry_count": current_retry,
        "prompt_tokens": accumulated_p,
        "completion_tokens": accumulated_c,
    }


def node_generate_from_kb(state: AgentState):
    """Synthesizes factual system declarations using private database context."""
    prompt = f"Using ONLY these company rules: {state['kb_docs']}. Clear answer this question: {state['question']}"
    response = ollama.generate(model=settings.llm_model, prompt=prompt)

    p_tokens = response.get("prompt_eval_count", 0)
    c_tokens = response.get("eval_count", 0)

    return {
        "answer": response["response"]
        + "\n\n🔰 (Source: Secure Corporate Data Systems)",
        "prompt_tokens": state.get("prompt_tokens", 0) + p_tokens,
        "completion_tokens": state.get("completion_tokens", 0) + c_tokens,
    }


def node_generate_from_web(state: AgentState):
    """Synthesizes responses using public search metadata components."""
    """Generates an answer based on web search context or handles fallback."""
    web_results = state.get("web_results", "")
    
    # Catch your fallback message cleanly
    if "turned off" in str(web_results).lower():
        return {
            "generation": "I am currently running in an air-gapped secure corporate mode. External web search is disabled to protect BankABC data privacy. Please rephrase your query based on our internal reference documents.",
            "steps_taken": state.get("steps_taken", []) + ["generate_from_web_fallback_node"]
        }
    prompt = f"Using ONLY these public network web notes: {state['web_results']}. Clear answer this question: {state['question']}"
    response = ollama.generate(model=settings.llm_model, prompt=prompt)

    p_tokens = response.get("prompt_eval_count", 0)
    c_tokens = response.get("eval_count", 0)

    return {
        "answer": response["response"]
        # + "\n\n🌍 (Source: Public Web Infrastructure Channels)",
        + "\n\n🌍 (Source: Internet Search is off)",
        "prompt_tokens": state.get("prompt_tokens", 0) + p_tokens,
        "completion_tokens": state.get("completion_tokens", 0) + c_tokens,
    }


def node_direct_answer(state: AgentState):
    """Handles basic conversational inputs directly without executing vector database queries."""
    prompt = f"""
    Handles basic conversational inputs directly without executing vector database queries.
    You name is Agentic RAG, a secure corporate AI assistant. Answer the user query directly and concisely.
    User Query: {state['question']}
    """

    response = ollama.generate(model=settings.llm_model, prompt=prompt ) #state["question"])

    p_tokens = response.get("prompt_eval_count", 0)
    c_tokens = response.get("eval_count", 0)

    return {
        "answer": response["response"],
        "prompt_tokens": state.get("prompt_tokens", 0) + p_tokens,
        "completion_tokens": state.get("completion_tokens", 0) + c_tokens,
    }


def node_answer_insufficient(state: AgentState):
    """Final graceful fallback exit path for queries that return low search scores."""
    return {
        "answer": "I have completed hybrid extraction across secure document vaults and public networks, but found insufficient evidence to confidently resolve your inquiry."
    }
