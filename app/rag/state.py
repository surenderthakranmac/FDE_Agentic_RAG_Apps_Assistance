from typing import List
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    The master state dictionary blueprint for the Enterprise IT Copilot workflow graph.
    Serves as the shared memory notepad tracking values across node executions.
    """

    # 📝 Core Query Parameters
    question: str  # The original unedited support query submitted by the employee
    current_query: (
        str  # The active search query string (modified if query rewriting triggers)
    )

    # 📦 Extraction Context Holders
    kb_docs: str  # Text snippet payloads surfaced by the re-ranked hybrid vector search
    kb_grade: str  # Quality assessment score belonging to private data match ('good' or 'weak')
    web_results: str  # Raw text telemetry scraped from public fallback channels
    web_grade: str  # Quality assessment score belonging to public web metrics ('good' or 'weak')

    # 🤖 Output Variables
    answer: str  # The finalized factual response synthesized by the LLM
    source_used: str  # Tracking indicator showing final dataset source ('kb', 'web_search', or 'direct')
    retry_count: int  # Processing loop protection counter preventing infinite graph routing recourses

    # 📊 System Telemetry Trackers
    prompt_tokens: (
        int  # Cumulative input evaluator tokens consumed across all model nodes
    )
    completion_tokens: (
        int  # Cumulative output generation tokens consumed across all model nodes
    )
