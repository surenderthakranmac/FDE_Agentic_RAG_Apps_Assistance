# Isolates the mathematical array and matrix configurations required to compute text cross-comparisons.
import numpy as np
import ollama
from app.core.config import settings
from app.core.logger_setup import logger


def local_rerank_nodes(query: str, retrieved_nodes: list, top_k: int = 1) -> str:
    """
    Performs precise cross-comparison mapping between raw query vectors
    and returned database payloads to bubble up the best matching context chunk.
    """
    if not retrieved_nodes:
        return "No text snippets available to re-rank."

    query_vector = np.array(
        ollama.embeddings(model=settings.embedding_model, prompt=query)["embedding"]
    )
    scored_nodes = []

    for node in retrieved_nodes:
        text_content = node.payload["text_content"]
        node_vector = np.array(
            ollama.embeddings(model=settings.embedding_model, prompt=text_content)[
                "embedding"
            ]
        )

        # Calculate standard mathematical Cosine Similarity matrix dot arrays
        similarity = np.dot(query_vector, node_vector) / (
            np.linalg.norm(query_vector) * np.linalg.norm(node_vector)
        )
        scored_nodes.append((similarity, text_content))

    scored_nodes.sort(key=lambda x: x, reverse=True)
    top_score = float(scored_nodes[0][0])
    best_hit_text = scored_nodes[0][1]

    logger.info(
        "Re-ranker evaluation sequence execution completed",
        extra={"telemetry": {"top_score": round(top_score, 4)}},
    )
    return best_hit_text
