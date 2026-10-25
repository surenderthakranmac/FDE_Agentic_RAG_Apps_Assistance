# Isolates the mathematical array and matrix configurations required to compute text cross-comparisons.
import numpy as np
import ollama
from app.core.config import settings
from app.core.logger_setup import logger


def local_rerank_nodes(query: str, retrieved_nodes: list, top_k: int = 4) -> str:
    """
    Performs precise cross-comparison mapping between raw query vectors
    and returned database payloads to bubble up the best matching context chunks.
    Upgraded to allow multi-chunk context passing to prevent single-page bottlenecks.
    """
    if not retrieved_nodes:
        return "No text snippets available to re-rank."

    # Generate query array vectors a single time outside of loop steps
    query_vector = np.array(
        ollama.embeddings(model=settings.embedding_model, prompt=query)["embedding"]
    )
    scored_nodes = []

    for node in retrieved_nodes:
        # Pull text accurately from your unified payload schema layout keys
        text_content = node.payload.get("text_content", "")
        if not text_content.strip():
            continue

        try:
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
        except Exception as e:
            logger.error(f"Error computing inline embeddings during re-ranking: {e}")
            continue

    if not scored_nodes:
        return "No valid text vectors computed during re-ranking loop."

    # Sort candidates by highest cosine similarity metrics score first
    scored_nodes.sort(key=lambda x: x[0], reverse=True)

    # Track the highest achieved metric parameter for telemetry records
    top_score = float(scored_nodes[0][0])

    # Extract text from the top distinct nodes requested by top_k
    top_hits = scored_nodes[:top_k]

    # Merge multiple relevant structural data pages together using explicit prompt partitions
    combined_context_payload = (
        "\n\n==================================================\n\n".join(
            [hit_text for score, hit_text in top_hits]
        )
    )

    logger.info(
        "Re-ranker evaluation sequence execution completed",
        extra={
            "telemetry": {
                "top_score": round(top_score, 4),
                "chunks_delivered": len(top_hits),
            }
        },
    )

    return combined_context_payload
