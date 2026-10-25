
import concurrent.futures # ◄ For parallelizing your CPU loops
import os
import re
import json
import pickle
import time
from typing import List
from pypdf import PdfReader
from docx import Document as DocxReader
import openpyxl
import ollama
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from app.core.config import settings
from app.core.logger_setup import logger
from app.core.config import settings, qdrant_client # ◄ Import qdrant_client here


# 🟢 FDE ENHANCEMENT: Whitelist defining supported enterprise file formats
ALLOWED_OFFICE_EXTENSIONS = (".pdf", ".docx", ".xlsx")

BM25_PICKLE_PATH = os.path.join(settings.qdrant_path, "bm25_matrix.pkl")
MANIFEST_PATH = os.path.join(settings.qdrant_path, "ingestion_manifest.json")
CORPUS_CACHE_PATH = os.path.join(settings.qdrant_path, "text_corpus_cache.json")


import re
from typing import List

def hybrid_chunk_text(text: str, max_words: int = 400) -> List[str]:
    """
    Chunks text by sentence boundaries with a guaranteed fallback constraint 
    to prevent tokens from overflowing local embedding windows (e.g., Ollama bge-m3).
    """
    # Lowered max_words to 400 to safely fit within Ollama's 2048 token window
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+", text.strip())
    
    chunks = []
    current_chunk = []
    current_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        
        # Fallback: If a single sentence is structurally giant, slice it directly
        if len(words) > max_words:
            # Clear out what we have so far
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_count = 0
            
            # Sub-divide the massive chunk by character/word blocks directly
            for i in range(0, len(words), max_words):
                sub_slice = words[i:i + max_words]
                chunks.append(" ".join(sub_slice))
            continue

        # Normal operation chunk packing
        if current_count + len(words) > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_count = 0
            
        current_chunk.append(sentence)
        current_count += len(words)
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks



def generate_sparse_vector(text: str) -> dict:
    global bm25_instance
    if bm25_instance is None:
        if os.path.exists(BM25_PICKLE_PATH):
            with open(BM25_PICKLE_PATH, "rb") as f:
                bm25_instance = pickle.load(f)
        else:
            return {"indices": [], "values": []}

    tokens = text.lower().split()
    scores = bm25_instance.get_scores(tokens)
    return {
        "indices": [idx for idx, score in enumerate(scores) if score > 0],
        "values": [float(scores[idx]) for idx, score in enumerate(scores) if score > 0],
    }


def load_json_file(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def run_native_ingestion(force_reindex: bool = False):
    """
    True Incremental Multi-Format Ingestion Engine.
    Optimized with Concurrent Thread Pool workers to speed up CPU local embeddings.    """

    global bm25_instance
    logger.info("🛠️ [Ingestion Engine] Initializing dynamic multi-format database validation scan...")
    start_total_pipeline = time.perf_counter()

    probe_dims = len(ollama.embeddings(model=settings.embedding_model, prompt="probe")["embedding"])
    # client = QdrantClient(path=settings.qdrant_path)

    if not qdrant_client.collection_exists(settings.collection_name):
        qdrant_client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={"dense-node": models.VectorParams(size=probe_dims, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse-node": models.SparseVectorParams()},
        )
        force_reindex = True

    current_manifest = load_json_file(MANIFEST_PATH)
    cached_corpus = load_json_file(CORPUS_CACHE_PATH)

    new_manifest = {}
    new_corpus = {}
    updated_files = []
    deleted_files = []

    if os.path.exists(settings.lectures_folder):
        for file_name in os.listdir(settings.lectures_folder):
            if file_name.lower().endswith(ALLOWED_OFFICE_EXTENSIONS):
                file_path = os.path.join(settings.lectures_folder, file_name)
                mtime = os.path.getmtime(file_path)
                new_manifest[file_name] = mtime

                if force_reindex or file_name not in current_manifest or current_manifest[file_name] != mtime:
                    updated_files.append(file_name)

    for old_file in current_manifest.keys():
        if old_file not in new_manifest:
            deleted_files.append(old_file)

    if not updated_files and not deleted_files and not force_reindex:
        collection_info = qdrant_client.get_collection(collection_name=settings.collection_name)
        if os.path.exists(BM25_PICKLE_PATH):
            with open(BM25_PICKLE_PATH, "rb") as f:
                bm25_instance = pickle.load(f)
        logger.info(
            "💾 [Ingestion Cache Hit] Files match on disk. Skipping all extraction and embedding loops.",
            extra={
                "telemetry": {
                    "total_points": collection_info.points_count,
                    "latency_ms": round((time.perf_counter() - start_total_pipeline) * 1000, 2),
                }
            },
        )
        # client.close()
        return

    logger.info(f"⚠️ [Ingestion Cache Miss] Processing delta updates for: {updated_files if updated_files else 'Deletion Sync'}")

    for file_name, mtime in new_manifest.items():
        if file_name in cached_corpus and file_name not in updated_files:
            new_corpus[file_name] = cached_corpus[file_name]

    # Extraction Loop
    for file_name in updated_files:
        file_path = os.path.join(settings.lectures_folder, file_name)
        start_parse = time.perf_counter()
        file_text_accumulator = []
        ext = file_name.lower()

        try:
            if ext.endswith(".pdf"):
                reader = PdfReader(file_path)
                for page in reader.pages:
                    t = page.extract_text()
                    if t: file_text_accumulator.append(t)
            elif ext.endswith(".docx"):
                doc = DocxReader(file_path)
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip(): file_text_accumulator.append(paragraph.text)
            elif ext.endswith(".xlsx"):
                wb = openpyxl.load_workbook(file_path, data_only=True)
                for sheet in wb.worksheets:
                    for row in sheet.iter_rows(values_only=True):
                        cells = [str(cell).strip() for cell in row if cell is not None]
                        if cells: file_text_accumulator.append(" | ".join(cells))

            new_corpus[file_name] = "\n".join(file_text_accumulator)
            logger.info(f"Natively extracted text delta data", extra={"telemetry": {"file_name": file_name, "latency_ms": round((time.perf_counter() - start_parse) * 1000, 2)}})
        except Exception as e:
            logger.error(f"Failed parsing file: {file_name}. Error: {e}")

    os.makedirs(settings.qdrant_path, exist_ok=True)
    with open(CORPUS_CACHE_PATH, "w") as f:
        json.dump(new_corpus, f)

    all_chunks = []
    file_chunk_mapping = {}
    for file_name in sorted(new_corpus.keys()):
        file_chunks = hybrid_chunk_text(new_corpus[file_name])
        file_chunk_mapping[file_name] = file_chunks
        all_chunks.extend(file_chunks)

    tokenized_corpus = [chunk.lower().split() for chunk in all_chunks]
    bm25_instance = BM25Okapi(tokenized_corpus)
    with open(BM25_PICKLE_PATH, "wb") as f:
        pickle.dump(bm25_instance, f)

    if deleted_files or force_reindex or len(current_manifest) == 0:
        qdrant_client.delete_collection(settings.collection_name)
        qdrant_client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={"dense-node": models.VectorParams(size=probe_dims, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse-node": models.SparseVectorParams()},
        )
        files_to_vectorize = list(new_corpus.keys())
        starting_point_idx = 0
    else:
        files_to_vectorize = updated_files
        starting_point_idx = qdrant_client.get_collection(collection_name=settings.collection_name).points_count

    # ─────────────────────────────────────────────────────────────────
    # 🔥 FIX: CONCURRENT MULTI-THREADED EMBEDDING WORKERS
    # ─────────────────────────────────────────────────────────────────
    points = []
    point_id_counter = starting_point_idx

    # Worker helper function for concurrent vector generation
    def process_single_chunk(chunk, file_name):
        try:
            dense_v = ollama.embeddings(model=settings.embedding_model, prompt=chunk)["embedding"]
            sparse_v = generate_sparse_vector(chunk)
            return {"dense": dense_v, "sparse": sparse_v, "text": chunk, "file": file_name}
        except Exception as ex:
            logger.error(f"Error processing text block chunk via Ollama: {ex}")
            return None

    for file_name in files_to_vectorize:
        chunks_to_process = file_chunk_mapping.get(file_name, [])
        logger.info(
            f"Calculating embeddings for updated document via parallel workers",
            extra={"telemetry": {"file_name": file_name, "chunks": len(chunks_to_process)}}
        )

        # Using ThreadPoolExecutor allows Ollama to process chunks concurrently 
        # instead of freezing on a single channel line.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_single_chunk, chk, file_name) for chk in chunks_to_process]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    points.append(
                        models.PointStruct(
                            id=point_id_counter,
                            vector={"dense-node": result["dense"], "sparse-node": result["sparse"]},
                            payload={"text_content": result["text"], "source_file": result["file"]},
                        )
                    )
                    point_id_counter += 1

    if points:
        qdrant_client.upsert(collection_name=settings.collection_name, points=points)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(new_manifest, f, indent=4)

    # client.close()
    logger.info(
        "Local storage schema collection matrix fully synchronized incrementally.",
        extra={"telemetry": {"new_points_added": len(points), "total_pipeline_latency_ms": round((time.perf_counter() - start_total_pipeline) * 1000, 2)}}
    )
