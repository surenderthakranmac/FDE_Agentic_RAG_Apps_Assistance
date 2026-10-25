"""
Module Name: ingestion_config.py
Description: Establishes necessary imports, handles document parser routing rules,
             and declares persistent global variables for the local hybrid RAG matrix.
"""

import concurrent.futures
import os
import re
import json
import pickle
import time
from typing import List

# ─── LANGCHAIN CORE FRAMEWORK UTILITIES ───
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─── OFFICE SPREADSHEET & DOCUMENT EXTRACTORS ───
from docx import Document as DocxReader
import openpyxl

# ─── VECTOR METADATA STORAGE ENGINE LAYERS ───
import ollama
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

# ─── ENTERPRISE CONFIG & LOGGING UTILITIES ───
from app.core.config import settings, qdrant_client
from app.core.logger_setup import logger

# 🟢 FDE VALIDATION SCANS: Supported corporate extension safety whitelist
ALLOWED_OFFICE_EXTENSIONS = (".pdf", ".docx", ".xlsx")

# Persistent storage paths for state tracking and matrix lookup data
BM25_PICKLE_PATH = os.path.join(settings.qdrant_path, "bm25_matrix.pkl")
MANIFEST_PATH = os.path.join(settings.qdrant_path, "ingestion_manifest.json")
CORPUS_CACHE_PATH = os.path.join(settings.qdrant_path, "text_corpus_cache.json")

# Global baseline reference mapping placeholder for structural keyword searches
bm25_instance = None
"""
Module Name: hybrid_splitter.py
Description: Subclasses the standard LangChain RecursiveCharacterTextSplitter 
             to enforce sentence packing layout rules with strict token backup loops.
"""


class EnterpriseHybridTextSplitter(RecursiveCharacterTextSplitter):
    """
    Inherits structural character separation priorities from LangChain, adding
    deterministic regex logic to isolate sentences and enforce explicit maximum length guards.
    """

    def __init__(self, max_words: int = 400, **kwargs):
        """
        Initializes the custom splitter parameters.
        :param max_words: Hard ceiling for total words packed inside an individual chunk.
        """
        super().__init__(**kwargs)
        self.max_words = max_words

    def split_text(self, text: str) -> List[str]:
        """
        Slices text strings by recursively walking paragraph breaks, then refines
        them via sentence tracking boundaries to keep contextual concepts together.
        """
        # Step 1: Utilize parent character rules (\n\n, \n) to form base blocks
        base_blocks = super().split_text(text)
        final_hybrid_chunks = []

        for block in base_blocks:
            # Step 2: Use custom negative-lookbehind regex to accurately split sentences
            sentences = re.split(
                r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+", block.strip()
            )

            current_chunk = []
            current_count = 0

            for sentence in sentences:
                words = sentence.split()

                # Critical Fallback Branch: Handle huge tables or unpunctuated system logs
                if len(words) > self.max_words:
                    # Flush any previously accumulated structural buffer chunks
                    if current_chunk:
                        final_hybrid_chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_count = 0

                    # Direct mathematical stride slice to guarantee context alignment
                    for i in range(0, len(words), self.max_words):
                        sub_slice = words[i : i + self.max_words]
                        final_hybrid_chunks.append(" ".join(sub_slice))
                    continue

                # Normal packing operation check: Flush if constraints would be violated
                if current_count + len(words) > self.max_words and current_chunk:
                    final_hybrid_chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_count = 0

                current_chunk.append(sentence)
                current_count += len(words)

            # Clear remaining buffered lines out of loop frame limits
            if current_chunk:
                final_hybrid_chunks.append(" ".join(current_chunk))

        return final_hybrid_chunks


"""
Module Name: ingestion_helpers.py
Description: Manages underlying storage extraction processes, serializes schema, 
             and creates numerical indexing values for BM25 keyword matching scores.
"""


def generate_sparse_vector(text: str) -> dict:
    """
    Calculates dynamic BM25 score metrics over target text elements.
    Used for cross-indexing against dense node embeddings inside Qdrant.
    """
    global bm25_instance

    # Lazy load the compiled corpus binary directly if missing from cache context
    if bm25_instance is None:
        if os.path.exists(BM25_PICKLE_PATH):
            with open(BM25_PICKLE_PATH, "rb") as f:
                bm25_instance = pickle.load(f)
        else:
            # Return an empty matrix fallback vector if no index base has been written
            return {"indices": [], "values": []}

    # Tokenize input context string using standard clean character spaces
    tokens = text.lower().split()
    scores = bm25_instance.get_scores(tokens)

    # Map valid score hits back to corresponding index positions
    return {
        "indices": [idx for idx, score in enumerate(scores) if score > 0],
        "values": [float(scores[idx]) for idx, score in enumerate(scores) if score > 0],
    }


def load_json_file(path: str) -> dict:
    """
    Safely reads file paths to protect runtime threads from unexpected JSON crash patterns.
    """
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(
                f"⚠️ Read failure on file path mapping matrix: {path}. Error: {e}"
            )
            return {}
    return {}


"""
Module Name: ingestion_bootstrap.py
Description: Evaluates modification timestamps on disk to process file changes incrementally,
             and handles collection initialization logic inside Qdrant.
"""


def run_native_ingestion(force_reindex: bool = False):
    """
    Main orchestration function. Evaluates workspace files to process delta changes,
    preventing duplicate vector computations.
    """
    global bm25_instance
    logger.info(
        "🛠️ [Ingestion Engine] Initializing dynamic multi-format database validation scan..."
    )
    start_total_pipeline = time.perf_counter()

    # Step 1: Query Ollama to verify actual output dimensions needed for collection setup
    probe_dims = len(
        ollama.embeddings(model=settings.embedding_model, prompt="probe")["embedding"]
    )

    # Step 2: Establish base storage configuration layers if missing
    if not qdrant_client.collection_exists(settings.collection_name):
        logger.info(
            f"📁 Creating pristine collection space: {settings.collection_name}"
        )
        qdrant_client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={
                "dense-node": models.VectorParams(
                    size=probe_dims, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={"sparse-node": models.SparseVectorParams()},
        )
        force_reindex = True

    # Step 3: Load existing document maps to match changes
    current_manifest = load_json_file(MANIFEST_PATH)
    cached_corpus = load_json_file(CORPUS_CACHE_PATH)

    new_manifest = {}
    new_corpus = {}
    updated_files = []
    deleted_files = []

    # Step 4: Map filesystem states to calculate incremental processing requirements
    if os.path.exists(settings.lectures_folder):
        for file_name in os.listdir(settings.lectures_folder):
            if file_name.lower().endswith(ALLOWED_OFFICE_EXTENSIONS):
                file_path = os.path.join(settings.lectures_folder, file_name)
                mtime = os.path.getmtime(file_path)
                new_manifest[file_name] = mtime

                # Check if file is completely new or has modification updates
                if (
                    force_reindex
                    or file_name not in current_manifest
                    or current_manifest[file_name] != mtime
                ):
                    updated_files.append(file_name)

    # Step 5: Track documents that were deleted from the folder
    for old_file in current_manifest.keys():
        if old_file not in new_manifest:
            deleted_files.append(old_file)

    # Short-circuit early if there are no data configuration changes to process
    if not updated_files and not deleted_files and not force_reindex:
        collection_info = qdrant_client.get_collection(
            collection_name=settings.collection_name
        )
        if os.path.exists(BM25_PICKLE_PATH):
            with open(BM25_PICKLE_PATH, "rb") as f:
                bm25_instance = pickle.load(f)
        logger.info(
            "💾 [Ingestion Cache Hit] Files match on disk. Skipping all extraction and embedding loops.",
            extra={
                "telemetry": {
                    "total_points": collection_info.points_count,
                    "latency_ms": round(
                        (time.perf_counter() - start_total_pipeline) * 1000, 2
                    ),
                }
            },
        )
        return

    logger.info(
        f"⚠️ [Ingestion Cache Miss] Processing delta updates for: {updated_files if updated_files else 'Deletion Sync'}"
    )

    # Retain unchanged files in the active processing context
    for file_name, mtime in new_manifest.items():
        if file_name in cached_corpus and file_name not in updated_files:
            new_corpus[file_name] = cached_corpus[file_name]

    # Initialize the customized hybrid page character splitter
    text_splitter = EnterpriseHybridTextSplitter(
        max_words=400,
        chunk_size=1600,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    file_chunk_mapping = {}
    # ─────────────────────────────────────────────────────────────────
    # 📑 MULTI-FORMAT EXTRACTION LOOP (Inlined inside run_native_ingestion)
    # ─────────────────────────────────────────────────────────────────
    for file_name in updated_files:
        file_path = os.path.join(settings.lectures_folder, file_name)
        start_parse = time.perf_counter()
        file_text_accumulator = []
        file_chunks_accumulator = []
        ext = file_name.lower()

        try:
            # Upgrade implementation pattern to utilize PyMuPDFLoader layout logic
            if ext.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
                lc_pages = loader.load()

                for doc_page in lc_pages:
                    page_text = doc_page.page_content
                    page_num = doc_page.metadata.get("page", 0) + 1

                    if page_text.strip():
                        file_text_accumulator.append(page_text)

                        # Apply hybrid chunk partitioning dynamically at the individual page frame boundaries
                        page_slices = text_splitter.split_text(page_text)
                        for slice_text in page_slices:
                            # Prepend clean string metadata context parameters to text chunks
                            metadata_enhanced_text = f"[Source: {file_name} | Page: {page_num}]\n{slice_text}"
                            file_chunks_accumulator.append(metadata_enhanced_text)

            # Process Microsoft Word files using paragraph extraction
            elif ext.endswith(".docx"):
                doc = DocxReader(file_path)
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        file_text_accumulator.append(paragraph.text)

                # Split unified text blocks down into target chunks
                file_chunks_accumulator = text_splitter.split_text(
                    "\n".join(file_text_accumulator)
                )

            # Process Microsoft Excel sheets via cell string concatenation layers
            elif ext.endswith(".xlsx"):
                wb = openpyxl.load_workbook(file_path, data_only=True)
                for sheet in wb.worksheets:
                    for row in sheet.iter_rows(values_only=True):
                        cells = [str(cell).strip() for cell in row if cell is not None]
                        if cells:
                            file_text_accumulator.append(" | ".join(cells))

                file_chunks_accumulator = text_splitter.split_text(
                    "\n".join(file_text_accumulator)
                )

            # Store the extracted text strings back into the global cache mappings
            new_corpus[file_name] = "\n".join(file_text_accumulator)
            file_chunk_mapping[file_name] = file_chunks_accumulator
            all_chunks.extend(file_chunks_accumulator)

            logger.info(
                "Extracted and split text delta data via LangChain Hybrid tools",
                extra={
                    "telemetry": {
                        "file_name": file_name,
                        "chunks": len(file_chunks_accumulator),
                        "latency_ms": round(
                            (time.perf_counter() - start_parse) * 1000, 2
                        ),
                    }
                },
            )
        except Exception as e:
            logger.error(
                f"❌ Critical extraction fault on target file: {file_name}. Error details: {e}"
            )
    # ─────────────────────────────────────────────────────────────────
    # 💾 MATRIX INDICES SYNCHRONIZATION (Inlined inside run_native_ingestion)
    # ─────────────────────────────────────────────────────────────────

    # Process unmodified cached corporate files to keep the global corpus intact
    for file_name in sorted(cached_corpus.keys()):
        if file_name not in updated_files and file_name in cached_corpus:
            new_corpus[file_name] = cached_corpus[file_name]
            cached_chunks = text_splitter.split_text(cached_corpus[file_name])
            file_chunk_mapping[file_name] = cached_chunks
            all_chunks.extend(cached_chunks)

    # Commit the raw string values back to cache directories on physical storage
    os.makedirs(settings.qdrant_path, exist_ok=True)
    with open(CORPUS_CACHE_PATH, "w") as f:
        json.dump(new_corpus, f)

    # Compute lexical statistics and store the updated BM25 lookup matrices
    logger.info(
        "🧮 Compiling incremental corpus matrices into local BM25 mapping arrays..."
    )
    tokenized_corpus = [chunk.lower().split() for chunk in all_chunks]
    bm25_instance = BM25Okapi(tokenized_corpus)
    with open(BM25_PICKLE_PATH, "wb") as f:
        pickle.dump(bm25_instance, f)

    # Determine collection handling and initial index counter assignments
    if deleted_files or force_reindex or len(current_manifest) == 0:
        logger.info(
            "🗑️ Resetting Qdrant collection to ensure full alignment across nodes..."
        )
        qdrant_client.delete_collection(settings.collection_name)
        qdrant_client.create_collection(
            collection_name=settings.collection_name,
            vectors_config={
                "dense-node": models.VectorParams(
                    size=probe_dims, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={"sparse-node": models.SparseVectorParams()},
        )
        files_to_vectorize = list(new_corpus.keys())
        starting_point_idx = 0
    else:
        # Increment index allocations from existing collection records
        files_to_vectorize = updated_files
        starting_point_idx = qdrant_client.get_collection(
            collection_name=settings.collection_name
        ).points_count
    # ─────────────────────────────────────────────────────────────────
    # 🚀 THREAD WORKERS & BATCH UPLOAD (Final segment of run_native_ingestion)
    # ─────────────────────────────────────────────────────────────────
    points = []
    point_id_counter = starting_point_idx

    def process_single_chunk(chunk: str, file_name: str) -> dict:
        """
        Isolated helper task that calls local Ollama endpoints concurrently
        and extracts sparse scores from memory.
        """
        try:
            dense_v = ollama.embeddings(model=settings.embedding_model, prompt=chunk)[
                "embedding"
            ]
            sparse_v = generate_sparse_vector(chunk)
            return {
                "dense": dense_v,
                "sparse": sparse_v,
                "text": chunk,
                "file": file_name,
            }
        except Exception as ex:
            logger.error(
                f"❌ Local processing fault encountered via Ollama API thread worker: {ex}"
            )
            return None

    # Iterate through individual files to generate model vectors concurrently
    for file_name in files_to_vectorize:
        chunks_to_process = file_chunk_mapping.get(file_name, [])
        logger.info(
            "Calculating embeddings for updated document via parallel workers",
            extra={
                "telemetry": {"file_name": file_name, "chunks": len(chunks_to_process)}
            },
        )

        # Max workers scaled to 2 to prevent thread locking on core local Ollama execution paths
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(process_single_chunk, chk, file_name)
                for chk in chunks_to_process
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    points.append(
                        models.PointStruct(
                            id=point_id_counter,
                            vector={
                                "dense-node": result["dense"],
                                "sparse-node": result["sparse"],
                            },
                            payload={
                                "text_content": result["text"],
                                "source_file": result["file"],
                            },
                        )
                    )
                    point_id_counter += 1

    # Execute atomic write upsert cleanly outside of concurrent processing worker loops
    if points:
        logger.info(
            f"🚀 Initiating mass batch payload upload of {len(points)} points to Qdrant collection cluster..."
        )
        qdrant_client.upsert(collection_name=settings.collection_name, points=points)

    # Save manifest states to secure the cache boundary layer
    with open(MANIFEST_PATH, "w") as f:
        json.dump(new_manifest, f, indent=4)

    logger.info(
        "Local storage schema collection matrix fully synchronized incrementally.",
        extra={
            "telemetry": {
                "new_points_added": len(points),
                "total_pipeline_latency_ms": round(
                    (time.perf_counter() - start_total_pipeline) * 1000, 2
                ),
            }
        },
    )
