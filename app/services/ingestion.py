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
import fitz  # Raw PyMuPDF engine for asset extraction
import io
from PIL import Image  # Ensure pillow is installed: pip install pillow

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
        super().__init__(**kwargs)
        self.max_words = max_words

    def split_text(self, text: str) -> List[str]:
        # Step 1: Utilize parent character rules (\n\n, \n) to form base blocks
        base_blocks = super().split_text(text)
        final_hybrid_chunks = []

        for block in base_blocks:
            # Step 2: Clean sentence boundary isolation regex rules
            sentences = re.split(
                r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s+", block.strip()
            )

            current_chunk = []
            current_count = 0

            for sentence in sentences:
                words = sentence.split()
                if not words:
                    continue

                # Critical Fallback: Handle huge uninterrupted text segments safely
                if len(words) > self.max_words:
                    if current_chunk:
                        final_hybrid_chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_count = 0

                    for i in range(0, len(words), self.max_words):
                        sub_slice = words[i : i + self.max_words]
                        final_hybrid_chunks.append(" ".join(sub_slice))
                    continue

                # Clean Buffer Management: Flush safely when constraints are crossed
                if current_count + len(words) > self.max_words:
                    if current_chunk:
                        final_hybrid_chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_count = len(words)
                else:
                    current_chunk.append(sentence)
                    current_count += len(words)

            # Flush out remaining sentences gracefully
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
            # Upgrade implementation pattern to utilize PyMuPDF layout + vision logic
            # Upgrade implementation pattern to utilize PyMuPDF layout + vision logic
            if ext.endswith(".pdf"):
                doc_file = fitz.open(file_path)
                full_document_text_list = []

                for page_idx in range(len(doc_file)):
                    page = doc_file[page_idx]
                    page_num = page_idx + 1
                    page_text = page.get_text()

                    page_accumulator = [page_text]

                    # Extract image artifacts embedded on this page layer
                    image_list = page.get_images(full=True)
                    if image_list:
                        logger.info(
                            f"📸 Processing {len(image_list)} images on Page {page_num} using Qwen2.5-VL 3B..."
                        )

                        for img_idx, img_meta in enumerate(image_list):
                            xref = img_meta[0]  # Grab correct index position element
                            base_image = doc_file.extract_image(xref)
                            image_bytes = base_image["image"]

                            try:
                                # 🚀 OPTIMIZATION: Reduce visual token overhead by scaling down images
                                pil_img = Image.open(io.BytesIO(image_bytes))
                                max_dim = 1024
                                if max(pil_img.size) > max_dim:
                                    pil_img.thumbnail(
                                        (max_dim, max_dim), Image.Resampling.LANCZOS
                                    )

                                # Convert back to compact JPEG bytes format
                                buffer = io.BytesIO()
                                pil_img.save(buffer, format="JPEG", quality=80)
                                optimized_image_bytes = buffer.getvalue()

                                # Call the lightweight 3B Vision model via Ollama
                                response = ollama.chat(
                                    model="qwen2.5vl:3b",  # ◄ Optimized 3B vision target
                                    messages=[
                                        {
                                            "role": "user",
                                            "content": (
                                                "Analyze this system administration screenshot or user manual illustration. "
                                                "Describe the text headers, input fields, labels, buttons, windows, and system contexts visible "
                                                "so it can be indexed cleanly for a technical support chatbot."
                                            ),
                                            "images": [optimized_image_bytes],
                                        }
                                    ],
                                )
                                caption_text = response["message"]["content"]

                                visual_context_node = (
                                    f"\n\n[EMBEDDED_SCREENSHOT_START]\n"
                                    f"Description of image asset #{img_idx + 1} on page {page_num}: {caption_text}\n"
                                    f"[EMBEDDED_SCREENSHOT_END]\n\n"
                                )
                                page_accumulator.append(visual_context_node)

                            except Exception as vlm_err:
                                logger.error(
                                    f"⚠️ 3B VLM processing failed on page {page_num}: {vlm_err}"
                                )
                                page_accumulator.append(
                                    f"\n\n[IMAGE_ASSET_DETECTED_ON_PAGE_{page_num}]\n\n"
                                )

                    # Anchor the merged text streams with explicit structural boundaries
                    combined_page_stream = "\n".join(page_accumulator)
                    if combined_page_stream.strip():
                        tagged_text = f"\n\n[PAGE_START_{page_num}]\n{combined_page_stream}\n[PAGE_END_{page_num}]\n\n"
                        full_document_text_list.append(tagged_text)
                        file_text_accumulator.append(page_text)

                # Step 2: Split the unified multi-modal document text as a single stream
                unified_doc_string = "".join(full_document_text_list)
                document_slices = text_splitter.split_text(unified_doc_string)

                # Step 3: Extract and assign page numbers to metadata based on anchors
                for slice_text in document_slices:
                    page_match = re.search(r"\[PAGE_START_(\d+)\]", slice_text)
                    detected_page = page_match.group(1) if page_match else "Unknown"

                    clean_slice = re.sub(
                        r"\[PAGE_START_\d+\]|\[PAGE_END_\d+\]|\[EMBEDDED_SCREENSHOT_START\]|\[EMBEDDED_SCREENSHOT_END\]",
                        "",
                        slice_text,
                    ).strip()

                    if clean_slice:
                        metadata_enhanced_text = f"[Source: {file_name} | Page: {detected_page}]\n{clean_slice}"
                        file_chunks_accumulator.append(metadata_enhanced_text)

                doc_file.close()

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
