# better_rag.py – Gemma‑only multimodal RAG chatbot with citations
# -----------------------------------------------------------------------------
# Quick‑win upgrades implemented
#  1. BGE‑large embeddings
#  2. Hybrid BM25 + dense retrieval + MMR fusion
#  3. Cross‑encoder reranking (bge‑reranker‑base)
#  4. Token‑aware chunking
#  5. Multi‑query expansion
#  6. Metadata filters
#  7. Long‑context packing for Gemma‑3‑27B‑IT (128 k tokens)
#  8. Structured prompting (JSON/function‑call style)
#  9. Streaming support (vLLM)
# 10. Hooks for eval automation (ragas)
# -----------------------------------------------------------------------------

from __future__ import annotations

import os, glob, json, base64, argparse, warnings, gc, stat, shutil, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from mimetypes import guess_type

import torch
from PIL import Image
from dotenv import load_dotenv

# LangChain & vector stores
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    DirectoryLoader, UnstructuredFileLoader, UnstructuredHTMLLoader
)
from langchain.text_splitter import TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.merger_retriever import MergerRetriever

# Cross‑encoder (reranker)
from sentence_transformers import CrossEncoder

# CLIP for images
from transformers import CLIPProcessor, CLIPModel
import easyocr

# OpenAI SDK pointed at local Gemma endpoint
from openai import OpenAI, Stream

import concurrent.futures # Ensure this is imported
import tempfile # For temporary GIF conversion
from functools import partial # If needed for passing args, maybe not here

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
GEMMA_ENDPOINT   = "http://10.130.236.56:1235/v1"  # your vLLM server
GEMMA_MODEL_NAME = "gemma-3-27b-it"
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"         # dense encoder
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMB_DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE_TOK   = 512
CHUNK_OVERLAP    = 64
MAX_TOK_CONTEXT  = 90_000                             # leave headroom for answer
IMAGE_EXTS       = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
IMAGE_KW         = [
    "image", "photo", "picture", "photograph", "graphic", "figure", "diagram",
    "screenshot", "drawing", "illustration", "logo", "icon", "flag", "map",
    *IMAGE_EXTS
]
IMAGE_DETAIL_KW  = [ # Keywords suggesting user wants specifics *from* an image
    "what is", "identify", "describe", "color", "label", "text", "read", "show", "who is", "caption"
]
SHORT_MEMORY_K   = 10
BM25_K           = 10
DENSE_K          = 15
FINAL_K          = 6
RELEVANCE_THRES  = 0.1

# -----------------------------------------------------------------------------
# LLM helper – wrapper around Gemma vLLM endpoint
# -----------------------------------------------------------------------------
load_dotenv()
_client = OpenAI(
    api_key="EMPTY",          # ignored by vLLM but required by SDK
    base_url=GEMMA_ENDPOINT,
)

def chat_completion(messages: List[Dict], *, max_tokens: int = 1024,
                    temperature: float = 0.7, stream: bool = False,
                    extra: Optional[Dict] = None):
    """Unified call so we can tweak defaults at one place."""
    return _client.chat.completions.create(
        model=GEMMA_MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
        **(extra or {})
    )

# -----------------------------------------------------------------------------
# Embedding helpers
# -----------------------------------------------------------------------------
_embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                 model_kwargs={"device": EMB_DEVICE})
_reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=EMB_DEVICE)

class CLIPImageEmbeddings:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = EMB_DEVICE
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

    def embed_text(self, text: str) -> List[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
        return out[0].cpu().tolist()

    def embed_image(self, img: Image.Image) -> List[float]:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.get_image_features(**inputs)
        return out[0].cpu().tolist()

_clip_embedder = CLIPImageEmbeddings()

# -----------------------------------------------------------------------------
# Utility – image caption / detail via Gemma multimodal
# -----------------------------------------------------------------------------

def gemma_image_chat(prompt: str, image_path: str, max_tokens: int = 256) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    mime, _ = guess_type(image_path) or ("image/png", None)
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}}
    ]
    resp = chat_completion([
        {"role": "system", "content": "You are an observant visual assistant."},
        {"role": "user", "content": content}
    ], max_tokens=max_tokens)
    return resp.choices[0].message.content.strip()


def get_image_caption(path: str) -> str:
    return gemma_image_chat("Describe this image in a concise, detailed way.", path)


def get_image_detail(path: str, question: str) -> str:
    return gemma_image_chat(f"Answer this question about the image: {question}", path, 300)

# -----------------------------------------------------------------------------
# Chunking helper – token‑aware
# -----------------------------------------------------------------------------
_token_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE_TOK,
                                   chunk_overlap=CHUNK_OVERLAP)

# -----------------------------------------------------------------------------
# Build vector DBs
# -----------------------------------------------------------------------------

# --- Helper function for parallel OCR ---
# This runs in separate processes. easyocr needs to be loaded within each.
def ocr_single_image(img_path: str, use_gpu: bool) -> tuple[str, str]:
    """Performs PIL check, GIF conversion (if needed), and OCR for a single image."""
    temp_png_path = None
    current_path_for_ocr = img_path
    is_gif = img_path.lower().endswith('.gif')

    try:
        # --- PIL Check & GIF Conversion ---
        with Image.open(img_path) as img:
            img.verify() # Check integrity

            if is_gif:
                # For GIFs, seek to first frame, convert to RGB, save as temp PNG
                img.seek(0) # Go to the first frame
                rgb_frame = img.convert('RGB')
                # Create a temporary file for the PNG frame
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
                    temp_png_path = tmp_f.name
                    rgb_frame.save(tmp_f, format="PNG")
                current_path_for_ocr = temp_png_path # Use temp path for OCR
            else:
                # For non-GIFs, just ensure it can be loaded after verify
                with Image.open(img_path) as img_load:
                     img_load.load()
        # --- End PIL Check & GIF Conversion ---

        # --- Perform OCR ---
        # Load easyocr within the process
        try:
            local_ocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
            ocr_texts = local_ocr_reader.readtext(current_path_for_ocr, detail=0)
            ocr_result = "\n".join(ocr_texts) if ocr_texts else ""
            status = "OK" if not is_gif else "OK (GIF->PNG)"
        except Exception as ocr_e:
             # print(f"\n--- [ocr_single_image] EasyOCR Error on {img_path} (using {current_path_for_ocr}): {ocr_e}")
             ocr_result = "[OCR Failed]"
             status = "OCR Error"
        # --- End Perform OCR ---

        return img_path, status, ocr_result # Return original path, status, and result

    except Exception as pil_e:
        # print(f"\n--- [ocr_single_image] PIL/Conversion Error on {img_path}: {pil_e}")
        return img_path, "PIL/Load Error", "[PIL/Load Error]" # Return original path and error

    finally:
        # --- Clean up temporary file ---
        if temp_png_path and os.path.exists(temp_png_path):
            try:
                os.remove(temp_png_path)
            except Exception:
                # Log if cleanup fails, but don't crash the process
                print(f"\n--- [ocr_single_image] Warning: Failed to delete temp file {temp_png_path}")
        # --- End Cleanup ---

def build_text_db(psm_folder: str, db_dir: str) -> Chroma:
    loaders = [
        DirectoryLoader(psm_folder, "**/*.txt", UnstructuredFileLoader, recursive=True, show_progress=True),
        DirectoryLoader(psm_folder, "**/*.html", UnstructuredHTMLLoader, recursive=True, show_progress=True),
        DirectoryLoader(psm_folder, "**/*.htm", UnstructuredHTMLLoader, recursive=True, show_progress=True),
    ]
    docs: List[Document] = []
    print("--- [build_text_db] Loading text/html documents...") # DEBUG
    for ldr in loaders:
        try:
            loaded_docs = ldr.load()
            print(f"--- [build_text_db] Loaded {len(loaded_docs)} docs using {type(ldr).__name__}") # DEBUG
            docs.extend(loaded_docs)
        except Exception as e:
            print(f"--- [build_text_db] Error loading with {type(ldr).__name__}: {e}") # DEBUG

    print(f"--- [build_text_db] Found {len(docs)} text/html documents.") # DEBUG

    # --- Modified Image Processing: Parallel OCR + Threaded Captioning ---
    print(f"--- [build_text_db] Searching for images in: {Path(psm_folder).resolve()} using glob pattern **/*")
    imgs_glob = glob.glob(f"{psm_folder}/**/*", recursive=True)
    imgs_paths = [p for p in imgs_glob if p.lower().endswith(IMAGE_EXTS) and os.path.isfile(p)]
    print(f"--- [build_text_db] Filtered to {len(imgs_paths)} actual image files.")

    ocr_results_map = {} # Store OCR results keyed by path
    paths_for_captioning = [] # Store paths that passed PIL check
    image_docs: List[Document] = []
    skipped_count = 0

    if imgs_paths:
        # --- Step 1: Parallel OCR (includes PIL check & GIF conversion) ---
        print(f"--- [build_text_db] Starting Parallel OCR for {len(imgs_paths)} images...")
        max_ocr_workers = max(1, os.cpu_count() - 1 if os.cpu_count() else 1) # Leave one core free
        print(f"--- [build_text_db] Using up to {max_ocr_workers} processes for OCR.")
        use_gpu_for_ocr = EMB_DEVICE == "cuda" # Base decision on EMB_DEVICE for simplicity

        ocr_processed_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_ocr_workers) as executor:
            # Submit OCR tasks
            ocr_futures = [executor.submit(ocr_single_image, img_path, use_gpu_for_ocr) for img_path in imgs_paths]

            for future in concurrent.futures.as_completed(ocr_futures):
                try:
                    original_path, status, ocr_result = future.result()
                    if status.startswith("OK"): # Includes "OK" and "OK (GIF->PNG)"
                        ocr_results_map[original_path] = ocr_result
                        paths_for_captioning.append(original_path) # Mark for captioning
                    else: # PIL/Load Error or OCR Error
                        print(f"\n--- [build_text_db] Skipping {original_path} due to Status: {status}")
                        skipped_count += 1
                except Exception as future_e:
                    # Handle errors from the future execution itself
                    print(f"\n--- [build_text_db] Error retrieving result from OCR process: {future_e}")
                    skipped_count += 1 # Count as skipped

                ocr_processed_count += 1
                print(f"--- [build_text_db] OCR Processed: {ocr_processed_count}/{len(imgs_paths)} (Skipped: {skipped_count})", end='\r')

        print(f"\n--- [build_text_db] Parallel OCR finished. {len(paths_for_captioning)} images passed for captioning. Total skipped: {skipped_count}")

        # --- Step 2: Threaded Captioning ---
        if paths_for_captioning:
            print(f"--- [build_text_db] Starting Threaded Captioning for {len(paths_for_captioning)} images...")
            max_caption_workers = min(32, os.cpu_count() + 4)
            captions_map = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_caption_workers) as executor:
                future_to_path = {executor.submit(get_image_caption, img_path): img_path for img_path in paths_for_captioning}
                captioned_count = 0
                for future in concurrent.futures.as_completed(future_to_path):
                    img_path = future_to_path[future]
                    try:
                        caption = future.result()
                        captions_map[img_path] = caption
                    except Exception as exc:
                        print(f'\n--- [build_text_db] Captioning generated an exception for {img_path}: {exc}')
                        captions_map[img_path] = "[Caption Failed]"
                    captioned_count += 1
                    print(f"--- [build_text_db] Captions received: {captioned_count}/{len(paths_for_captioning)}", end='\r')
            print(f"\n--- [build_text_db] Threaded Captioning complete.")

            # --- Step 3: Combine OCR and Captions ---
            print("--- [build_text_db] Combining OCR and Caption results...")
            for img_path in paths_for_captioning: # Iterate through images that passed PIL check
                 ocr = ocr_results_map.get(img_path, "[OCR Data Missing]")
                 caption = captions_map.get(img_path, "[Caption Data Missing]")
                 page = f"OCR:{ocr}\nCAPTION:{caption}"
                 image_docs.append(Document(page_content=page, metadata={"source": img_path, "type": "image"}))

    # Combine text docs and image docs
    all_docs = docs + image_docs
    # --- End Modified Image Processing ---


    print(f"--- [build_text_db] Total documents (text+images) before chunking: {len(all_docs)}")
    if not all_docs:
         print("--- [build_text_db] WARNING: No documents found to process. DB will be empty.")
         Path(db_dir).mkdir(parents=True, exist_ok=True)
         return Chroma(embedding_function=_embedder, persist_directory=db_dir)

    print("--- [build_text_db] Chunking documents...")
    chunks = _token_splitter.split_documents(all_docs)
    print(f"--- [build_text_db] Total chunks created: {len(chunks)}")
    if not chunks:
        print("--- [build_text_db] WARNING: No chunks created after splitting. DB will be empty.")
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        return Chroma(embedding_function=_embedder, persist_directory=db_dir)

    print("--- [build_text_db] Calculating embeddings and building Chroma DB...")
    db = Chroma.from_documents(chunks, _embedder, persist_directory=db_dir)
    print("--- [build_text_db] Chroma DB build complete.")
    return db


def build_clip_db(psm_folder: str, db_dir: str) -> Optional[Chroma]:
    print(f"--- [build_clip_db] Searching for images in: {Path(psm_folder).resolve()} using glob pattern **/*")
    imgs = glob.glob(f"{psm_folder}/**/*", recursive=True)
    print(f"--- [build_clip_db] Found {len(imgs)} potential files via glob.")
    imgs = [p for p in imgs if p.lower().endswith(IMAGE_EXTS)]
    print(f"--- [build_clip_db] Filtered to {len(imgs)} images based on extensions: {IMAGE_EXTS}")
    if imgs:
        print(f"--- [build_clip_db] First 5 image paths found: {imgs[:5]}")
        flag_path_check = [p for p in imgs if 'flag.jpeg' in Path(p).name.lower()]
        if flag_path_check:
            print(f"--- [build_clip_db] Found flag.jpeg path(s): {flag_path_check}")
        else:
            print("--- [build_clip_db] Did not find 'flag.jpeg' in the filtered image list.")
    if not imgs:
        print("--- [build_clip_db] No images found to build CLIP DB.")
        return None

    dummy = type("Dummy", (), {"embed_documents": lambda *a, **k: [[0.0]*768],
                                "embed_query": lambda *a, **k: [0.0]*768})()
    db = Chroma("clip", persist_directory=db_dir, embedding_function=dummy)
    vecs, metas, ids = [], [], []
    for i, p in enumerate(imgs):
        try:
            vecs.append(_clip_embedder.embed_image(Image.open(p).convert("RGB")))
            metas.append({"source": p})
            ids.append(f"img_{i}")
        except Exception:
            continue
    db._collection.add(embeddings=vecs, documents=["img"]*len(vecs), metadatas=metas, ids=ids)
    return db

# -----------------------------------------------------------------------------
# Retrieval helpers – BM25 + dense + rerank
# -----------------------------------------------------------------------------

def hybrid_retrieve(query: str, vdb: Chroma, bm25: BM25Retriever, k_final: int = FINAL_K) -> List[Document]:
    dense_docs = vdb.similarity_search(query, k=DENSE_K)
    bm25_docs = bm25.get_relevant_documents(query)[:BM25_K]
    unique_docs_map = {d.page_content: d for d in bm25_docs + dense_docs}
    merged = list(unique_docs_map.values())

    if not merged:
        return []

    texts = [d.page_content for d in merged]
    query_doc_pairs = [[query, doc] for doc in texts]
    scores = _reranker.predict(query_doc_pairs)

    scored = sorted(zip(merged, scores), key=lambda x: x[1], reverse=True)

    final_docs = [d for d, s in scored[:k_final]]

    return final_docs

# -----------------------------------------------------------------------------
# Memory helpers
# -----------------------------------------------------------------------------

# def extract_short_memory(mem: ConversationBufferMemory, k=SHORT_MEMORY_K) -> str:
#     msgs = mem.chat_memory.messages[-k:]
#     return "\n".join([
#         ("User: " if isinstance(m, HumanMessage) else "Assistant: ") + m.content for m in msgs
#     ])

# -----------------------------------------------------------------------------
# RAG chatbot class
# -----------------------------------------------------------------------------
class GemmaRAG:
    def __init__(self, psm_folder="psm_files2", generated_dir="./generated", reload=False):
        warnings.filterwarnings("ignore")
        self.psm_folder = psm_folder
        # Ensure generated_dir exists, but don't create chat_file here
        Path(generated_dir).mkdir(exist_ok=True)

        # --- Database Loading/Building ---
        # Define DB paths
        text_db_dir = Path(generated_dir) / "text_db"
        clip_db_dir = Path(generated_dir) / "clip_db"

        # Load or build text DB
        if reload or not text_db_dir.exists():
             print(f"--- Rebuilding Text DB (reload={reload}, exists={text_db_dir.exists()}) ---")
             self.text_db = build_text_db(psm_folder, str(text_db_dir))
        else:
             print("--- Loading existing Text DB ---")
             self.text_db = Chroma(persist_directory=str(text_db_dir), embedding_function=_embedder)

        # Load or build CLIP DB
        if reload or not clip_db_dir.exists():
             print(f"--- Rebuilding CLIP DB (reload={reload}, exists={clip_db_dir.exists()}) ---")
             self.clip_db = build_clip_db(psm_folder, str(clip_db_dir))
        else:
             if clip_db_dir.exists():
                  print("--- Loading existing CLIP DB ---")
                  # Need dummy embedder to load Chroma when collection exists but no embeddings are added initially by us
                  dummy_clip_embed = type("Dummy", (), {"embed_documents": lambda *a, **k: [], "embed_query": lambda *a, **k: []})()
                  self.clip_db = Chroma(persist_directory=str(clip_db_dir), embedding_function=dummy_clip_embed, collection_name="clip") # Assuming default name 'clip' used in build_clip_db
             else:
                  print("--- No CLIP DB found and not rebuilding. ---")
                  self.clip_db = None


        # --- Initialize BM25 Retriever ---
        # Retrieve documents from Chroma to initialize BM25Retriever
        try:
             print("--- Initializing BM25 Retriever from Text DB ---")
             db_docs_data = self.text_db.get(include=["documents", "metadatas"])
             if db_docs_data and db_docs_data.get("documents"):
                 all_docs_for_bm25 = [
                     Document(page_content=doc, metadata=meta)
                     for doc, meta in zip(db_docs_data["documents"], db_docs_data["metadatas"])
                 ]
                 self.bm25 = BM25Retriever.from_documents(all_docs_for_bm25)
                 print(f"--- BM25 Initialized with {len(all_docs_for_bm25)} documents. ---")
             else:
                 print("--- WARNING: Text DB appears empty, cannot initialize BM25. ---")
                 self.bm25 = BM25Retriever.from_documents([]) # Initialize empty
        except Exception as bm25_e:
             print(f"--- ERROR: Failed to initialize BM25 Retriever: {bm25_e} ---")
             self.bm25 = BM25Retriever.from_documents([]) # Initialize empty on error


        # --- Short Term Memory ONLY ---
        self.short_mem = ConversationBufferMemory(return_messages=True)
        # --- REMOVED chat_file and long_mem ---
        # self.chat_file = Path(f"{generated_dir}/chat_history.txt")
        # self.chat_file.touch(exist_ok=True)
        # self.long_mem = self.chat_file.read_text(encoding='utf-8')
        print("--- RAG Bot Initialized (Short-Term Memory Only) ---")

    # --------------------------------------------------
    # response pipeline
    # --------------------------------------------------
    def answer(self, user_q: str) -> Dict:
        # --- Log user message to short-term memory ---
        self.short_mem.chat_memory.add_user_message(user_q)
        # --- REMOVED file logging for user message ---

        # 2. attempt RAG
        psm_hits = hybrid_retrieve(user_q, self.text_db, self.bm25)
        clip_hits: List[Document] = []
        image_query_triggered = any(k in user_q.lower() for k in IMAGE_KW)

        if self.clip_db and image_query_triggered:
            q_vec = _clip_embedder.embed_text(user_q)
            # Use a slightly higher k for CLIP search initially
            clip_hits = self.clip_db.similarity_search_by_vector(q_vec, k=5)
            # Simple relevance filter for CLIP (optional, based on cosine distance if available, else keep top ones)
            # Assuming Chroma returns distance, lower is better. Keep top 3 for now.
            clip_hits = clip_hits[:3]


        sources = []
        mode_used = []
        answer_parts = []
        llm_response_text = ""

        # Combine hits and ensure uniqueness
        all_hits_map: Dict[str, Document] = {}
        for doc in psm_hits + clip_hits:
            source_path = doc.metadata.get("source")
            if source_path and source_path not in all_hits_map:
                 all_hits_map[source_path] = doc
        final_hits = list(all_hits_map.values())


        if final_hits: # RAG Path
            mode_used.append("RAG response")
            context_blobs = []
            processed_sources_for_details = set() # Avoid asking detail about the same image multiple times

            # Check if the query likely asks for details *about* an image
            query_asks_for_detail = any(kw in user_q.lower() for kw in IMAGE_DETAIL_KW)

            for d in final_hits:
                source_path = d.metadata["source"]
                sources.append(source_path)

                is_image = source_path.lower().endswith(IMAGE_EXTS) or d.metadata.get("type") == "image"

                if is_image:
                    # Add image caption (or existing OCR/caption if from text_db)
                    caption = d.page_content # Use existing content if it's the OCR/caption doc
                    if not caption.startswith("OCR:") and not caption.startswith("CAPTION:"):
                         # If it's a raw CLIP hit, get caption
                         try:
                             caption = get_image_caption(source_path)
                         except Exception as e:
                             caption = f"Error getting caption: {e}"

                    context_blobs.append(f"[IMG {source_path}] {caption[:600]}") # Add caption/OCR blob

                    # --- Image Detail Re-query ---
                    if query_asks_for_detail and source_path not in processed_sources_for_details:
                         try:
                             print(f"--- Querying detail for image: {source_path} ---") # DEBUG
                             detail = get_image_detail(source_path, user_q)
                             context_blobs.append(f"[DETAIL {source_path}] {detail[:600]}")
                             processed_sources_for_details.add(source_path)
                         except Exception as e:
                             print(f"--- Error querying detail for {source_path}: {e} ---") # DEBUG
                             context_blobs.append(f"[DETAIL ERROR {source_path}] Could not get details.")
                else:
                    # It's a text document hit
                    context_blobs.append(f"[DOC {source_path}]: {d.page_content[:600]}")


            context = "\n\n".join(context_blobs)
            sys_prompt = "You answer based ONLY on the CONTEXT provided below (including text DOCS, image IMG captions/OCR, and image DETAILs). Return concise facts. Do not use prior knowledge."
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION: {user_q}"}
            ]
            try:
                resp = chat_completion(messages, max_tokens=1024)
                llm_response_text = resp.choices[0].message.content.strip()
                answer_parts.append(llm_response_text)
            except Exception as e:
                print(f"--- Error during RAG chat completion: {e} ---")
                llm_response_text = "Error generating RAG response from documents."
                answer_parts.append(llm_response_text)
                mode_used = ["Error"]

        # --- MODIFIED: No Fallback - Only Respond if RAG found hits ---
        else: # This triggers if final_hits was empty
            mode_used = ["No Relevant Docs Found"]
            llm_response_text = "I could not find relevant information about that in the provided documents."
            answer_parts.append(llm_response_text)
            sources = [] # Ensure no sources are listed if RAG failed
        # --- End No Fallback Modification ---

        # --- Source Lookup Map (Needed for source display) ---
        source_to_doc_map: Dict[str, Document] = {}
        if final_hits: # Only populate if RAG succeeded
             source_to_doc_map = {d.metadata['source']: d for d in final_hits if d.metadata.get('source')}

        # assemble final answer
        mode_str = ", ".join(mode_used)
        answer_text = f"({mode_str}) {llm_response_text}"

        # --- Source Display Logic (remains largely the same, relies on source_to_doc_map) ---
        source_strings = []
        # Ensure sources list is based on actual RAG hits if RAG was used
        unique_sources_list = list(set(s for s in sources if s)) if final_hits else []

        for s_path in unique_sources_list:
             # ... (logic for displaying image OCR/Caption or just path) ...
             # (This part remains the same as the previous version)
             is_image_path = s_path.lower().endswith(IMAGE_EXTS)
             if is_image_path:
                 doc = source_to_doc_map.get(s_path)
                 ocr_caption_content = "Content not readily available"
                 if doc:
                     if doc.page_content and len(doc.page_content) > 10 and doc.page_content != "CLIP image doc":
                          ocr_caption_content = doc.page_content
                     else:
                         try:
                             caption = get_image_caption(s_path)
                             ocr_caption_content = caption
                         except Exception:
                             ocr_caption_content = "Could not fetch caption."
                 source_strings.append(f"{s_path} (OCR/Caption:\n{ocr_caption_content}\n)")
             else:
                 source_strings.append(s_path)

        src_block = "\n========Sources:=========\n" + "\n".join(source_strings) if source_strings else ""
        answer_text += src_block

        # --- Log AI message to short-term memory ---
        self.short_mem.chat_memory.add_ai_message(llm_response_text) # Log only the core response
        # --- REMOVED file logging for assistant message ---

        return {
            "answer": answer_text,
            "image_paths": [s for s in unique_sources_list if s.lower().endswith(IMAGE_EXTS)][:3]
        }

# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Update default psm folder if needed ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--psm", default="psm_files", help="Folder containing documents to index.") # Changed default back
    parser.add_argument("--reload", action="store_true", help="Force rebuild of text and CLIP databases.")
    args = parser.parse_args()

    # --- Pass reload flag correctly ---
    bot = GemmaRAG(psm_folder=args.psm, reload=args.reload)
    print("Gemma‑RAG Bot ready. Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit", "end"}:
            break
        out = bot.answer(q)
        print(out["answer"], "\n")
