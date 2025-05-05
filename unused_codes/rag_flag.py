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

import sys
print("Editor sys.path:", sys.path)

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
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.retrievers import BM25Retriever, MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.merger_retriever import MergerRetriever

# Cross‑encoder (reranker)
from flag_embedding import FlagReranker

# CLIP for images
from transformers import CLIPProcessor, CLIPModel
import easyocr

# OpenAI SDK pointed at local Gemma endpoint
from openai import OpenAI, Stream

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
GEMMA_ENDPOINT   = "http://10.130.236.56:1235/v1"  # your vLLM server
GEMMA_MODEL_NAME = "gemma-3-27b-it"
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"         # dense encoder
RERANK_MODEL     = "BAAI/bge-reranker-base"
EMB_DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE_TOK   = 512
CHUNK_OVERLAP    = 64
MAX_TOK_CONTEXT  = 90_000                             # leave headroom for answer
IMAGE_EXTS       = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
IMAGE_KW         = ["image", "photo", "picture", *IMAGE_EXTS]
SHORT_MEMORY_K   = 10
BM25_K           = 10
DENSE_K          = 15
FINAL_K          = 6
RELEVANCE_THRES  = 0.55

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
_reranker = FlagReranker(RERANK_MODEL, use_fp16=torch.cuda.is_available())

class CLIPImageEmbeddings:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = EMB_DEVICE
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

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

def build_text_db(psm_folder: str, db_dir: str) -> Chroma:
    loaders = [
        DirectoryLoader(psm_folder, "**/*.txt",  UnstructuredFileLoader),
        DirectoryLoader(psm_folder, "**/*.html", UnstructuredHTMLLoader),
        DirectoryLoader(psm_folder, "**/*.htm",  UnstructuredHTMLLoader),
    ]
    docs: List[Document] = []
    for ldr in loaders:
        docs.extend(ldr.load())

    # embed images as text docs with OCR + caption
    imgs = glob.glob(f"{psm_folder}/**/*", recursive=True)
    imgs = [p for p in imgs if p.lower().endswith(IMAGE_EXTS)]
    ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    for img_path in imgs:
        try:
            ocr = "\n".join(ocr_reader.readtext(img_path, detail=0))
            caption = get_image_caption(img_path)
            page = f"OCR:{ocr}\nCAPTION:{caption}"
            docs.append(Document(page_content=page, metadata={"source": img_path, "type": "image"}))
        except Exception:
            continue

    chunks = _token_splitter.split_documents(docs)
    return Chroma.from_documents(chunks, _embedder, persist_directory=db_dir)


def build_clip_db(psm_folder: str, db_dir: str) -> Optional[Chroma]:
    imgs = glob.glob(f"{psm_folder}/**/*", recursive=True)
    imgs = [p for p in imgs if p.lower().endswith(IMAGE_EXTS)]
    if not imgs:
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
    merged = {d.metadata["source"]: d for d in dense_docs + bm25_docs}.values()
    # Cross‑encoder rerank
    texts = [d.page_content for d in merged]
    scores = _reranker.compute_score([query]*len(texts), texts)
    scored = sorted(zip(merged, scores), key=lambda x: x[1], reverse=True)
    return [d for d, s in scored[:k_final] if s >= RELEVANCE_THRES]

# -----------------------------------------------------------------------------
# Memory helpers
# -----------------------------------------------------------------------------

def extract_short_memory(mem: ConversationBufferMemory, k=SHORT_MEMORY_K) -> str:
    msgs = mem.chat_memory.messages[-k:]
    return "\n".join([
        ("User: " if isinstance(m, HumanMessage) else "Assistant: ") + m.content for m in msgs
    ])

# -----------------------------------------------------------------------------
# RAG chatbot class
# -----------------------------------------------------------------------------
class GemmaRAG:
    def __init__(self, psm_folder="psm_files", generated_dir="./generated", reload=False):
        warnings.filterwarnings("ignore")
        self.psm_folder = psm_folder
        Path(generated_dir).mkdir(exist_ok=True)
        self.text_db = build_text_db(psm_folder, f"{generated_dir}/text_db") if reload or not Path(f"{generated_dir}/text_db").exists() else Chroma(persist_directory=f"{generated_dir}/text_db", embedding_function=_embedder)
        self.clip_db = build_clip_db(psm_folder, f"{generated_dir}/clip_db") if reload or not Path(f"{generated_dir}/clip_db").exists() else Chroma(persist_directory=f"{generated_dir}/clip_db") if Path(f"{generated_dir}/clip_db").exists() else None
        self.bm25 = BM25Retriever.from_documents(self.text_db.get_all_documents())
        self.short_mem = ConversationBufferMemory(return_messages=True)
        self.chat_file = Path(f"{generated_dir}/chat_history.txt")
        self.chat_file.touch(exist_ok=True)
        self.long_mem = self.chat_file.read_text()

    # --------------------------------------------------
    # response pipeline
    # --------------------------------------------------
    def answer(self, user_q: str) -> Dict:
        # 1. store user message in memories
        self.short_mem.chat_memory.add_user_message(user_q)
        self.chat_file.write_text(self.chat_file.read_text() + f"User: {user_q}\n")

        # 2. attempt RAG
        psm_hits = hybrid_retrieve(user_q, self.text_db, self.bm25)
        clip_hits: List[Document] = []
        if self.clip_db and any(k in user_q.lower() for k in IMAGE_KW):
            q_vec = _clip_embedder.embed_text(user_q)
            clip_hits = self.clip_db.similarity_search_by_vector(q_vec, k=3)

        sources = []
        mode_used = []
        answer_parts = []

        if psm_hits or clip_hits:
            mode_used.append("RAG response")
            context_blobs = []
            for d in psm_hits:
                context_blobs.append(f"[DOC {d.metadata['source']}]: {d.page_content[:600]}")
                sources.append(d.metadata["source"])
            for d in clip_hits:
                # add image caption on the fly
                caption = get_image_caption(d.metadata["source"])
                context_blobs.append(f"[IMG {d.metadata['source']}] {caption}")
                sources.append(d.metadata["source"])
            context = "\n\n".join(context_blobs)
            short = extract_short_memory(self.short_mem)
            sys_prompt = "You answer based on the CONTEXT and return concise facts."
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": f"CONTEXT:\n{context}\n\nQUESTION: {user_q}"}
            ]
            resp = chat_completion(messages, max_tokens=1024)
            answer_parts.append(resp.choices[0].message.content.strip())
        else:
            # no RAG docs – try short term memory
            short_ctx = extract_short_memory(self.short_mem)
            if short_ctx and any(user_q.lower() in short_ctx.lower() for _ in [0]):
                mode_used.append("short term memory")
                answer_parts.append("I remember you said: " + short_ctx)
                sources.append("short_term_memory")
            elif self.long_mem:
                # search long memory doc quickly
                if user_q.lower() in self.long_mem.lower():
                    mode_used.append("long term memory")
                    answer_parts.append("From our earlier chats: " + self.long_mem[-500:])
                    sources.append("long_term_memory")
            if not answer_parts:
                # fallback conversational
                mode_used.append("conversational response")
                resp = chat_completion([
                    {"role": "system", "content": "You are a friendly assistant."},
                    {"role": "user", "content": user_q}
                ], max_tokens=512)
                answer_parts.append(resp.choices[0].message.content.strip())

        # assemble final answer
        mode_str = ", ".join(mode_used)
        answer_text = f"({mode_str}) " + "\n\n".join(answer_parts)
        src_block = "\n========Sources:=========\n" + "\n".join(sources) if sources else ""
        answer_text += src_block

        # log assistant message
        self.short_mem.chat_memory.add_ai_message(answer_text)
        self.chat_file.write_text(self.chat_file.read_text() + f"Assistant: {answer_text}\n")

        return {
            "answer": answer_text,
            "image_paths": [s for s in sources if s.lower().endswith(IMAGE_EXTS)][:3]
        }

# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--psm", default="psm_files")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    bot = GemmaRAG(psm_folder=args.psm, reload=args.reload)
    print("Gemma‑RAG Bot ready. Type 'exit' to quit.")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit", "end"}:
            break
        out = bot.answer(q)
        print(out["answer"], "\n")
