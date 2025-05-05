# RAG.py

import os
import glob
import time
import shutil
import warnings
import base64
import json
import gc
import stat
from mimetypes import guess_type
from dotenv import load_dotenv
from PIL import Image
import torch
import argparse
import requests  # Added for HTTP requests

# LangChain & Other Dependencies
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    DirectoryLoader, UnstructuredFileLoader, UnstructuredHTMLLoader
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

import easyocr
from pydantic import BaseModel, Field
from typing import List

# Optional: CLIP for direct image embeddings
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------------------------
# Structured JSON output
# -----------------------------------------------------------------------------
class QAFormat(BaseModel):
    answer: str = Field(..., description="Main answer to the user query")
    image_paths: List[str] = Field(default_factory=list, description="List of relevant image paths (max 3)")

# -----------------------------------------------------------------------------
# CLIPImageEmbeddings (for image retrieval)
# -----------------------------------------------------------------------------
class CLIPImageEmbeddings:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        return outputs[0].cpu().tolist()

    def embed_image(self, pil_image: Image.Image):
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        return outputs[0].cpu().tolist()

# -----------------------------------------------------------------------------
# EasyOCR Image Loader (text-based)
# -----------------------------------------------------------------------------
class EasyOCRImageLoader:
    def __init__(self, path: str, languages=["en"], device="cpu", client=None):
        self.path = path
        self.languages = languages
        self.reader = easyocr.Reader(self.languages, gpu=(device=="cuda"))
        self.client = client  # Added client parameter

    def _is_image_file(self, fn: str):
        ext = os.path.splitext(fn.lower())[1]
        return ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    def _load_img(self, fp: str):
        ocr_results = self.reader.readtext(fp, detail=0)
        text_ocr = "\n".join(ocr_results) if ocr_results else ""
        caption = get_image_caption(fp, self.client)  # Pass client
        combined_for_embedding = f"OCR: {text_ocr}\nGPT caption: {caption}"
        return [Document(
            page_content=combined_for_embedding,
            metadata={
                "source": fp,
                "ocr_text": text_ocr,
                "gpt_caption": caption
            }
        )]

    def load(self):
        docs = []
        if os.path.isdir(self.path):
            for fn in os.listdir(self.path):
                fp = os.path.join(self.path, fn)
                if os.path.isfile(fp) and self._is_image_file(fn):
                    docs.extend(self._load_img(fp))
            return docs
        else:
            if os.path.isfile(self.path) and self._is_image_file(self.path):
                return self._load_img(self.path)
            return []

# -----------------------------------------------------------------------------
# Utility Functions: Manifest and Directory Deletion
# -----------------------------------------------------------------------------
def compute_psm_manifest(folder_path: str):
    mani = {}
    for root, dirs, files in os.walk(folder_path):
        for fn in files:
            fullp = os.path.join(root, fn)
            mani[fullp] = os.path.getmtime(fullp)
    return mani

def load_manifest_psm(manifest_path):
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            try:
                return json.loads(f.read().strip())
            except:
                pass
    return {}

def save_manifest_psm(mani, manifest_path):
    with open(manifest_path, "w") as f:
        json.dump(mani, f)

def manifest_changed(m1, m2):
    return m1 != m2

def handle_remove_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        raise

def safe_rmtree(path, max_attempts=5, delay=2):
    attempt = 0
    while attempt < max_attempts:
        try:
            gc.collect()
            shutil.rmtree(path, onerror=handle_remove_error)
            return
        except Exception as e:
            time.sleep(delay)
            attempt += 1
    if os.name == "nt":
        os.system(f"rmdir /s /q \"{path}\"")
        if not os.path.exists(path):
            return
    raise Exception(f"Could not delete '{path}' after {max_attempts} attempts.")

def remove_directory(path):
    if not os.path.exists(path):
        return
    try:
        safe_rmtree(path)
    except Exception as e:
        new_path = path + "_old"
        try:
            os.rename(path, new_path)
        except Exception as e2:
            raise Exception(f"Failed to delete or rename '{path}': {e2}")

# -----------------------------------------------------------------------------
# Local API Configuration and Call Function
# -----------------------------------------------------------------------------
API_URL = "http://10.130.236.73:1234/v1/chat/completions"  # Replace with your local API endpoint

def call_local_api(messages, max_tokens=200, temperature=0.3):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llava-v1.5-7b:3",  # Updated model name
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {e}")

# -----------------------------------------------------------------------------
# GPT-4o Image Captioning & Detail Re-query
# -----------------------------------------------------------------------------
def get_image_caption(filepath, client):
    try:
        prompt = (
            f"File: {filepath}\n"
            "Describe this image in a detailed but concise way. "
            "Mention shapes, colors, objects. Do not mention the file path."
        )
        with open(filepath, "rb") as f:
            data = f.read()
        b64_str = base64.b64encode(data).decode("utf-8")
        mime, _ = guess_type(filepath) or ("image/png", None)
        # Since the local API likely does not handle image URLs, we'll pass the prompt and the base64 string as separate messages
        messages = [
            {"role": "system", "content": "You are a creative, observant image describer."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"data:{mime};base64,{b64_str}"}
        ]
        response = call_local_api(messages, max_tokens=200, temperature=0.7)
        return response
    except Exception as e:
        return f"Error describing image: {e}"

def get_image_detail_response(image_path, user_query, client):
    try:
        prompt = f"File: {image_path}\nAnswer this question about the image: {user_query}"
        with open(image_path, "rb") as f:
            data = f.read()
        b64_str = base64.b64encode(data).decode("utf-8")
        mime, _ = guess_type(image_path) or ("image/png", None)
        messages = [
            {"role": "system", "content": "You are a helpful assistant skilled at extracting image details."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"data:{mime};base64,{b64_str}"}
        ]
        response = call_local_api(messages, max_tokens=300, temperature=0.7)
        return response
    except Exception as e:
        return f"Error retrieving image detail: {e}"

# -----------------------------------------------------------------------------
# Summarize Chat History at Startup
# -----------------------------------------------------------------------------
def summarize_chat_history_at_start(chat_history, client, chat_file):
    if not chat_history:
        return ""
    lines = chat_history.splitlines()
    pivot = int(len(lines) * 0.8)
    older = lines[:pivot]
    newer = lines[pivot:]
    older_text = "\n".join(older)
    prompt = (
        "Below is an older portion of chat history. Summarize it to keep only key details "
        "and personal facts without mentioning file paths or PSM document details:\n\n"
        f"{older_text}\n\nNow produce a concise summary (a few sentences or bullet points)."
    )
    try:
        messages = [
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": prompt}
        ]
        summary = call_local_api(messages, max_tokens=400, temperature=0.5)
    except Exception as e:
        print(f"Error summarizing older lines: {e}")
        return chat_history  # Return original if summarization fails
    summary_lines = [
        "Assistant: SUMMARY BLOCK:\n",
        summary + "\n",
        "Assistant: END SUMMARY BLOCK\n"
    ]
    new_chat = "".join(summary_lines) + "\n".join(newer)
    try:
        with open(chat_file, "w", encoding="utf-8") as f:
            f.write(new_chat)
        print("Older lines summarized and appended to chat_history as summary block.")
    except Exception as e:
        print(f"Error writing summary to chat_history.txt: {e}")
    return new_chat

# -----------------------------------------------------------------------------
# Detect "Generic" Responses & Fallback
# -----------------------------------------------------------------------------
def is_generic(ans: str):
    triggers = ["i don't know", "no info", "not mentioned", "doesn't provide", "i do not recall", "i have no record"]
    return any(t in ans.lower() for t in triggers)

def call_gpt_no_context(query: str, client):
    try:
        messages = [
            {"role": "system", "content": "You are a friendly assistant with general world knowledge."},
            {"role": "user", "content": query}
        ]
        response = call_local_api(messages, max_tokens=2000, temperature=0.9)
        return response
    except Exception as e:
        return f"Error calling local model no context: {e}"

def augment_fallback(query, ans):
    if any(k in query.lower() for k in ["cat", "dog", "rabbit", "name"]):
        return ans + " By the way, that's a cool name!"
    return ans

# -----------------------------------------------------------------------------
# Retrieval Helpers
# -----------------------------------------------------------------------------
def retrieve_with_threshold(db, query, k=3, threshold=0.55):
    if not db:
        return []
    results = db.similarity_search_with_score(query, k=k)
    filtered = []
    for doc, score in results:
        if score >= threshold:
            doc.metadata["score"] = score
            filtered.append(doc)
    return filtered

def retrieve_images_clip(db, query_text, clip_embedder, k=3):
    if not db:
        return []
    query_vec = clip_embedder.embed_text(query_text)
    results = db.similarity_search_by_vector(query_vec, k=k)
    return results

def needs_image_requery(doc: Document, user_q: str, image_keywords: List[str]) -> bool:
    """
    Return True if this doc is an image (by file extension) and
    either the user query or the doc's content contains keywords suggesting extra image details are desired.
    """
    src = doc.metadata.get("source", "").lower()
    if not src.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        return False
    if any(kw in user_q.lower() for kw in image_keywords):
        return True
    # A naive check: if any word in the user query also appears in the doc content, assume relevance.
    user_words = set(user_q.lower().split())
    doc_words = set(doc.page_content.lower().split())
    if user_words.intersection(doc_words):
        return True
    return False

# -----------------------------------------------------------------------------
# LLM Helper: Build Final JSON Answer
# -----------------------------------------------------------------------------
def build_final_json_answer(system_prompt: str, context: str, client) -> QAFormat:
    try:
        template = f"""
You are a helpful assistant. Output **only** valid JSON following this schema without any code fences or markdown formatting:
{{
  "answer": "...",
  "image_paths": []
}}
If no images are relevant, "image_paths" is empty.

[CONTEXT]
{context}
Do not include any additional text, code fences, or markdown formatting. Output only the JSON object as specified.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": template}
        ]

        response = call_local_api(messages, max_tokens=200, temperature=0.3)
        raw = response.strip()

        # Remove code fences if present
        if raw.startswith("```") and "json" in raw:
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                raw = raw[start:end+1]
            except:
                pass

        # Attempt to parse the JSON
        try:
            data = json.loads(raw)
            # Validate against QAFormat
            return QAFormat(**data)
        except json.JSONDecodeError as e:
            # If JSON parsing fails, return the raw answer with empty image_paths
            return QAFormat(answer=raw, image_paths=[])
    except Exception as e:
        # If any other exception occurs, return a generic error message
        return QAFormat(answer="I'm sorry, something went wrong while processing your request.", image_paths=[])

# -----------------------------------------------------------------------------
# RAG Chatbot Class
# -----------------------------------------------------------------------------
class RAG:
    def __init__(self, 
                 psm_folder="PSM", 
                 chat_history="", 
                 reload_psm=False, 
                 reload_img=False, 
                 device=None):
        # Load environment variables
        load_dotenv()
        self.client = None  # Not needed anymore
        warnings.filterwarnings("ignore")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device.upper()}")

        # Directories and files
        self.PSM_FOLDER = psm_folder
        self.CHAT_FILE = "./generated/chat_history.txt"
        self.PSM_DB_DIR = "./generated/chroma_db_psm"
        self.PSM_IMAGE_DB_DIR = "./generated/chroma_db_images"
        self.CHAT_DB_DIR = "./generated/chroma_db_chat"
        self.PSM_MANIFEST = "./generated/manifest_psm.json"
        self.MAX_CHAT_LINES = 10000
        self.SHORT_TERM_COUNT = 10
        self.RELEVANCE_THRESHOLD = 0.55
        self.IMAGE_KEYWORDS = ["image", "photo", "picture", ".png", ".jpg", ".jpeg", ".gif", ".bmp"]

        # Initialize or load chat history
        if chat_history:
            self.chat_history = chat_history
        else:
            self.chat_history = self.load_chat_history()

        # Summarize chat history if necessary and write back to chat_history.txt
        self.chat_history = summarize_chat_history_at_start(self.chat_history, self.client, self.CHAT_FILE)

        # Initialize databases
        self.psm_db = self.build_or_load_psm_db(force=reload_psm)
        self.clip_db = self.build_or_load_image_db(force=reload_img)
        self.clip_embedder = CLIPImageEmbeddings() if self.clip_db else None
        self.chat_db = self.build_chat_db()
        self.chat_docs = self.load_chat_doc()
        self.full_history_doc = self.chat_docs[0] if self.chat_docs else None

        # Initialize session memory
        self.session_memory = ConversationBufferMemory(memory_key="session", return_messages=True)

    def load_chat_history(self):
        if not os.path.exists(self.CHAT_FILE):
            print("No chat_history.txt => starting with empty history.")
            return ""
        with open(self.CHAT_FILE, "r", encoding="utf-8") as f:
            lines = f.read()
        print("Loaded chat history.")
        return lines

    def build_or_load_psm_db(self, force=False):
        cur_mani = compute_psm_manifest(self.PSM_FOLDER)
        old_mani = load_manifest_psm(self.PSM_MANIFEST)
        changed = force or manifest_changed(cur_mani, old_mani)
        if changed:
            print("PSM folder changed or forced => building new PSM (text) DB.")
            remove_directory(self.PSM_DB_DIR)
            if os.path.exists(self.PSM_MANIFEST):
                os.remove(self.PSM_MANIFEST)
            docs_text = DirectoryLoader(self.PSM_FOLDER, glob="**/*.txt", loader_cls=UnstructuredFileLoader).load()
            docs_html = DirectoryLoader(self.PSM_FOLDER, glob="**/*.html", loader_cls=UnstructuredHTMLLoader).load()
            docs_htm = DirectoryLoader(self.PSM_FOLDER, glob="**/*.htm", loader_cls=UnstructuredHTMLLoader).load()
            docs_csv = DirectoryLoader(self.PSM_FOLDER, glob="**/*.csv", loader_cls=UnstructuredFileLoader).load()
            docs_css = DirectoryLoader(self.PSM_FOLDER, glob="**/*.css", loader_cls=UnstructuredFileLoader).load()
            docs_js  = DirectoryLoader(self.PSM_FOLDER, glob="**/*.js",  loader_cls=UnstructuredFileLoader).load()
            imgs = glob.glob(os.path.join(self.PSM_FOLDER, "**/*.*"), recursive=True)
            imgs = [p for p in imgs if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
            docs_imgs = []
            for ip in imgs:
                loader = EasyOCRImageLoader(ip, ["en"], device=self.device, client=self.client)  # Pass client
                docs_imgs.extend(loader.load())
            all_docs = docs_text + docs_html + docs_htm + docs_csv + docs_css + docs_js + docs_imgs
            spl = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            splitted = spl.split_documents(all_docs)
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                          model_kwargs={"device": self.device})
            db = Chroma.from_documents(splitted, embed, persist_directory=self.PSM_DB_DIR)
            db.persist()
            save_manifest_psm(cur_mani, self.PSM_MANIFEST)
            print("PSM DB (text) built & saved.")
        else:
            print("PSM folder unchanged => loading existing PSM DB.")
            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                          model_kwargs={"device": self.device})
            db = Chroma(persist_directory=self.PSM_DB_DIR, embedding_function=embed)
        return db

    def build_or_load_image_db(self, force=False):
        if not os.path.exists(self.PSM_FOLDER):
            return None
        db_dir = self.PSM_IMAGE_DB_DIR
        # If DB exists and not forced, load existing DB.
        if not force and os.path.exists(db_dir):
            print("CLIP-based DB found, loading existing DB (skip building).")
            class DummyEmbeddings:
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return [[0.0]*768 for _ in texts]
                def embed_query(self, text: str) -> List[float]:
                    return [0.0]*768
            dummy = DummyEmbeddings()
            db = Chroma(collection_name="clip_image_store", persist_directory=db_dir, embedding_function=dummy)
            return db
        if force and os.path.exists(db_dir):
            remove_directory(db_dir)
        imgs = glob.glob(os.path.join(self.PSM_FOLDER, "**/*.*"), recursive=True)
        imgs = [p for p in imgs if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]
        if not imgs:
            print("No images found for CLIP DB.")
            return None
        class DummyEmbeddings:
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [[0.0]*768 for _ in texts]
            def embed_query(self, text: str) -> List[float]:
                return [0.0]*768
        dummy = DummyEmbeddings()
        db = Chroma(collection_name="clip_image_store", persist_directory=db_dir, embedding_function=dummy)
        clip_embedder = CLIPImageEmbeddings(device=self.device)
        docs = []
        vectors = []
        for i, ip in enumerate(imgs):
            if not os.path.isfile(ip):
                continue
            try:
                pil_img = Image.open(ip).convert("RGB")
            except:
                continue
            vec = clip_embedder.embed_image(pil_img)
            doc = Document(page_content="CLIP image doc", metadata={"source": ip})
            docs.append(doc)
            vectors.append(vec)
        if not docs:
            print("No valid images to embed for CLIP DB.")
            return None
        print(f"Building new CLIP DB with {len(docs)} images.")
        texts = [f"Image_{i}" for i in range(len(docs))]
        metadatas = [d.metadata for d in docs]
        ids = [f"img_{i}" for i in range(len(docs))]
        db._collection.add(embeddings=vectors, documents=texts, metadatas=metadatas, ids=ids)
        db.persist()
        return db

    def build_chat_db(self):
        remove_directory(self.CHAT_DB_DIR)
        if not self.chat_history:
            print("No chat_history => no Chat DB.")
            return None
        doc = Document(page_content=self.chat_history, metadata={"source": self.CHAT_FILE})
        spl = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        splitted = spl.split_documents([doc])
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={"device": self.device})
        chatdb = Chroma.from_documents(splitted, embed, persist_directory=self.CHAT_DB_DIR)
        chatdb.persist()
        print("Chat DB built from chat_history.\n")
        return chatdb

    def load_chat_doc(self):
        if not self.chat_history:
            print("No chat_history => returning empty.")
            return []
        return [Document(page_content=self.chat_history, metadata={"source": self.CHAT_FILE})]

    def Get_Rag_Response(self, user_q: str) -> dict:
        if user_q.lower() == "reset":
            self.session_memory.chat_memory.clear()
            # Write the reset action to chat_history.txt
            try:
                with open(self.CHAT_FILE, "a", encoding="utf-8") as f:
                    f.write("Assistant: Short-term memory has been cleared.\n")
            except Exception as e:
                print(f"Error writing reset to chat_history.txt: {e}")
            return {"answer": "Short-term memory has been cleared.", "image_paths": []}

        self.session_memory.chat_memory.add_user_message(user_q)
        self.chat_history += f"User: {user_q}\n"

        # Write user message to chat_history.txt
        try:
            with open(self.CHAT_FILE, "a", encoding="utf-8") as f:
                f.write(f"User: {user_q}\n")
        except Exception as e:
            print(f"Error writing user message to chat_history.txt: {e}")

        st_lines = self.session_memory.chat_memory.messages[-self.SHORT_TERM_COUNT:]
        st_msgs = []
        for msg in st_lines:
            if isinstance(msg, HumanMessage):
                st_msgs.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                st_msgs.append(f"Assistant: {msg.content}")
            else:
                st_msgs.append(f"Unknown: {msg.content}")
        short_hist = "\n".join(st_msgs) if st_msgs else "None"

        # Retrieve text-based docs from the PSM and chat DBs
        psm_hits = retrieve_with_threshold(self.psm_db, user_q, k=3, threshold=self.RELEVANCE_THRESHOLD)
        chat_hits = retrieve_with_threshold(self.chat_db, user_q, k=3, threshold=self.RELEVANCE_THRESHOLD)

        # Optionally perform CLIP-based retrieval if user query references image keywords
        clip_hits = []
        if self.clip_db and self.clip_embedder and any(kw in user_q.lower() for kw in self.IMAGE_KEYWORDS):
            clip_hits = retrieve_images_clip(self.clip_db, user_q, self.clip_embedder, k=3)

        # Re-query image details if needed
        image_details = []
        for doc in psm_hits:
            if needs_image_requery(doc, user_q, self.IMAGE_KEYWORDS):
                detail = get_image_detail_response(doc.metadata["source"], user_q, self.client)
                image_details.append(f"[Detail from {doc.metadata['source']}]: {detail}")
        image_details_context = "\n".join(image_details) if image_details else "No additional image details."

        final_system_prompt = (
            "You are PSM Bot, referencing PSM docs + chat_history (and image details if provided).\n"
            "Provide a concise, accurate answer in JSON with fields 'answer' and 'image_paths'.\n\n"
            f"Recent {self.SHORT_TERM_COUNT} messages:\n{short_hist}\n\nUser Query:\n{user_q}\n"
        )

        psm_context = "\n\n".join([f"[DOC: {doc.metadata.get('source','?')}] \n{doc.page_content[:600]}"
                                   for doc in psm_hits]) if psm_hits else "No PSM context."
        chat_context = "\n\n".join([f"[CHAT HIT]\n{doc.page_content[:600]}"
                                    for doc in chat_hits]) if chat_hits else "No chat context."
        clip_context = "No CLIP image context."
        if clip_hits:
            clip_context = "\n".join([f"[CLIP IMAGE: {doc.metadata.get('source','?')}]"
                                      for doc in clip_hits])
        combined_context = (
            f"[PSM doc context]\n{psm_context}\n\n"
            f"[Chat_history doc context]\n{chat_context}\n\n"
            f"[CLIP-based image context]\n{clip_context}\n\n"
            f"[Re-queried image details]\n{image_details_context}\n\n"
            f"[Full chat_history doc]\n{self.full_history_doc.page_content if self.full_history_doc else 'No full doc'}"
        )

        structured_answer = build_final_json_answer(final_system_prompt, combined_context, self.client)

        if is_generic(structured_answer.answer):
            fb = call_gpt_no_context(user_q, self.client)
            fb = augment_fallback(user_q, fb)
            if image_details:
                fb += "\n\n" + "\n".join(image_details)
            structured_answer.answer = fb
            structured_answer.image_paths = []
            approach = "fallback + image detail"
        else:
            approach = "RAG + short-term"

        if len(structured_answer.image_paths) > 3:
            structured_answer.image_paths = structured_answer.image_paths[:3]

        final_dict = structured_answer.dict()
        # Optionally, print the response
        # print(f"Bot ({approach}) => JSON result:\n{json.dumps(final_dict, indent=2)}\n")

        # Update session memory and chat history
        self.session_memory.chat_memory.add_ai_message(structured_answer.json())
        self.chat_history += f"Assistant: {structured_answer.json()}\n"

        # Write assistant response to chat_history.txt
        try:
            with open(self.CHAT_FILE, "a", encoding="utf-8") as f:
                f.write(f"Assistant: {structured_answer.json()}\n")
        except Exception as e:
            print(f"Error writing assistant response to chat_history.txt: {e}")

        # Return both 'answer' and 'image_paths'
        return final_dict

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reload-psm", action="store_true", help="Force rebuild of PSM DB.")
    parser.add_argument("--reload-img", action="store_true", help="Force rebuild of CLIP image DB.")
    args = parser.parse_args()

    # Initialize RAG chatbot
    chatbot = RAG(reload_psm=args.reload_psm, reload_img=args.reload_img)

    print("PSM Bot. Type 'reset' to clear short-term memory, 'end' to quit.\n")

    while True:
        try:
            user_q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if user_q.lower() in ["end", "exit", "quit"]:
            print("Session ended.")
            break

        if not user_q:
            print("Please enter a valid query.")
            continue

        try:
            response = chatbot.Get_Rag_Response(user_q)
            print(f"Bot => JSON result:\n{json.dumps(response, indent=2)}\n")
        except Exception as e:
            print(f"An error occurred while processing your request: {e}\n")

if __name__ == "__main__":
    main()