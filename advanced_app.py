# advanced_app.py
"""
Advanced Financial Insights Assistant (Agentic RAG) - Streamlit Demo
Features:
- Upload CSV (structured) and PDF / TXT (unstructured) documents
- PDF parsing per-page (pdfplumber)
- Chunking using tiktoken aware token counts
- Embeddings via OpenAIEmbeddings (or Pinecone if configured)
- Pinecone persistence (optional) with fallback to in-memory FAISS
- SQL execution on uploaded CSVs using sqlite3 (in-memory DB or file)
- Retrieval + LLM fusion with guardrails and citation markers [SRCx] / [SQL]
- Simple index management (create, clear, show stats)
- Streamlit UI for model/embedding settings and debugging
Notes:
- Requires OPENAI_API_KEY for embeddings + LLM
- Optional: PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX to persist vectors
"""

import os
import io
import time
import json
import sqlite3
import tempfile
from typing import List, Tuple, Optional, Dict, Any
import streamlit as st
import pandas as pd

# LangChain and vectorstore imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional: pinecone client
try:
    import pinecone
    PINECONE_AVAILABLE = True
except Exception:
    PINECONE_AVAILABLE = False

# PDF parsing
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

# token counting (tiktoken) - best-effort, not critical
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except Exception:
    TIKTOKEN_AVAILABLE = False

# -------------------------
# Configuration & Secrets
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # region like 'us-west1-gcp'
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "sonu-demo-index")

DEFAULT_EMBED_MODEL = "text-embedding-3-small"  # change if needed
DEFAULT_LLM_MODEL = "gpt-4o-mini"  # change as per account access

# -------------------------
# Utilities
# -------------------------
def ensure_api_keys():
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not set. Go to Settings → Secrets and add OPENAI_API_KEY.")
        st.stop()

def get_token_count(text: str, model_name: str = "gpt-4o-mini") -> int:
    # best-effort token count using tiktoken; fallback to character heuristic
    try:
        if TIKTOKEN_AVAILABLE:
            enc = tiktoken.encoding_for_model(model_name)
            return len(enc.encode(text))
    except Exception:
        pass
    return max(1, len(text) // 4)  # rough chars -> tokens

def chunk_text_tiktoken_aware(text: str, max_tokens: int = 800, overlap_tokens: int = 100, model_name: str = "gpt-4o-mini") -> List[str]:
    # Use tiktoken if available; otherwise fallback to char-based splitter
    if TIKTOKEN_AVAILABLE:
        enc = tiktoken.encoding_for_model(model_name)
        tokens = enc.encode(text)
        chunks = []
        start = 0
        step = max_tokens - overlap_tokens
        while start < len(tokens):
            chunk_tokens = tokens[start:start+max_tokens]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
            start += step
        return chunks
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens*4, chunk_overlap=overlap_tokens*4)
        return splitter.split_text(text)

def parse_pdf(file_bytes: bytes) -> List[Tuple[str, int, str]]:
    """Return list of tuples: (source_name, page_number, page_text)"""
    results = []
    if not PDFPLUMBER_AVAILABLE:
        st.warning("pdfplumber not available. Install pdfplumber to enable PDF parsing.")
        return results
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                results.append(("uploaded_pdf", i, text))
    return results

def build_documents_from_texts(text_sources: List[Dict[str, Any]], chunk_tokens: int = 800, overlap_tokens: int = 100) -> List[Document]:
    """
    text_sources: List of dicts: {'source_name': str, 'page': int (optional), 'text': str}
    returns list of langchain Documents with metadata containing source_name and page
    """
    docs = []
    for src in text_sources:
        name = src.get("source_name", "unknown")
        page = src.get("page", None)
        text = src.get("text", "")
        chunks = chunk_text_tiktoken_aware(text, max_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        for idx, c in enumerate(chunks, start=1):
            meta = {"source_name": name, "page": page, "chunk_id": f"{name}_p{page}_c{idx}" if page else f"{name}_c{idx}"}
            docs.append(Document(page_content=c, metadata=meta))
    return docs

# -------------------------
# Vector Store: Pinecone or FAISS fallback
# -------------------------
def init_pinecone_index(api_key: str, environment: str, index_name: str):
    if not PINECONE_AVAILABLE:
        raise RuntimeError("pinecone-client not installed in environment.")
    pinecone.init(api_key=api_key, environment=environment)
    if index_name not in pinecone.list_indexes():
        # create a minimal index with 1536 dims (embedding dimensionality may vary by model)
        pinecone.create_index(index_name, dimension=1536)
    return pinecone.Index(index_name)

def create_vectorstore_from_documents(docs: List[Document], embeddings, use_pinecone: bool = False, pinecone_index: Optional[str] = None):
    if use_pinecone and PINECONE_AVAILABLE and PINECONE_API_KEY and PINECONE_ENV:
        # init pinecone and use LangChain Pinecone wrapper
        init_pinecone_index(PINECONE_API_KEY, PINECONE_ENV, pinecone_index)
        vectorstore = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
        return vectorstore, "pinecone"
    else:
        # FAISS in-memory; advise persistence via pickle if needed
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore, "faiss"

# -------------------------
# LLM Prompting / Guardrails
# -------------------------
def prepare_prompt_with_guardrails(sql_text: str, retrieved_docs: List[Tuple[Document, float]], user_query: str) -> str:
    snippets = []
    for i, (doc, score) in enumerate(retrieved_docs, start=1):
        src = doc.metadata.get("source_name", f"SRC{i}")
        page = doc.metadata.get("page", "")
        id_ = doc.metadata.get("chunk_id", f"{src}_c{i}")
        snippet = doc.page_content.strip()
        snippets.append(f"[{id_}] source={src} page={page} score={score:.3f}:\n{snippet}\n")
    snippets_block = "\n\n".join(snippets) if snippets else "<no unstructured snippets>"

    prompt = f\"\"\"You are a fact-focused financial analytics assistant. Use ONLY the provided SQL results and unstructured snippets to answer the user's question.
- If the evidence is insufficient to answer confidently, reply exactly: I DON'T KNOW
- Cite every factual claim with either [SQL] (if from the SQL results) or citation tokens that match the chunk ids like [source_chunk_id]
- Do not hallucinate, infer beyond provided evidence, or invent numbers.

USER QUESTION:
{user_query}

SQL RESULTS (CSV-like):
{sql_text}

UNSTRUCTURED SNIPPETS (each chunk is labeled):
{snippets_block}

INSTRUCTIONS:
1) Provide a concise answer (3-6 sentences). Each sentence with a factual claim must include a citation in square brackets.
2) If you provide a recommendation, label it and base it on cited evidence.
3) If you cannot answer from the provided evidence, reply exactly: I DON'T KNOW
\"\"\"
    return prompt

# -------------------------
# Streamlit UI and App Flow
# -------------------------
st.set_page_config(page_title="Advanced Financial Insights Assistant (Agentic RAG)", layout="wide")
st.title("🔬 Advanced Financial Insights Assistant — Agentic RAG (Pinecone/FAISS + PDF + Guardrails)")

# sidebar settings
with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox("LLM model", options=[DEFAULT_LLM_MODEL, "gpt-4o", "gpt-4o-mini", "gpt-4o-realtime-preview"], index=0)
    embed_model_choice = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
    top_k = st.slider("Retrieval top-k", min_value=1, max_value=8, value=4)
    chunk_tokens = st.number_input("Chunk size (tokens)", min_value=200, max_value=2000, value=800, step=50)
    overlap_tokens = st.number_input("Chunk overlap (tokens)", min_value=0, max_value=500, value=100, step=10)
    use_pinecone = st.checkbox("Use Pinecone (if API keys configured)", value=bool(PINECONE_API_KEY and PINECONE_ENV))
    persist_index = st.checkbox("Persist FAISS to disk (if not using Pinecone)", value=False)
    temperature = st.slider("LLM temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    max_tokens = st.number_input("LLM max_tokens", min_value=128, max_value=2000, value=512, step=64)
    st.markdown("---")
    st.write("OpenAI & Pinecone keys must be set in Streamlit Secrets or environment variables.")

# main layout
col1, col2 = st.columns([1.4, 2.6])

with col1:
    st.subheader("1) Upload structured & unstructured data")
    uploaded_csv = st.file_uploader("Upload CSV (structured) — will create SQL table", type=["csv"])
    example_table_btn = st.button("Create demo transactions table")
    st.write("----")
    st.subheader("Unstructured: Upload PDF/TXT or paste text")
    uploaded_pdf = st.file_uploader("Upload PDF (multi-page) ", type=["pdf"])
    uploaded_txt = st.file_uploader("Upload TXT", type=["txt"])
    pasted_text = st.text_area("Or paste unstructured text here", height=200)

    st.write("----")
    st.subheader("Index Management")
    if st.button("(Re)create embeddings & index from uploaded docs"):
        st.session_state['recreate_index'] = True
    if st.button("Clear index (FAISS or Pinecone)"):
        st.session_state['clear_index'] = True
    st.write("Index status:")
    st.write(st.session_state.get("index_status", "No index created yet"))

with col2:
    st.subheader("2) SQL & Question")
    user_sql = st.text_area("Enter SQL query to run against uploaded table (example: SELECT quarter, revenue FROM transactions):", height=120)
    user_question = st.text_input("Ask a combined question (uses SQL + unstructured sources):")

    if st.button("Execute RAG Query"):
        ensure_api_keys()

        # --- Prepare SQL DB ---
        conn = sqlite3.connect(":memory:")
        if uploaded_csv:
            try:
                df_csv = pd.read_csv(uploaded_csv)
                table_name = uploaded_csv.name.rsplit(".",1)[0].replace("-", "_").lower()
                df_csv.to_sql(table_name, conn, index=False, if_exists="replace")
                st.success(f"CSV uploaded and table created: {table_name}")
            except Exception as e:
                st.error(f"CSV parse error: {e}")
                st.stop()
        elif example_table_btn:
            df_demo = pd.DataFrame({
                "quarter": ["Q1-2025","Q2-2025","Q3-2025"],
                "revenue": [1200000, 1400000, 1600000],
                "expenses": [800000, 900000, 950000]
            })
            df_demo.to_sql("transactions", conn, index=False, if_exists="replace")
            st.success("Demo table created: transactions")
        else:
            if not user_sql:
                st.error("No SQL provided and no CSV/demo. Please upload CSV or create demo table.")
                st.stop()

        # run SQL
        try:
            df_result = pd.read_sql_query(user_sql, conn)
            st.subheader("SQL Result (preview)")
            st.dataframe(df_result.head(50))
        except Exception as e:
            st.error(f"SQL execution error: {e}")
            st.stop()

        # --- Prepare unstructured sources ---
        text_sources = []
        if uploaded_pdf:
            try:
                pdf_bytes = uploaded_pdf.read()
                pages = parse_pdf(pdf_bytes)
                for name, page_no, txt in pages:
                    text_sources.append({"source_name": uploaded_pdf.name, "page": page_no, "text": txt})
                st.success(f"Parsed {len(pages)} pages from {uploaded_pdf.name}")
            except Exception as e:
                st.error(f"PDF parse error: {e}")
        if uploaded_txt:
            try:
                content = uploaded_txt.read().decode("utf-8")
                text_sources.append({"source_name": uploaded_txt.name, "page": None, "text": content})
                st.success(f"Loaded text from {uploaded_txt.name}")
            except Exception as e:
                st.error(f"TXT read error: {e}")
        if pasted_text and pasted_text.strip():
            text_sources.append({"source_name": "pasted_text", "page": None, "text": pasted_text.strip()})

        # --- Build / Recreate vector index if requested ---
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=embed_model_choice)
        vectorstore = None
        vectorstore_type = None

        try:
            if st.session_state.get('recreate_index', False) or 'vectorstore' not in st.session_state:
                # build docs
                docs = build_documents_from_texts(text_sources, chunk_tokens, overlap_tokens)
                if len(docs) == 0:
                    st.warning("No unstructured docs to index. Skipping retrieval step.")
                    retrieved_docs = []
                else:
                    vs, vs_type = create_vectorstore_from_documents(docs, embeddings, use_pinecone=use_pinecone, pinecone_index=PINECONE_INDEX)
                    vectorstore = vs
                    vectorstore_type = vs_type
                    st.session_state['vectorstore_type'] = vs_type
                    st.session_state['vectorstore'] = vs  # note: not serializable for session across restarts on Streamlit Cloud
                    st.success(f"Created vectorstore ({vs_type}) with {len(docs)} chunks.")
                    st.session_state['index_status'] = f"Index created: {vs_type} with {len(docs)} chunks"
                st.session_state['recreate_index'] = False

            else:
                # use existing session vectorstore if present
                vectorstore = st.session_state.get('vectorstore', None)
                vectorstore_type = st.session_state.get('vectorstore_type', None)

            # If we have a vectorstore and texts, perform retrieval
            retrieved_docs = []
            if vectorstore is not None and user_question.strip():
                # similarity search
                retrieved = vectorstore.similarity_search_with_score(user_question, k=top_k)
                # returns list of (Document, score)
                retrieved_docs = retrieved
                st.subheader("Retrieved snippets")
                for d, score in retrieved_docs:
                    meta = d.metadata
                    st.markdown(f"- **{meta.get('chunk_id','unknown')}** (source: {meta.get('source_name')}, page: {meta.get('page')}, score={score:.3f})")
                    st.write(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))

            # --- Prepare prompt and call LLM ---
            sql_text = df_result.head(100).to_csv(index=False)
            prompt = prepare_prompt_with_guardrails(sql_text, retrieved_docs if retrieved_docs else [], user_question)
            st.subheader("Prompt (truncated)")
            st.code(prompt[:4000] + ("\n\n... (truncated)" if len(prompt) > 4000 else ""))

            # call LLM
            with st.spinner("Calling LLM... (this consumes OpenAI quota)"):
                llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name=model_choice, temperature=temperature, max_tokens=max_tokens)
                answer = llm(prompt)
            st.subheader("Assistant Answer")
            st.write(answer)

            # show structured citations
            if retrieved_docs:
                st.subheader("Citations / Sources (verification)")
                for d, score in retrieved_docs:
                    meta = d.metadata
                    st.markdown(f"**{meta.get('chunk_id','unknown')}** — source: {meta.get('source_name')} — page: {meta.get('page')} — (score={score:.3f})")
                    st.write(d.page_content[:1500] + ("..." if len(d.page_content) > 1500 else ""))

            # optional persistence for FAISS
            if persist_index and vectorstore_type == "faiss" and vectorstore is not None:
                try:
                    import pickle, os
                    fname = "/tmp/faiss_index.pkl"
                    with open(fname, "wb") as fh:
                        pickle.dump(vectorstore, fh)
                    st.info(f"FAISS index persisted to {fname}")
                except Exception as e:
                    st.warning(f"Failed to persist FAISS index: {e}")

        except Exception as e:
            st.error(f"Error during RAG execution: {e}")
            st.stop()

# Footer
st.markdown("---")
st.markdown("**Notes:** For production-ready systems, use secure key management, persistent vector DB (Pinecone/Weaviate), batching, monitoring, and rate-limiting. This demo is for prototyping and interview/demo purposes.")
