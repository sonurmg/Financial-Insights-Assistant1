# advanced_app_fixed.py
"""
Advanced Financial Insights Assistant (Agentic RAG) - Streamlit Demo (Fixed)
"""

import os
import io
import sqlite3
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
# Utilities
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

    prompt = f"""
You are a fact-focused financial analytics assistant. Use ONLY the provided SQL results and unstructured snippets to answer the user's question.
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
"""
    return prompt

# Placeholder minimal Streamlit UI to test prompt function
st.set_page_config(page_title="Advanced Financial Insights Assistant (Fixed)", layout="wide")
st.title("✅ Advanced Financial Insights Assistant (Fixed Version)")

st.write("This is the corrected version. The guardrails prompt now compiles without syntax errors.")
sql_text = "quarter,revenue\nQ1,1200000\nQ2,1400000"
retrieved_docs = []
user_query = "Summarize Q2 revenue compared with unstructured data."
prompt_example = prepare_prompt_with_guardrails(sql_text, retrieved_docs, user_query)
st.subheader("Generated Prompt Example")
st.code(prompt_example)
