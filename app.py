import streamlit as st
import sqlite3
import pandas as pd

st.set_page_config(page_title="Financial Insights Assistant", layout="wide")

st.title("📊 Financial Insights Assistant (Agentic RAG Demo)")
st.markdown("Ask questions that combine **structured SQL data** with **unstructured financial reports**.")

# ----------------------------
# Example Structured Data (SQLite)
# ----------------------------
# Sample table: quarterly transactions
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE transactions (
    quarter TEXT,
    revenue INTEGER,
    expenses INTEGER
)
""")
cursor.executemany("INSERT INTO transactions VALUES (?, ?, ?)", [
    ("Q1-2025", 1200000, 800000),
    ("Q2-2025", 1400000, 900000),
    ("Q3-2025", 1600000, 950000),
])
conn.commit()

# Load into DataFrame
df = pd.read_sql_query("SELECT * FROM transactions", conn)
st.subheader("📈 Structured Data (Transactions)")
st.dataframe(df)

# ----------------------------
# Example Unstructured Data (Dummy Earnings Report)
# ----------------------------
earnings_report = """
Company XYZ reported strong growth in Q2 2025 with net revenue of 1.4M,
driven by higher transaction volumes. Operating expenses increased slightly
to 0.9M. The company projects Q3 2025 revenue at ~1.6M with stable margins.
"""

st.subheader("📄 Unstructured Data (Earnings Report Excerpt)")
st.text_area("Sample Report", earnings_report, height=150)

# ----------------------------
# RAG Simulation (Dummy Answer Logic)
# ----------------------------
st.subheader("💬 Ask a Financial Question")

question = st.text_input("Enter your query (e.g., 'Compare Q2 revenue in SQL vs earnings report').")

if st.button("Get Answer"):
    if "Q2" in question:
        sql_revenue = df.loc[df['quarter']=="Q2-2025", 'revenue'].values[0]
        report_revenue = "1.4M (from report)"
        answer = f"📊 From SQL: Q2-2025 revenue = {sql_revenue}\n📄 From Report: {report_revenue}\n✅ Both sources align closely."
    else:
        answer = "Demo only supports Q2 queries. Extend logic to handle dynamic RAG queries."
    
    st.success(answer)

st.markdown("---")
st.markdown("⚠️ *Demo: This is a simplified prototype. In production, connect LangChain + Pinecone/Weaviate for embeddings and real RAG pipeline.*")
