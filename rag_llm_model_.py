import os
import gc
import streamlit as st
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================
# UI
# =========================
st.set_page_config(page_title="Fresh Fetch AI", page_icon="🤖")
st.title("🤖 Fresh Fetch AI (Stable RAG)")
st.write("Live Web + AI Answer System")


# =========================
# CONFIG
# =========================
LLM_MODEL = "google/flan-t5-small"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================
# WEB SEARCH
# =========================
def search_web(query, max_results=2):
    try:
        urls = []
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)

            for r in results:
                if "href" in r:
                    urls.append(r["href"])

        return urls

    except:
        return []


# =========================
# SCRAPE WEBSITE
# =========================
def scrape_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=8)

        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        return soup.get_text(" ", strip=True)[:1500]

    except:
        return ""


# =========================
# LOAD MODEL (SAFE)
# =========================
@st.cache_resource
def load_model():
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    return model, tokenizer


# =========================
# GENERATE ANSWER (NO PIPELINE)
# =========================
def generate_answer(model, tokenizer, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return answer


# =========================
# EMBEDDINGS
# =========================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# =========================
# VECTOR DB
# =========================
def build_db(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_text(text)

    db = FAISS.from_texts(
        chunks,
        load_embeddings()
    )

    return db


# =========================
# MAIN APP
# =========================
query = st.text_input("Ask your question")

if st.button("Search & Answer"):

    if not query:
        st.warning("Please enter a question")

    else:

        # STEP 1
        with st.spinner("🔍 Searching web..."):
            urls = search_web(query)

        st.subheader("Sources")
        for u in urls:
            st.write(u)

        # STEP 2
        text = ""
        with st.spinner("📄 Scraping data..."):
            for u in urls:
                text += scrape_url(u) + "\n"

        if not text.strip():
            st.error("No data found")
            st.stop()

        # STEP 3
        with st.spinner("🧠 Building knowledge base..."):
            db = build_db(text)

        # STEP 4
        with st.spinner("🤖 Loading AI model..."):
            model, tokenizer = load_model()

        # STEP 5
        with st.spinner("✍️ Generating answer..."):

            docs = db.similarity_search(query, k=2)
            context = "\n".join([d.page_content for d in docs])

            prompt = f"""
Use the context to answer the question.

Context:
{context}

Question:
{query}
"""

            answer = generate_answer(model, tokenizer, prompt)

        # OUTPUT
        st.subheader("Answer")
        st.write(answer)
