import os
import gc
import streamlit as st
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Fresh Fetch AI", page_icon="🤖")
st.title("🤖 Fresh Fetch AI (RAG + Web Search)")
st.write("Ask anything with live web + AI answer")

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
# SCRAPE
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

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return generator


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
# ANSWER GENERATION
# =========================

def generate_answer(query, db, generator):

    docs = db.similarity_search(query, k=2)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer using only context.

Context:
{context}

Question:
{query}
"""

    result = generator(prompt, max_new_tokens=120)

    return result[0]["generated_text"]


# =========================
# MAIN APP
# =========================

query = st.text_input("Ask your question")

if st.button("Search & Answer"):

    if not query:
        st.warning("Enter a question")

    else:

        with st.spinner("Searching web..."):
            urls = search_web(query)

        st.subheader("Sources")
        for u in urls:
            st.write(u)

        text = ""
        with st.spinner("Scraping data..."):
            for u in urls:
                text += scrape_url(u) + "\n"

        if not text.strip():
            st.error("No content found")
            st.stop()

        with st.spinner("Building knowledge base..."):
            db = build_db(text)

        with st.spinner("Loading AI model..."):
            generator = load_model()

        with st.spinner("Generating answer..."):
            answer = generate_answer(query, db, generator)

        st.subheader("Answer")
        st.write(answer)
