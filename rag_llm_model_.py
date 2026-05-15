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
st.title("🤖 Fresh Fetch AI - RAG Assistant")
st.write("Ask anything with live web + AI reasoning")


# =========================
# CONFIG
# =========================
LLM_MODEL = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# =========================
# WEB SEARCH
# =========================
def search_web(query, max_results=3):
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
        res = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(res.text, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)

        return text[:2000]

    except:
        return ""


# =========================
# MODEL LOAD
# =========================
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    return model, tokenizer


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
        chunk_size=400,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    db = FAISS.from_texts(
        chunks,
        load_embeddings()
    )

    return db


# =========================
# GENERATION (FIXED)
# =========================
def generate_answer(model, tokenizer, query, context):

    prompt = f"""
You are an expert assistant.

Use ONLY the context below.
Give detailed, structured answer with bullet points.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=350,
        temperature=0.2,
        do_sample=False
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


# =========================
# MAIN APP
# =========================
query = st.text_input("Ask your question")

if st.button("Get Answer"):

    if not query:
        st.warning("Please enter a question")

    else:

        # STEP 1: SEARCH
        with st.spinner("Searching web..."):
            urls = search_web(query)

        st.subheader("Sources")
        for u in urls:
            st.write(u)

        # STEP 2: SCRAPE
        text = ""
        with st.spinner("Reading sources..."):
            for u in urls:
                text += scrape_url(u) + "\n"

        if not text.strip():
            st.error("No content found")
            st.stop()

        # STEP 3: VECTOR DB
        with st.spinner("Building knowledge base..."):
            db = build_db(text)

        # STEP 4: LOAD MODEL
        with st.spinner("Loading AI model..."):
            model, tokenizer = load_model()

        # STEP 5: RAG RETRIEVAL
        docs = db.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])

        # STEP 6: ANSWER
        with st.spinner("Generating answer..."):
            answer = generate_answer(model, tokenizer, query, context)

        st.subheader("Answer")
        st.write(answer)
