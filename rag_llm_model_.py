import streamlit as st
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# =========================
# UI
# =========================
st.set_page_config(page_title="Fresh Fetch AI", page_icon="🤖")
st.title("🤖 Fresh Fetch AI (Clean RAG System)")
st.write("Accurate answers from cleaned web data")


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
# CLEAN SCRAPER (IMPORTANT FIX)
# =========================
def scrape_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)

        soup = BeautifulSoup(res.text, "html.parser")

        # remove junk
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)

        # aggressive cleaning
        sentences = text.split(".")
        clean = []

        bad_words = [
            "login", "sign up", "cookie", "subscribe",
            "menu", "advertisement", "privacy policy",
            "terms", "copyright"
        ]

        for s in sentences:
            s = s.strip().lower()

            if len(s) < 40:
                continue
            if any(b in s for b in bad_words):
                continue

            clean.append(s)

        return ". ".join(clean[:15])

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
        chunk_size=350,
        chunk_overlap=50
    )

    chunks = splitter.split_text(text)

    embeddings = load_embeddings()

    return FAISS.from_texts(chunks, embeddings)


# =========================
# STRICT ANSWER GENERATION (FIXED OUTPUT QUALITY)
# =========================
def generate_answer(model, tokenizer, query, context):

    prompt = f"""
You are an expert assistant.

RULES:
- Use ONLY the context below
- If context is not enough, say "insufficient information"
- Do NOT copy website menus or irrelevant text
- Give structured bullet points
- Be accurate and clear

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.2,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================
# MAIN APP
# =========================
query = st.text_input("Ask your question")

if st.button("Get Answer"):

    if not query:
        st.warning("Please enter a question")
        st.stop()

    # STEP 1 - SEARCH
    with st.spinner("Searching web..."):
        urls = search_web(query)

    st.subheader("Sources")
    for u in urls:
        st.write(u)

    # STEP 2 - SCRAPE CLEAN DATA
    raw_text = ""
    with st.spinner("Extracting clean content..."):
        for u in urls:
            raw_text += scrape_url(u) + "\n"

    if not raw_text.strip():
        st.error("No useful content extracted")
        st.stop()

    # STEP 3 - VECTOR DB
    with st.spinner("Building knowledge base..."):
        db = build_db(raw_text)

    # STEP 4 - LOAD MODEL
    with st.spinner("Loading AI model..."):
        model, tokenizer = load_model()

    # STEP 5 - RETRIEVE
    docs = db.similarity_search(query, k=5)
    context = "\n".join([d.page_content for d in docs])

    # STEP 6 - GENERATE
    with st.spinner("Generating answer..."):
        answer = generate_answer(model, tokenizer, query, context)

    # OUTPUT
    st.subheader("Answer")
    st.write(answer)
