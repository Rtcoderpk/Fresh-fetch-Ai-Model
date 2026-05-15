# streamlit_app.py

import os
import gc
import requests
import streamlit as st

from bs4 import BeautifulSoup
from ddgs import DDGS

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Fresh Fetch AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Fresh Fetch AI")
st.caption("Fast RAG AI Chatbot with Live Web Search")

# =====================================================
# CONFIG
# =====================================================

LLM_MODEL = "google/flan-t5-small"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =====================================================
# SEARCH WEB
# =====================================================

def search_web(query, max_results=2):

    try:

        urls = []

        with DDGS() as ddgs:

            results = ddgs.text(
                query,
                max_results=max_results
            )

            for r in results:

                if "href" in r:
                    urls.append(r["href"])

        return urls

    except Exception as e:

        st.error(f"Search Error: {e}")
        return []

# =====================================================
# SCRAPE WEBSITE
# =====================================================

def scrape_url(url):

    try:

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=8
        )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "noscript"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        return text[:1500]

    except:
        return ""

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL
    )

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=80
    )

    return generator

# =====================================================
# BUILD VECTOR DB
# =====================================================

@st.cache_resource
def load_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

def build_vector_db(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_text(text)

    embeddings = load_embeddings()

    db = FAISS.from_texts(
        chunks,
        embeddings
    )

    return db

# =====================================================
# GENERATE ANSWER
# =====================================================

def generate_answer(query, db, generator):

    docs = db.similarity_search(
        query,
        k=2
    )

    context = "\n".join([
        d.page_content for d in docs
    ])

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""

    result = generator(prompt)

    return result[0]["generated_text"]

# =====================================================
# MAIN APP
# =====================================================

query = st.text_input(
    "Ask Your Question"
)

if st.button("Generate Answer"):

    if not query.strip():

        st.warning("Please enter a question")

    else:

        with st.spinner("🔍 Searching Web..."):

            urls = search_web(query)

        if not urls:

            st.error("No search results found")

        else:

            st.subheader("🌐 Sources")

            for url in urls:
                st.write(url)

            all_text = ""

            with st.spinner("📄 Reading Websites..."):

                for url in urls:

                    text = scrape_url(url)

                    if text:
                        all_text += text + "\n"

            if not all_text.strip():

                st.error("Could not extract website content")

            else:

                with st.spinner("🧠 Creating AI Memory..."):

                    db = build_vector_db(
                        all_text
                    )

                with st.spinner("🚀 Loading AI Model..."):

                    generator = load_model()

                with st.spinner("✍️ Generating Response..."):

                    answer = generate_answer(
                        query,
                        db,
                        generator
                    )

                st.subheader("✅ AI Answer")

                st.write(answer)
