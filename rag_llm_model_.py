
import os
import gc
import torch
import requests
import streamlit as st

from bs4 import BeautifulSoup
from ddgs import DDGS
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Fresh Fetch AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Fresh Fetch RAG AI Model")
st.write("AI chatbot with live web search + RAG")

# =========================
# ENV SETTINGS
# =========================

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# =========================
# MODELS
# =========================

LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =========================
# SEARCH WEB
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

    except Exception as e:
        st.error(f"Search Error: {e}")
        return []

# =========================
# SCRAPE URL
# =========================

def scrape_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        res = requests.get(
            url,
            headers=headers,
            timeout=10
        )

        soup = BeautifulSoup(res.text, "html.parser")

        for s in soup(["script", "style"]):
            s.decompose()

        text = soup.get_text(separator=" ", strip=True)

        return text[:4000]

    except:
        return ""

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:

        # GPU quantization
        if torch.cuda.is_available():

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )

            tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL
            )

            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                quantization_config=bnb_config,
                device_map="auto"
            )

        # CPU fallback
        else:

            tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL
            )

            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL
            )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            temperature=0.7
        )

        return pipe

    except Exception as e:
        st.error(f"Model Loading Error: {e}")
        return None

# =========================
# PROCESS RAG
# =========================

def build_rag(query):

    with st.spinner("🔍 Searching Web..."):
        urls = search_web(query)

    if not urls:
        return None

    st.subheader("🌐 Sources")

    for u in urls:
        st.write(u)

    all_text = ""

    with st.spinner("📄 Scraping Websites..."):

        for url in urls:
            text = scrape_url(url)

            if text:
                all_text += text + "\n"

    if not all_text.strip():
        return None

    with st.spinner("🧠 Creating Vector Database..."):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(all_text)

        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        db = FAISS.from_texts(
            chunks,
            embeddings
        )

    return db

# =========================
# GENERATE ANSWER
# =========================

def generate_answer(query, db, generator):

    docs = db.similarity_search(query, k=2)

    context = "\n".join(
        [d.page_content for d in docs]
    )

    prompt = f"""
<|system|>
You are a helpful AI assistant.
Answer only from provided context.

Context:
{context}
</s>

<|user|>
{query}
</s>

<|assistant|>
"""

    result = generator(prompt)

    answer = result[0]["generated_text"]

    if "<|assistant|>" in answer:
        answer = answer.split("<|assistant|>")[-1]

    return answer.strip()

# =========================
# MAIN UI
# =========================

query = st.text_input(
    "Ask Anything"
)

if st.button("Generate Answer"):

    if not query.strip():
        st.warning("Please enter a question")

    else:

        db = build_rag(query)

        if db is None:
            st.error("Failed to build RAG context")

        else:

            with st.spinner("🚀 Loading AI Model..."):
                generator = load_model()

            if generator is None:
                st.error("Model failed to load")

            else:

                with st.spinner("✍️ Generating Answer..."):

                    answer = generate_answer(
                        query,
                        db,
                        generator
                    )

                st.subheader("✅ AI Answer")
                st.write(answer)


