import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai

from scraper import scrape_with_anakin
from rag import split_document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================
# Load Environment Variables
# ==========================
load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")

# ==========================
# FastAPI Setup
# ==========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Request Models
# ==========================
class URLRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

# ==========================
# Home Route
# ==========================
@app.get("/")
def home():
    return {"message": "AI RAG Assistant Running"}

# ==========================
# Scrape & Index Website
# ==========================
@app.post("/scrape")
def scrape(req: URLRequest):

    text = scrape_with_anakin(req.url)

    chunks = split_document(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    db.persist()

    return {
        "message": f"Successfully indexed {len(chunks)} chunks"
    }

# ==========================
# Ask Question
# ==========================
@app.post("/query")
def query(req: QueryRequest):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    docs = db.similarity_search(
        req.question,
        k=3
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the question.

If the answer is partially available, provide the best answer possible from the context.

Only say "I couldn't find that information in the provided documents."
when there is absolutely no relevant information.

Context:
{context}

Question:
{req.question}

Answer:
"""

    response = model.generate_content(prompt)

    return {
        "answer": response.text
    }