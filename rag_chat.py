import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==========================
# Load Environment Variables
# ==========================
load_dotenv()

# ==========================
# Configure Gemini API
# ==========================
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel("gemini-2.5-flash")

# ==========================
# Load Embedding Model
# ==========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================
# Load Chroma Database
# ==========================
db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

print("===================================")
print("      RAG Chatbot is Ready!")
print("Type 'exit' to quit.")
print("===================================")

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    # Retrieve relevant documents
    docs = db.similarity_search(question, k=3)

    # Combine retrieved text
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prompt for Gemini
    prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY using the information provided in the context below.

If the answer is not found in the context, say:
"I couldn't find that information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.generate_content(prompt)

    print("\nAssistant:")
    print(response.text)