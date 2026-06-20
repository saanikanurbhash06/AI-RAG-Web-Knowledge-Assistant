from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the existing vector database
db = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

while True:
    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    results = db.similarity_search(question, k=3)

    print("\nTop Results:\n")

    for i, doc in enumerate(results, start=1):
        print(f"Result {i}")
        print("-" * 50)
        print(doc.page_content)
        print()