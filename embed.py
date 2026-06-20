from scraper import scrape_with_anakin
from rag import split_document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

url = input("Enter Website URL: ")

text = scrape_with_anakin(url)

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

print("Vector database created successfully!")
print("Total chunks:", len(chunks))