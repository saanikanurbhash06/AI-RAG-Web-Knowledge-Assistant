from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_document(file_path):
    """Load HTML file and extract plain text."""
    with open(file_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")

    return text


def split_document(text):
    """Split document into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


if __name__ == "__main__":
    text = load_document("data/knowledge.html")
    chunks = split_document(text)

    print("Total Chunks:", len(chunks))
    print("-" * 50)
    print(chunks[0])