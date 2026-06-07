from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from config import get_embeddings

DATA_DIR = Path("data")
PERSIST_DIR = "chroma_db"


def load_documents():
    """Lee todos los PDFs de data/ y devuelve sus páginas como documentos."""
    docs = []
    for pdf in DATA_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
        print(f"  ✓ {pdf.name}")
    print(f"Cargadas {len(docs)} páginas en total.")
    return docs


def split_documents(docs):
    """Corta los documentos en chunks con solapamiento."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=150,
        # Intenta cortar respetando estructura: primero párrafos, luego
        # líneas, luego oraciones, y recién al final caracteres sueltos.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Generados {len(chunks)} chunks.")
    return chunks


def main():
    print("📥 Cargando documentos...")
    docs = load_documents()
    print("✂️  Cortando en chunks...")
    chunks = split_documents(docs)
    print("🧠 Vectorizando y guardando (esto puede tardar la primera vez)...")
    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"✅ Base vectorial guardada en ./{PERSIST_DIR}")


if __name__ == "__main__":
    main()