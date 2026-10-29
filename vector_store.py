"""Utilidades para crear y consultar una base de datos vectorial persistente.

La base usa ChromaDB con persistencia en disco y embeddings locales de
sentence-transformers. Esto permite indexar el PDF una sola vez y consultar
sin consumir tokens de la API de IA.
"""

import hashlib
import logging
import os
import sys
from typing import List

from dotenv import load_dotenv

load_dotenv()

# En Windows la consola puede usar cp1252 y fallar con acentos/emojis
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Silencia avisos inofensivos del lector de PDF
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pypdf").setLevel(logging.ERROR)


DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_COLLECTION_NAME = "io_libro"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MAX_CHUNKS = 500


def get_env(key: str, default: str = "") -> str:
    """Lee una variable de entorno (también acepta nombres en minúsculas)."""
    return os.environ.get(key) or os.environ.get(key.lower()) or default


def get_chroma_path() -> str:
    """Devuelve la ruta donde se persiste ChromaDB."""
    path = get_env("CHROMA_PATH", DEFAULT_CHROMA_PATH)
    return os.path.abspath(path)


def get_embedding_model() -> str:
    """Devuelve el modelo de embeddings local a usar."""
    return get_env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_embedding_function(model_name: str = None):
    """Devuelve la función de embeddings de sentence-transformers."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    model_name = model_name or get_embedding_model()
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_chroma_client(persist_path: str = None):
    """Crea o reusa un cliente ChromaDB persistente."""
    import chromadb

    persist_path = persist_path or get_chroma_path()
    os.makedirs(persist_path, exist_ok=True)
    return chromadb.PersistentClient(path=persist_path)


def get_collection(client=None, embedding_function=None, collection_name: str = None):
    """Obtiene la colección de ChromaDB, creándola si no existe."""
    client = client or get_chroma_client()
    embedding_function = embedding_function or get_embedding_function()
    collection_name = collection_name or DEFAULT_COLLECTION_NAME
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )


def cargar_pdf(pdf_path: str) -> List:
    """Carga un PDF y lo divide en fragmentos. Retorna la lista de chunks."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No se encontró el PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documentos_pdf = loader.load()
    print(f"✅ PDF cargado: {len(documentos_pdf)} páginas")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = text_splitter.split_documents(documentos_pdf)
    print(f"✅ {len(chunks)} fragmentos creados")
    return chunks


def _chunk_id(text: str, page: int, idx: int) -> str:
    """Genera un ID determinístico para un fragmento."""
    base = f"{page}:{idx}:{text.strip()[:200]}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def indexar_chunks(chunks, collection, max_chunks: int = DEFAULT_MAX_CHUNKS, batch_size: int = 100):
    """Indexa los fragmentos en ChromaDB usando upsert para evitar duplicados."""
    chunks = chunks[:max_chunks]
    if not chunks:
        print("⚠️ No hay fragmentos para indexar.")
        return

    total = len(chunks)
    print(f"🔄 Indexando {total} fragmentos en ChromaDB...")

    for i in range(0, total, batch_size):
        lote = chunks[i : i + batch_size]
        ids = [_chunk_id(c.page_content, c.metadata.get("page", 0), idx) for idx, c in enumerate(lote)]
        texts = [c.page_content for c in lote]
        metadatas = [{"page": c.metadata.get("page")} for c in lote]

        collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        print(f"  {min(i + len(lote), total)}/{total} fragmentos indexados", flush=True)

    print(f"✅ Base vectorial persistida: {collection.count()} documentos")


def buscar_contexto(query: str, collection, k: int = 4):
    """Busca los k fragmentos más relevantes para la consulta.

    Retorna una lista de diccionarios con: text, page, distance.
    """
    results = collection.query(query_texts=[query], n_results=k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    out = []
    for text, meta, distance in zip(documents, metadatas, distances):
        out.append(
            {
                "text": text,
                "page": meta.get("page") if meta else None,
                "distance": distance,
            }
        )
    return out


def format_context(results: List[dict], separator: str = "\n---\n") -> str:
    """Formatea los resultados de búsqueda como un único string de contexto."""
    return separator.join(r["text"] for r in results)
