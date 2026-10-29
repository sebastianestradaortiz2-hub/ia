"""Genera o actualiza la base vectorial a partir del PDF del libro.

Uso:
    python indexar.py

La primera vez descarga el modelo de embeddings local (~80 MB) y procesa el PDF.
Las siguientes ejecuciones solo actualizan los fragmentos que cambiaron.
"""

import os
import sys

# Asegura que el PDF se busque en la misma carpeta que este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(
    BASE_DIR,
    "Investigacion-Operaciones10Edicion-Frederick-S-Hillier.pdf",
)

from vector_store import (
    cargar_pdf,
    get_chroma_client,
    get_collection,
    get_embedding_function,
    indexar_chunks,
)


def main():
    if not os.path.exists(PDF_PATH):
        print(f"❌ No se encontró el PDF en {PDF_PATH}")
        print("   Colocá el archivo PDF en la misma carpeta que indexar.py")
        sys.exit(1)

    print(f"📄 PDF: {PDF_PATH}")

    chunks = cargar_pdf(PDF_PATH)
    if not chunks:
        print("⚠️ No se generaron fragmentos.")
        sys.exit(0)

    client = get_chroma_client()
    ef = get_embedding_function()
    collection = get_collection(client, ef)

    indexar_chunks(chunks, collection)
    print("\n✅ Listo. La base vectorial quedó guardada en la carpeta configurada.")


if __name__ == "__main__":
    main()
