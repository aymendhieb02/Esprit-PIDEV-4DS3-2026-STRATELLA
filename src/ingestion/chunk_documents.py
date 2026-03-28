from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_txt_documents(folder_path: str) -> list[dict]:
    """
    Charge tous les fichiers .txt du dossier.
    """
    documents = []
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Le dossier n'existe pas : {folder_path}")

    txt_files = list(folder.glob("*.txt"))

    for file_path in txt_files:
        try:
            content = file_path.read_text(encoding="utf-8").strip()

            if content:
                documents.append(
                    {
                        "file_name": file_path.name,
                        "content": content,
                        "source_path": str(file_path),
                    }
                )

        except Exception as e:
            print(f"[ERREUR] Impossible de lire {file_path.name} : {e}")

    return documents


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> list[dict]:
    """
    Découpe les documents en chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []

    for doc in documents:
        chunks = splitter.split_text(doc["content"])

        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['file_name']}_chunk_{i + 1}",
                    "file_name": doc["file_name"],
                    "source_path": doc["source_path"],
                    "chunk_text": chunk,
                    "chunk_index": i + 1,
                }
            )

    return all_chunks


def display_chunks(chunks: list[dict], max_display: int = 10) -> None:
    """
    Affiche un aperçu des chunks générés.
    """
    print(f"\nNombre total de chunks : {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:max_display], start=1):
        preview = chunk["chunk_text"][:250].replace("\n", " ")
        print(f"Chunk {i}")
        print(f"ID         : {chunk['chunk_id']}")
        print(f"Document   : {chunk['file_name']}")
        print(f"Index      : {chunk['chunk_index']}")
        print(f"Texte      : {preview}...")
        print("-" * 60)


if __name__ == "__main__":
    folder_path = "data/raw_docs"

    documents = load_txt_documents(folder_path)
    chunks = chunk_documents(documents, chunk_size=500, chunk_overlap=100)

    display_chunks(chunks)