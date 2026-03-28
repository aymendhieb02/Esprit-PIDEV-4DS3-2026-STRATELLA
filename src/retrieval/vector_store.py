from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_txt_documents(folder_path: str):
    documents = []
    folder = Path(folder_path)

    excluded_files = {"scenarios_questions.txt"}

    for file_path in folder.glob("*.txt"):
        print("Trouvé :", file_path.name)

        if file_path.name in excluded_files:
            print(f"[INFO] Fichier exclu de l'indexation : {file_path.name}")
            continue

        content = file_path.read_text(encoding="utf-8").strip()

        if content:
            documents.append(
                {
                    "file_name": file_path.name,
                    "content": content,
                }
            )

    return documents


def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []

    for doc in documents:
        texts = splitter.split_text(doc["content"])

        for i, text in enumerate(texts):
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "source": doc["file_name"],
                        "chunk_id": f"{doc['file_name']}_chunk_{i+1}"
                    }
                }
            )

    return chunks


def create_vector_store(chunks):
    print("Création des embeddings...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    return vectorstore


def save_vector_store(vectorstore, path="vectorstore/faiss_index"):
    vectorstore.save_local(path)
    print(f"Vector store sauvegardé dans : {path}")


def test_search(vectorstore, query, k=3):
    print(f"\nQuestion : {query}\n")

    results = vectorstore.similarity_search(query, k=k)

    for i, doc in enumerate(results, 1):
        print(f"Résultat {i}")
        print(f"Source : {doc.metadata['source']}")
        print(f"Chunk  : {doc.metadata['chunk_id']}")
        print(f"Texte  : {doc.page_content[:300]}...")
        print("-" * 60)


if __name__ == "__main__":
    folder = "data/raw_docs"

    docs = load_txt_documents(folder)
    print(f"\nNombre de documents indexés : {len(docs)}")

    chunks = chunk_documents(docs)
    print(f"Nombre de chunks générés : {len(chunks)}")

    vectorstore = create_vector_store(chunks)
    save_vector_store(vectorstore)

    test_query = "Pourquoi mon rapport est vide ?"
    test_search(vectorstore, test_query)