from pathlib import Path


def load_txt_documents(folder_path: str) -> list[dict]:
    """
    Charge tous les fichiers .txt d'un dossier.
    Retourne une liste de dictionnaires contenant :
    - file_name
    - content
    - source_path
    """
    documents = []
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Le dossier n'existe pas : {folder_path}")

    txt_files = list(folder.glob("*.txt"))

    if not txt_files:
        print("Aucun fichier .txt trouvé dans le dossier.")
        return documents

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
            else:
                print(f"[INFO] Fichier vide ignoré : {file_path.name}")

        except Exception as e:
            print(f"[ERREUR] Impossible de lire {file_path.name} : {e}")

    return documents


def display_documents(documents: list[dict]) -> None:
    """
    Affiche un résumé des documents chargés.
    """
    print(f"\nNombre de documents chargés : {len(documents)}\n")

    for i, doc in enumerate(documents, start=1):
        preview = doc["content"][:300].replace("\n", " ")
        print(f"Document {i}")
        print(f"Nom       : {doc['file_name']}")
        print(f"Chemin    : {doc['source_path']}")
        print(f"Aperçu    : {preview}...")
        print("-" * 60)


if __name__ == "__main__":
    folder_path = "data/raw_docs"
    documents = load_txt_documents(folder_path)
    display_documents(documents)