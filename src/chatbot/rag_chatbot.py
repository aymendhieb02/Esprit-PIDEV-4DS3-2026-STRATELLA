import os
from dotenv import load_dotenv
import cohere

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


def load_vector_store(path: str = "vectorstore/faiss_index"):
    """
    Charge la base vectorielle FAISS sauvegardée localement.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        folder_path=path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def retrieve_context(vectorstore, user_question: str, k: int = 3):
    """
    Récupère les chunks les plus proches de la question.
    """
    results = vectorstore.similarity_search(user_question, k=k)
    return results


def build_context_text(retrieved_docs):
    """
    Transforme les chunks retrouvés en texte de contexte.
    """
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "source_inconnue")
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
        text = doc.page_content.strip()

        context_parts.append(
            f"[Source: {source} | Chunk: {chunk_id}]\n{text}"
        )

    return "\n\n".join(context_parts)


def generate_answer(user_question: str, context_text: str):
    """
    Génère une réponse avec Cohere Chat API V2.
    """
    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise ValueError("COHERE_API_KEY introuvable dans .env")

    co = cohere.ClientV2(api_key=api_key)

    system_message = (
        "Tu es un assistant utilisateur pour une application d'analyse documentaire. "
        "Réponds en français. "
        "Explique simplement, clairement, et sans jargon technique. "
        "Aide un utilisateur non informaticien. "
        "Base-toi uniquement sur le contexte fourni. "
        "Si l'information manque dans le contexte, dis-le honnêtement. "
        "Donne des étapes pratiques quand c'est utile."
    )

    response = co.chat(
        model="command-a-03-2025",
        messages=[
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Contexte :\n{context_text}\n\nQuestion :\n{user_question}"
            },
        ],
        temperature=0.3,
    )

    return response.message.content[0].text.strip()


def ask_rag(user_question: str):
    """
    Pipeline complet :
    question -> retrieval -> génération
    """
    vectorstore = load_vector_store()
    retrieved_docs = retrieve_context(vectorstore, user_question, k=3)
    context_text = build_context_text(retrieved_docs)
    answer = generate_answer(user_question, context_text)

    print("\n===== QUESTION =====")
    print(user_question)

    print("\n===== CONTEXTE RETROUVÉ =====")
    print(context_text)

    print("\n===== RÉPONSE DU CHATBOT =====")
    print(answer)


if __name__ == "__main__":
    question = input("Pose ta question : ")
    ask_rag(question)