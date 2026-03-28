import os
import streamlit as st
import cohere
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# Chargement des variables d'environnement
load_dotenv()


# -----------------------------
# Configuration de la page
# -----------------------------
st.set_page_config(
    page_title="Chatbot RAG - Assistant Utilisateur",
    page_icon="🤖",
    layout="centered"
)


# -----------------------------
# Fonctions principales
# -----------------------------
@st.cache_resource
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
    return vectorstore.similarity_search(user_question, k=k)


def build_context_text(retrieved_docs):
    """
    Transforme les documents retrouvés en texte de contexte.
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
    Génère une réponse avec Cohere.
    """
    api_key = os.getenv("COHERE_API_KEY")

    if not api_key:
        raise ValueError("COHERE_API_KEY introuvable dans le fichier .env")

    co = cohere.ClientV2(api_key=api_key)

    system_message = (
        "Tu es un assistant utilisateur pour une application d'analyse documentaire. "
        "Réponds toujours en français. "
        "Explique simplement, clairement, sans jargon technique. "
        "Aide un utilisateur non informaticien. "
        "Base-toi uniquement sur le contexte fourni. "
        "Si l'information n'est pas dans le contexte, dis-le honnêtement. "
        "Donne des étapes pratiques quand c'est utile. "
        "Ne réponds pas sur les détails techniques internes."
    )

    response = co.chat(
        model="command-r-plus-08-2024",
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
    question -> retrieval -> contexte -> génération
    """
    vectorstore = load_vector_store()
    retrieved_docs = retrieve_context(vectorstore, user_question, k=3)
    context_text = build_context_text(retrieved_docs)
    answer = generate_answer(user_question, context_text)

    return answer, retrieved_docs, context_text


# -----------------------------
# Interface Streamlit
# -----------------------------
st.title("🤖 Assistant Utilisateur - Chatbot RAG")
st.write(
    "Pose une question sur l'utilisation de l'application, "
    "l'interprétation des résultats ou les problèmes fréquents."
)

# Initialisation historique
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie utilisateur
user_question = st.chat_input("Écris ta question ici...")

if user_question:
    # Afficher message utilisateur
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Réponse assistant
    with st.chat_message("assistant"):
        with st.spinner("Recherche de la meilleure réponse..."):
            try:
                answer, retrieved_docs, context_text = ask_rag(user_question)
                st.markdown(answer)

                with st.expander("Voir les sources utilisées"):
                    for i, doc in enumerate(retrieved_docs, start=1):
                        st.markdown(f"**Source {i} :** {doc.metadata.get('source', 'inconnue')}")
                        st.write(doc.page_content)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                error_message = f"Une erreur s'est produite : {str(e)}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )