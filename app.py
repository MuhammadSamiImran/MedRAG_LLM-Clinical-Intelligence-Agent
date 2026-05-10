import streamlit as st
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os



st.set_page_config(
    page_title="MedRAG — Clinical Intelligence Agent",
    page_icon="🏥",
    layout="centered"
)


@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_vector_store():
    embedding_model = load_embedding_model()
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


SYSTEM_PROMPT = """You are MedRAG, a helpful medical information assistant.
You answer questions based on the medical documents provided to you.
Explain everything in simple, clear language that anyone can understand.

IMPORTANT RULES:
- Only answer based on the context provided to you
- If the answer is not in the context, say honestly:
  "I could not find information about this in my medical documents."
- Never provide a personal diagnosis
- Always recommend consulting a real doctor for personal health concerns
- If asked about emergencies, tell the user to call emergency services immediately
"""


def rag_chat(question, vectorstore, conversation_history):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(question)

    context_parts = []
    for i, doc in enumerate(relevant_docs):
        source = doc.metadata.get("source", "unknown")
        filename = source.split("\\")[-1].split("/")[-1]
        page = doc.metadata.get("page", "?")
        context_parts.append(
            f"[Source {i+1}: {filename}, page {page}]\n{doc.page_content}"
        )
    context = "\n\n".join(context_parts)

    # Collect source filenames
    sources = list(set([
        doc.metadata.get("source", "unknown").split("\\")[-1].split("/")[-1]
        for doc in relevant_docs
    ]))

    rag_user_message = f"""Use the following medical document excerpts to answer the question.
If the answer is not clearly present in the excerpts, say so honestly.

MEDICAL DOCUMENT EXCERPTS:
{context}

PATIENT QUESTION:
{question}

Answer in simple language. Be clear and structured."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history
    messages.append({"role": "user", "content": rag_user_message})

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        stream=True
    )

    full_response = ""
    placeholder = st.empty()
    for chunk in stream:
        if chunk.choices[0].delta.content:
            full_response += chunk.choices[0].delta.content
            placeholder.markdown(full_response + "▌")
    placeholder.markdown(full_response)

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": full_response})

    return full_response, sources, conversation_history


st.title("MedRAG")
st.caption("Clinical Intelligence Agent — answers grounded in real medical documents")

st.warning(
    "For informational purposes only. "
    "Always consult a qualified doctor for personal medical advice."
)

st.divider()


if "messages" not in st.session_state:
    st.session_state.messages = []         
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []      

vectorstore = load_vector_store()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used from medical documents"):
                for src in message["sources"]:
                    st.markdown(f"- `{src}`")


if question := st.chat_input("Ask a medical question..."):

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("assistant"):
        answer, sources, updated_history = rag_chat(
            question,
            vectorstore,
            st.session_state.rag_history
        )
        st.session_state.rag_history = updated_history

        if sources:
            with st.expander("Sources used from medical documents"):
                for src in sources:
                    st.markdown(f"- `{src}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })


with st.sidebar:
    st.header("About MedRAG")
    st.markdown("""
    **MedRAG** is a Retrieval-Augmented Generation (RAG) 
    system built for medical information.
    
    **How it works:**
    1. Your question is converted to a vector
    2. Similar chunks are retrieved from medical documents
    3. Those chunks are sent to the LLM as context
    4. The LLM answers using only that context
                
    **Tech Stack:**
    - LLM: LLaMA 3.3 70B via Groq
    - Embeddings: sentence-transformers
    - Vector DB: ChromaDB
    - Framework: LangChain
    - UI: Streamlit
    """)

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.session_state.rag_history = []
        st.rerun()

    st.divider()
    st.caption("Built by Sami Imran — MedRAG Portfolio Project")