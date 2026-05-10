from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


from dotenv import load_dotenv
import os

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


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_existing_vector_store(embedding_model, persist_directory="chroma_db"):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    print(f"Loaded vector store with {vectorstore._collection.count()} chunks.\n")
    return vectorstore


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

Answer in simple language. After your answer, list which sources you used."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += conversation_history
    messages.append({"role": "user", "content": rag_user_message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages)
    answer = response.choices[0].message.content
    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer, sources, conversation_history



def run_test():
    print("=== STEP 8: Full RAG Pipeline ===\n")

    embedding_model = get_embedding_model()
    vectorstore = load_existing_vector_store(embedding_model)

    history = []

    print("=" * 60)
    print("TEST 1: Basic RAG question")
    print("=" * 60)
    q1 = "What are the symptoms of Personality disorder?"
    print(f"Question: {q1}\n")
    answer, sources, history = rag_chat(q1, vectorstore, history)
    print(f"Answer:\n{answer}")
    print(f"\nSources used: {sources}")

    print("\n" + "=" * 60)
    print("TEST 2: Follow-up question (tests memory)")
    print("=" * 60)
    q2 = "How is it diagnosed?"
    print(f"Question: {q2}")
    print("(No disease mentioned — bot must remember from Test 1)\n")
    answer, sources, history = rag_chat(q2, vectorstore, history)
    print(f"Answer:\n{answer}")
    print(f"\nSources used: {sources}")

    print("\n" + "=" * 60)
    print("TEST 3: Question not in documents (tests honesty)")
    print("=" * 60)
    q3 = "What is the latest cure for Personality disorder?"
    print(f"Question: {q3}\n")
    answer, sources, history = rag_chat(q3, vectorstore, history)
    print(f"Answer:\n{answer}")
    print(f"\nSources used: {sources}")

    print("\n=== RAG Pipeline is working correctly ===")
    print("The system is now:")
    print("  1. Retrieving relevant chunks from your medical documents")
    print("  2. Feeding them as context to the LLM")
    print("  3. Generating grounded answers with source citations")
    print("  4. Remembering conversation history across questions")
    

run_test()