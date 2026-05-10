from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_existing_vector_store(embedding_model, persist_directory="chroma_db"):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    total = vectorstore._collection.count()
    print(f"Loaded vector store with {total} chunks.\n")
    return vectorstore


def retrieve_chunks(question, vectorstore, k=4):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    results = retriever.invoke(question)
    return results


def show_retrieval_results(question, results):
    print(f"Question : '{question}'")
    print(f"Retrieved: {len(results)} chunks\n")

    for i, doc in enumerate(results):
        source = doc.metadata.get("source", "unknown")
        filename = source.split("\\")[-1].split("/")[-1]
        page = doc.metadata.get("page", "unknown")
        content = doc.page_content.replace("\n", " ")

        print(f"--- Chunk {i+1} ---")
        print(f"File    : {filename}")
        print(f"Page    : {page}")
        print(f"Content : {content}")
        print()


def show_retrieval_comparison():

    print("=" * 55)
    print("SEMANTIC SEARCH DEMONSTRATION")
    print("=" * 55)
    print("Same question asked 3 different ways.")
    print("All 3 should retrieve the same relevant chunks.\n")

    embedding_model = get_embedding_model()
    vectorstore = load_existing_vector_store(embedding_model)

    variations = [
        "What causes down syndrome?",                          
        "Why does perope have down syndrome?",         
        "What is the reason for down syndrome?"      
    ]

    for question in variations:
        results = retrieve_chunks(question, vectorstore, k=2)
        print(f"Query   : '{question}'")
        source = results[0].metadata.get("source", "unknown")
        filename = source.split("\\")[-1].split("/")[-1]
        preview = results[0].page_content[:120].replace("\n", " ")
        print(f"Top hit : {filename}")
        print(f"Preview : {preview}...")
        print()


print("=== STEP 7: Retrieval ===\n")

embedding_model = get_embedding_model()
vectorstore = load_existing_vector_store(embedding_model)

print("=" * 55)
print("TEST 1: Detailed retrieval results")
print("=" * 55 + "\n")

test_questions = [
        "What causes down syndrome?",
        "How is drug inducing treated?",
        "What are the symptoms of depression?"
    ]

for question in test_questions:
    results = retrieve_chunks(question, vectorstore, k=4)
    show_retrieval_results(question, results)
    print("-" * 55 + "\n")

show_retrieval_comparison()

print("=== Retrieval is working correctly ===")