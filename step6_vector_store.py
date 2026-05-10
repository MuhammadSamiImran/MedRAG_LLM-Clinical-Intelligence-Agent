from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def load_documents(folder_path="medical_docs"):
    loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_vector_store(chunks, embedding_model, persist_directory="chroma_db"):
    print(f"Building vector store from {len(chunks)} chunks...")
    print("This converts every chunk to a vector and saves to disk.")
    print("May take 1-3 minutes depending on how many documents you have.\n")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory)

    total_stored = vectorstore._collection.count()
    print(f"Successfully stored {total_stored} chunks in vector store.")
    print(f"Database saved to folder: '{persist_directory}'\n")

    return vectorstore


def test_vector_store(vectorstore):
    print("--- Testing the vector store with sample searches ---\n")

    test_questions = [
        "What causes down syndrome?",
        "How is drug inducing treated?",
        "What are the symptoms of depression?"
    ]

    for question in test_questions:
        print(f"Question: '{question}'")

        results = vectorstore.similarity_search(question, k=2)

        for i, doc in enumerate(results):
            source = doc.metadata.get("source", "unknown")
            # Clean up the path to show just the filename
            filename = source.split("\\")[-1].split("/")[-1]
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"  Result {i+1} | File: {filename}")
            print(f"  Preview: {preview}...")
        print()


def load_existing_vector_store(embedding_model, persist_directory="chroma_db"):
    print(f"Loading existing vector store from '{persist_directory}'...")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    total = vectorstore._collection.count()
    print(f"Loaded {total} chunks from vector store.\n")
    return vectorstore


print("=== STEP 6: Vector Store ===\n")

print("Loading documents...")
docs = load_documents()
print(f"Loaded {len(docs)} pages\n")

print("Splitting into chunks...")
chunks = split_documents(docs)
print(f"Created {len(chunks)} chunks\n")

print("Loading embedding model...")
embedding_model = get_embedding_model()
print("Embedding model ready.\n")

vectorstore = build_vector_store(chunks, embedding_model)

test_vector_store(vectorstore)

print("=== Vector store is ready ===")
