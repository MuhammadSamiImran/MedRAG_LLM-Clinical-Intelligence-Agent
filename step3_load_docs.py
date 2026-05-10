from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

def load_documents(folder_path="medical_docs"):
    print(f"Loading documents from: {folder_path}")

    
    loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    print(f"\nTotal pages loaded: {len(documents)}")

    print("\n--- Documents found ---")
    seen_sources = set()
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        if source not in seen_sources:
            print(f"  File: {source}")
            seen_sources.add(source)

    print("\n--- Sample of first page ---")
    print(documents[0].page_content[:500])

    return documents


docs = load_documents()