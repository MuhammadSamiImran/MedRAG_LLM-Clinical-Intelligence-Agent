from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents(folder_path="medical_docs"):
    loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, separators=["\n\n", "\n", ".", " "])

    chunks = splitter.split_documents(documents)
    return chunks


def show_chunk_info(chunks):
    print(f"Total chunks created: {len(chunks)}")

    sizes = [len(chunk.page_content) for chunk in chunks]
    print(f"Smallest chunk : {min(sizes)} characters")
    print(f"Largest chunk  : {max(sizes)} characters")
    print(f"Average chunk  : {int(sum(sizes)/len(sizes))} characters")

    print("\n--- Example Chunk 1 ---")
    print(f"Source : {chunks[0].metadata.get('source', 'unknown')}")
    print(f"Content: {chunks[0].page_content}")

    print("\n--- Example Chunk 2 ---")
    print(f"Source : {chunks[5].metadata.get('source', 'unknown')}")
    print(f"Content: {chunks[5].page_content}")

    print("\n--- Example Chunk 3 (showing overlap with chunk 2) ---")
    print(f"Source : {chunks[6].metadata.get('source', 'unknown')}")
    print(f"Content: {chunks[6].page_content}")

    print("\n--- Notice ---")
    print("Compare the END of Chunk 2 with the START of Chunk 3.")
    print("You will see the last ~50 characters are repeated.")
    print("That is the overlap working correctly.")


print("Step 1: Loading documents...")
docs = load_documents()
print(f"Loaded {len(docs)} pages\n")

print("Step 2: Splitting into chunks...")
chunks = split_documents(docs)

print("Step 3: Showing chunk information...")
show_chunk_info(chunks)