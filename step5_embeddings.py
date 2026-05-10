from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


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
    print("Loading embedding model...")

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    return model


def show_embedding_info(embedding_model):
    
    sample_text = "What causes down syndrome?"
    vector = embedding_model.embed_query(sample_text)

    print(f"--- What an embedding looks like ---")
    print(f"Input text : '{sample_text}'")
    print(f"Vector size: {len(vector)} numbers")
    print(f"First 5 numbers: {[round(n, 6) for n in vector[:5]]}")
    print(f"Last  5 numbers: {[round(n, 6) for n in vector[-5:]]}")
    print()

    
    import numpy as np

    def cosine_similarity(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return round(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))), 4)

    text_a = "mental disorder and high body temperature"
    text_b = "patient has different personalities"   
    text_c = "how to cook pasta at home"          

    vec_a = embedding_model.embed_query(text_a)
    vec_b = embedding_model.embed_query(text_b)
    vec_c = embedding_model.embed_query(text_c)

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)

    print("--- Semantic similarity test ---")
    print(f"Text A: '{text_a}'")
    print(f"Text B: '{text_b}'  (similar meaning)")
    print(f"Text C: '{text_c}'  (different meaning)")
    print()
    print(f"Similarity A vs B: {sim_ab}")  
    print(f"Similarity A vs C: {sim_ac}")
    print()


print("=== STEP 5: Embeddings ===\n")

print("Loading and splitting documents...")
docs = load_documents()
chunks = split_documents(docs)
print(f"Working with {len(chunks)} chunks\n")

embedding_model = get_embedding_model()

show_embedding_info(embedding_model)