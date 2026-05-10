# MedRAG — Clinical Intelligence Agent

**MedRAG** is a high-precision Retrieval-Augmented Generation (RAG) system designed to provide medical insights grounded in verified clinical documentation. Unlike standard LLMs that rely on pre-training data, MedRAG uses real-world clinical sources to ensure accuracy and transparency through mandatory source attribution.

---

##  Overview

In the medical domain, accuracy is non-negotiable. MedRAG solves the problem of AI "hallucinations" by forcing the model to only answer based on specific, provided documents. Every response includes citations (filename and page number), ensuring that users can verify information at the source.

---

##  Tech Stack

* **Language:** Python 3.10+
* **LLM:** LLaMA 3.3 70B (Orchestrated via **Groq API** for high-speed inference)
* **RAG Framework:** LangChain
* **Embeddings:** `sentence-transformers` (**all-MiniLM-L6-v2**)
* **Vector Database:** ChromaDB
* **UI:** Streamlit

---

##  How It Works

1. **Vectorization:** The user's query is converted into a semantic vector using HuggingFace embeddings.
2. **Semantic Retrieval:** ChromaDB searches the vector space to find the most relevant chunks of text from pre-loaded medical PDFs.
3. **Context Injection:** These chunks are injected into the LLM prompt as the "ground truth" context.
4. **Grounded Generation:** LLaMA 3.3 70B generates a response based *only* on that context, citing specific sources.
5. **Memory Management:** LangChain maintains conversation history for intuitive follow-up questions.

---

##  Key Features

* **Semantic Search:** Finds information based on meaning rather than just keyword matching.
* **Source Citations:** Automatically provides the filename and page number for every claim made.
* **Multi-turn Conversation:** Remembers previous context for a natural chat experience.
* **Safety Guardrails:** Includes an "Honest Fallback"—if the answer isn't in the documents, the model will state it doesn't know rather than guessing.
* **Streaming UI:** Real-time token streaming for a responsive user experience.

---

##  Project Structure

This project was developed through a modular **9-Stage Framework**, progressing from a basic chatbot to a production-grade RAG agent. Each stage is documented for educational reference:

* `Stage_1_Basic_Chat.py`: API integration and basic LLM interaction.
* `Stage_4_Chunking.py`: Logic for document splitting.
* `Stage_8_Full_RAG.py`: Integration of the retriever and generator.
* `App.py`: The final Streamlit deployment.

---

##  Disclaimer

*This project is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions you may have regarding a medical condition.*
