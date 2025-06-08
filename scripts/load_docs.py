from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss
import os

documents = SimpleDirectoryReader("articles").load_data()

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

d = 384
faiss_index = faiss.IndexFlatL2(d)
faiss.write_index(faiss_index, "index/faiss.index")
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)

index.storage_context.persist(persist_dir="index")
print("Local embedding-based FAISS index created.")
