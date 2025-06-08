from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever


def get_relevant_chunks(query, top_k=3):
    vector_store = FaissVectorStore.from_persist_dir("index")
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir="index",
    )
    index = load_index_from_storage(
        storage_context=storage_context, embed_model=embed_model)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    return [node.get_content() for node in nodes]
