# import logging

# from typing import List
# from pinecone import Pinecone
# from pinecone_text.sparse import BM25Encoder
# from langchain_community.retrievers import PineconeHybridSearchRetriever
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# from backend.core.config import settings

# class HybridRetriever:
    
#     def __init__(self):
#          self.logger = logging.getLogger("app")
#          self.top_k = settings.TOP_K
#          self.alpha = settings.ALPHA


#     def retrieve(self, query: str, namespace: str, top_k: int, alpha: float):
#         """Create a Pinecone hybrid search retriever."""
#         embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
#         query = embeddings.embed_query(query)
#         namespace = settings.PINECONE_NAMESPACE


#         total_fetched_result = []

#         try:
#                 pc = Pinecone(settings.PINECONE_API_KEY)
#                 index = pc.Index(settings.PINECONE_INDEX_NAME)
#                 bm25_encoder = BM25Encoder().default()
#                 total_fetched_result = []

#                 retriever = PineconeHybridSearchRetriever(
#                     embeddings= embeddings,
#                     sparse_encoder=bm25_encoder,
#                     index=index
#                     namespace=namespace,
#                     top_k=top_k,
#                     alpha=alpha,
#                     text_key="text",
#                 )
#                 result = retriever.invoke(query)

#                 for doc in result:
#                     fetched_result = [doc.page_content]
#                     total_fetched_result.extend(fetched_result)

#                 return total_fetched_result
#         except Exception as e:
#                 return (f"Failed to create retriever: {e}")