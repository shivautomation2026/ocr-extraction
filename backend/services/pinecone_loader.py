import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.documents import Document
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconeLoader:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
        self.index_name = 'sap-items'
        self.index = self.pc.Index(self.index_name)
        self.df = pd.read_csv("backend/assets/item_list.csv")
        self.df.dropna(subset=['ItemName'], inplace=True)
        logger.info("Successfully loaded the CSV file and removed NaN values")
        
        self.hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
    def load_items(self):
        codes = ['ME', 'EL', 'HE', 'CV', 'MS', 'AR', 'HS', 'FG', 'FL', 'PS', 'TR', 'RM', 'IT', 'BP', 'RR', 'OT', 'AS', 'SV']
        self.loaders = []
        try:
            for code in codes:
                items = self.df[self.df['ItemCode'].str.startswith(code)]
                per_item_loader = DataFrameLoader(items, page_content_column="ItemName").load()
                self.loaders.append(per_item_loader)
            logger.info("Successfully created DataFrame loaders for all categories")
            return self.loaders
        except Exception as e:
            logger.error(f"Error while loading items: {e}")
            raise e
    
    def line_item_loader(self, item: dict):
        try:
            item_format = Document(
                page_content = item['ItemName'],
                metadata = {
                    'ItemCode': item['ItemCode'],
                    'ItemCategory': item['ItemCategory'],
                    'UoM': item['UoM'],
                }
            )
            logger.info("Successfully created Document for line item")
            return item_format
        except Exception as e:
            logger.error(f"Error while creating line item Document: {e}")
            raise e
    
    def initialize_pinecone(self, dimension: int = 768):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = 'sap-items'
        if not pc.has_index(index_name):
            pc.create_index(
                name = index_name,
                dimension = dimension,
                metric = 'dotproduct',
                vector_type = 'dense',
            )
            self.index = pc.Index(index_name)
            return self.index
        self.index = pc.Index(index_name)

        logger.info('Pinecone index is ready to use')
        
        return self.index
     
    def store_in_pinecone(self, loader: list[Document] = None, line_item: dict = None, namespace: str = None):
        index = self.initialize_pinecone()
        
        categories = ["mechanical", "electrical", "heavy_equipments_and_automobiles", "civil", "mesh", "auxiliary_raw_material", "health_and_safety", "finished_goods", "fuel_lubricant_and_gas", "printing_and_stationary", "trading", "raw_material", "it_and_accessories", "business_promotion_and_marketing", "wastages", "other", "asset", "service"]
        try:
            if loader:
                for i in range(len(loader)):
                    vector_store = PineconeVectorStore(embedding=self.hf_embeddings, index = index, namespace = categories[i])
                    
                    vector_store.add_documents(documents = loader[i])
                logger.info("Successfully loaded all items in pinecone")
            else:
                vector_store = PineconeVectorStore(embedding=self.hf_embeddings, index = index, namespace = namespace)
                logger.info(f"Storing single line item: {line_item} in pinecone")
                vector_store.add_documents(documents = [self.line_item_loader(line_item)])
                logger.info("Successfully loaded line item in pinecone")
        except Exception as e:
            logger.error(f"Error while storing in pinecone: {e}")
            raise e
        
if __name__ == "__main__":
    pinecone = PineconeLoader()
    csv_items = pinecone.load_items()
    pinecone.store_in_pinecone(loader=csv_items)
    
            
            
        
    
        
               