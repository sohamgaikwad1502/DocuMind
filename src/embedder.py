import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class Embedder : 
    def __init__(self,model_name : str = "all-MiniLM-L6-v2") :
        logger.info(f"Loading embedding model : {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model ready - dimension : {self.dimension}")

    def embed_texts(self,texts: list[str] , batch_size: int = 64) -> list[list[float]] :
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy = True,
            normalize_embeddings = True,
        )
        return embeddings.tolist()
    
    def embed_query(self,query:str) -> list[float]:
        embedding = self.model.encode(
            query,
            convert_to_numpy = True, 
            normalize_embeddings= True,

        )

        return embedding.tolist()