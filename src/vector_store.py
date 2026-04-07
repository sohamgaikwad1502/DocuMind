import logging 
from pathlib import Path

import chromadb

from src.models import Chunk , RetrievedChunk
from src.embedder import Embedder

logger = logging.getLogger(__name__)

class VectorStore:
    def _init__(self,
                persist_dir : str,
                collection_name : str = "documents",
                embedder: Embedder | None = None,
                ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True,exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={'hnsq:space':'cosine'},

        )

        self.embedder = embedder or Embedder()
        logger.info(
            f"VectorStore ready — collection '{collection_name}' "
            f"contains {self.count} chunks"
        )

    def add_chunks(self,chunks:list[Chunk],batch_size: int = 128) -> None :
        if not chunks :
            logger.warning("No chunks to add")
            return 
        
        for i in range(0,len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            ids = [c.chunk_id for c in batch]
            texts = [c.content for c in batch]
            metadatas = [c.metadata for c in batch]
            embeddings = self.embedder.embed_texts(texts)

            self.collection.upsert(
                ids = ids,
                documents=texts,
                metadatas = metadatas,
                embeddings = embeddings
            )

            logger.info(f"Stored batch {i // batch_size + 1 } : {len(batch)} chunks")
        logger.info(f"Total chunks in store : {self.count}")

    def search(self , quert : str , top_k: int = 5 ) -> list[RetrievedChunk] :
        if self.count == 0 :
            return []
        
        query_embedding = self.collection.query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"],
        )
        
        retrieved: list[RetrievedChunk] = []

        for doc, meta, dist, cid in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            similarity = 1.0 - dist

            retrieved.append(RetrievedChunk(
                content=doc,
                chunk_id=cid,
                metadata=meta,
                score=round(similarity, 4),
            ))

        return retrieved
    
    def clear(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared.")

    @property
    def count(self) -> int:
        return self.collection.count()
