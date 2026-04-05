import hashlib 
import logging 
from typing import Any 

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import Document, Chunk 

logger = logging.getLogger(__name__)

class DocumentChunker : 
    def __init__(self ,chunk_size : int = 600 , chunk_overlap: int = 100):
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap 
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        )
    def chunk_documents(self,documents:list[Document]) :
        all_chunks:list[Chunk] = []

        for doc in documents :
            chunks = self._chunk_single(doc)
            all_chunks.extend(chunks)

        logger.info(
            f"Chunked {len(documents)} document(s)"
            f"(target:{self.chunk_size} tokens, overlap : {self.chunk_overlap})"

        )
        return all_chunks

    def _chunk_single(self,document:Document) -> list[Chunk] :
        text = document.content.strip()
        if not text : 
            return []
        
        text_chunk = self.splitter.split_text(text)

        chunks = []
        for idx , chunk_text in enumerate(text_chunk):
            chunk_id = self._make_id(document.metadata.get("source","unknown"), idx, chunk_text)
            chunks.append(Chunk(
                content=chunk_text,
                chunk_id=chunk_id,
                metadata={
                    **document.metadata,
                    "chunk_index": idx, 
                    "total_chunks" : len(text_chunk)
                }
            ))

        return chunks
    
    @staticmethod
    def _make_id(source:str,index:int,content:str) -> str:
        raw = f"{source}::chunk_{index} :: {content[:80]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]