from dataclasses import dataclass , field 
from typing import Any

@dataclass
class Document : 
    content: str
    metadata: dict[str,Any] = field(default_factory=dict)

@dataclass
class Chunk : 
    content : str 
    chunk_id : str
    metadata: dict[str,Any] = field(default_factory=dict)

@dataclass
class RetrievedChunk: 
    content:str
    chunk_id:str
    metadata : dict[str,Any]
    score: float

@dataclass
class RagResponse :
    answer : str
    retrieved_chunks : list[RetrievedChunk]
    query : str

    
