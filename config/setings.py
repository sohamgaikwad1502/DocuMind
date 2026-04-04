from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma_db"
PROMPTS_FILE = ROOT_DIR / "config" / "prompts.yaml"

CHUNK_SIZE = 600 
CHUNK_OVERLAP = 100


EMBEDDING_MODEL = "all_Mini-L6-v2"

COLLECTION_NAME = "documents"

TOP_K = 5

LLM_PROVIDER = "groq"
OLLAMA_MODEL = "llama3.1"
OLLAMA_BASE_URL = "http://localhost:11434"

LOG_LEVEL = "INFO"
