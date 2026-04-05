import logging
from pathlib import Path

from src.models import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".html"}

    def load_directory(self, directory: str | Path) -> list[Document]:
        """Load all supported documents from a directory (recursive)."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        documents = []
        for file_path in sorted(directory.rglob("*")):
            if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    docs = self.load_file(file_path)
                    documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} document(s) from {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path.name}: {e}")

        logger.info(f"Total documents loaded: {len(documents)}")
        return documents

    def load_file(self, file_path: str | Path) -> list[Document]:
        """Load a single file. Returns list because PDFs yield one doc per page."""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        loaders = {
            ".pdf": self._load_pdf,
            ".md": self._load_text_file,
            ".txt": self._load_text_file,
            ".html": self._load_html,
        }

        loader = loaders.get(suffix)
        if loader is None:
            raise ValueError(f"Unsupported file type: {suffix}")

        return loader(file_path)

    def load_url(self, url: str) -> list[Document]:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (RAG System Document Loader)"
        })
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = self._clean_text(soup.get_text(separator="\n"))
        title = soup.title.string.strip() if soup.title and soup.title.string else url

        if not text:
            return []

        return [Document(
            content=text,
            metadata={"source": url, "file_type": "web", "title": title}
        )]
    
# private loaders

    def _load_pdf(self, file_path: Path) -> list[Document]:
        import fitz  

        documents = []
        pdf = fitz.open(str(file_path))

        for page_num in range(len(pdf)):
            text = pdf[page_num].get_text().strip()
            if text:
                documents.append(Document(
                    content=text,
                    metadata={
                        "source": file_path.name,
                        "file_type": "pdf",
                        "page": page_num + 1,
                        "total_pages": len(pdf),
                    }
                ))

        pdf.close()
        return documents

    def _load_text_file(self, file_path: Path) -> list[Document]: 
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            return []

        file_type = "markdown" if file_path.suffix == ".md" else "text"
        return [Document(
            content=text,
            metadata={"source": file_path.name, "file_type": file_type}
        )]

    def _load_html(self, file_path: Path) -> list[Document]:
        """Load local HTML file, extracting text content."""
        from bs4 import BeautifulSoup

        html = file_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = self._clean_text(soup.get_text(separator="\n"))
        if not text:
            return []

        title = soup.title.string.strip() if soup.title and soup.title.string else file_path.name

        return [Document(
            content=text,
            metadata={"source": file_path.name, "file_type": "html", "title": title}
        )]

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        
        cleaned = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank:
                    cleaned.append("")
                prev_blank = True
            else:
                cleaned.append(line)
                prev_blank = False
        return "\n".join(cleaned).strip()