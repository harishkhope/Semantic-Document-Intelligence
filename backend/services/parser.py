import re
import fitz  # PyMuPDF


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def parse_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    raw = "\n".join(pages)
    return _clean_text(raw)


def parse_txt(file_bytes: bytes) -> str:
    raw = file_bytes.decode("utf-8", errors="replace")
    return _clean_text(raw)


def parse_file(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return parse_pdf(file_bytes)
    elif ext == "txt":
        return parse_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Only .pdf and .txt are supported.")
