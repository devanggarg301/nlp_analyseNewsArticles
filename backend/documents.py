"""Text extraction for article documents, never executes embedded content."""
import io
import zipfile
from pathlib import Path

MAX_UPLOAD=10_000_000

def extract_document(name,data):
    extension=Path(name).suffix.lower()
    title=Path(name).stem
    warnings=[]
    if extension==".pdf":
        from pypdf import PdfReader
        try:
            reader=PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("This PDF is password-protected. Upload an unlocked copy.")
            if len(reader.pages)>30:
                raise ValueError("Upload a PDF with at most 30 pages, containing one news article.")
            text="\n\n".join(page.extract_text() or "" for page in reader.pages)
            title=str((reader.metadata or {}).get("/Title") or title)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("The PDF could not be read. Try saving it again as a text-based PDF.") from exc
        if len(text.split())<15:
            raise ValueError("This PDF has no usable text and may be scanned. Use an OCR/text-based PDF or paste the article text.")
    elif extension==".docx":
        from docx import Document
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if sum(info.file_size for info in archive.infolist())>30_000_000:
                    raise ValueError("The expanded Word document is too large.")
            doc=Document(io.BytesIO(data))
            # iter_inner_content retains paragraph/table order in the Word document.
            blocks=[]
            from docx.text.paragraph import Paragraph
            for item in doc.iter_inner_content():
                if isinstance(item,Paragraph):
                    if item.text.strip(): blocks.append(item.text)
                else:
                    blocks.extend(" | ".join(cell.text for cell in row.cells) for row in item.rows)
            text="\n\n".join(blocks)
            title=doc.core_properties.title or title
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("The Word document could not be read. Save it as a .docx file and try again.") from exc
    elif extension==".doc":
        raise ValueError("Older .doc files must first be saved as .docx or PDF in Word.")
    else:
        raise ValueError("Choose PDF, Word (.docx), TXT or CSV.")
    if len(text.strip())<30:
        raise ValueError("The document does not contain enough article text.")
    if len(text.split())>3000:
        warnings.append("This document exceeds 3,000 words. Shorten the editable preview before adding it.")
    return {"title":title[:300],"publisher":"","text":text,"source_type":"upload","url":None,"published_at":None,"warnings":warnings}

