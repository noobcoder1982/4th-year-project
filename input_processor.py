"""
=============================================================================
TruthLens - Advanced Multi-Modal Article Input Processing System
=============================================================================
Supports:
1. Direct text / Paste
2. PDF files (.pdf) with paragraph and hyphenation reconstruction
3. Microsoft Word (.docx and legacy .doc)
4. Plain Text and Markdown (.txt, .text, .md)
5. Live Article URLs (HTML scraping, boilerplate removal, main body extraction)
=============================================================================
"""

import os
import re
import urllib.parse
from typing import Dict, Any, Optional
import requests
from bs4 import BeautifulSoup


# =============================================================================
# TEXT CLEANING & STRUCTURE PRESERVATION
# =============================================================================

def clean_extracted_text(text: str) -> str:
    """Clean extracted document text while preserving paragraph structure."""
    if not text:
        return ""

    # Replace carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix hyphens at end of lines (e.g., "inter-\nnational" -> "international")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Normalize multiple whitespace within lines while preserving paragraph breaks
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    # Group into paragraphs (consecutive non-empty lines merged with space, empty lines separate paragraphs)
    paragraphs = []
    current_para = []

    for line in lines:
        if line:
            current_para.append(line)
        else:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []

    if current_para:
        paragraphs.append(" ".join(current_para))

    # Join paragraphs with double newline
    result = "\n\n".join(paragraphs).strip()
    return result


# =============================================================================
# 1. DIRECT TEXT PROCESSOR
# =============================================================================

def extract_from_direct_text(text: str) -> Dict[str, Any]:
    """Process direct text or pasted content."""
    clean = clean_extracted_text(text)
    if not clean:
        return {
            "success": False,
            "text": "",
            "title": "Direct Text",
            "source_type": "text",
            "source_name": "Pasted Text",
            "word_count": 0,
            "error": "Input text is empty. Please provide article content."
        }

    words = clean.split()
    return {
        "success": True,
        "text": clean,
        "title": "Direct Text Input",
        "source_type": "text",
        "source_name": "Pasted Text",
        "word_count": len(words),
        "error": None
    }


# =============================================================================
# 2. PDF PROCESSOR
# =============================================================================

def extract_from_pdf(file_path: str) -> Dict[str, Any]:
    """Extract clean readable text from a PDF document."""
    if not file_path or not os.path.exists(file_path):
        return {
            "success": False,
            "text": "",
            "title": "",
            "source_type": "pdf",
            "source_name": os.path.basename(file_path) if file_path else "",
            "word_count": 0,
            "error": "PDF file not found or path is invalid."
        }

    filename = os.path.basename(file_path)

    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                return {
                    "success": False,
                    "text": "",
                    "title": filename,
                    "source_type": "pdf",
                    "source_name": filename,
                    "word_count": 0,
                    "error": f"The PDF file '{filename}' is password-protected and cannot be read."
                }

        page_texts = []
        for page_num, page in enumerate(reader.pages, 1):
            try:
                t = page.extract_text()
                if t and t.strip():
                    page_texts.append(t.strip())
            except Exception as pe:
                print(f"Warning: Failed to extract page {page_num}: {pe}")

        if not page_texts:
            return {
                "success": False,
                "text": "",
                "title": filename,
                "source_type": "pdf",
                "source_name": filename,
                "word_count": 0,
                "error": (
                    f"No selectable text found in '{filename}'. "
                    "The PDF may be a scanned image without an OCR text layer."
                )
            }

        raw_combined = "\n\n".join(page_texts)
        cleaned_text = clean_extracted_text(raw_combined)

        # Attempt to read PDF metadata title
        pdf_title = filename
        if reader.metadata and reader.metadata.title:
            pdf_title = reader.metadata.title.strip()

        return {
            "success": True,
            "text": cleaned_text,
            "title": pdf_title or filename,
            "source_type": "pdf",
            "source_name": filename,
            "word_count": len(cleaned_text.split()),
            "page_count": len(reader.pages),
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "title": filename,
            "source_type": "pdf",
            "source_name": filename,
            "word_count": 0,
            "error": f"Failed to parse PDF '{filename}': {str(e)}"
        }


# =============================================================================
# 3. DOC / DOCX PROCESSOR
# =============================================================================

def extract_from_docx(file_path: str) -> Dict[str, Any]:
    """Extract clean readable text from a modern Microsoft Word (.docx) document."""
    if not file_path or not os.path.exists(file_path):
        return {
            "success": False,
            "text": "",
            "title": "",
            "source_type": "docx",
            "source_name": os.path.basename(file_path) if file_path else "",
            "word_count": 0,
            "error": "Word document not found or path is invalid."
        }

    filename = os.path.basename(file_path)

    try:
        import docx
        doc = docx.Document(file_path)

        paragraphs = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if text:
                paragraphs.append(text)

        # Also extract table text
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    paragraphs.append(" | ".join(row_cells))

        if not paragraphs:
            return {
                "success": False,
                "text": "",
                "title": filename,
                "source_type": "docx",
                "source_name": filename,
                "word_count": 0,
                "error": f"The Word document '{filename}' is empty."
            }

        cleaned_text = clean_extracted_text("\n\n".join(paragraphs))

        # Title from first heading or filename
        doc_title = filename
        if doc.paragraphs and doc.paragraphs[0].text.strip():
            first_p = doc.paragraphs[0].text.strip()
            if len(first_p) <= 120 and "\n" not in first_p:
                doc_title = first_p

        return {
            "success": True,
            "text": cleaned_text,
            "title": doc_title,
            "source_type": "docx",
            "source_name": filename,
            "word_count": len(cleaned_text.split()),
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "text": "",
            "title": filename,
            "source_type": "docx",
            "source_name": filename,
            "word_count": 0,
            "error": f"Failed to parse Word document '{filename}': {str(e)}"
        }


def extract_from_doc(file_path: str) -> Dict[str, Any]:
    """
    Extract text from legacy .doc (Word 97-2003 binary format).
    Attempts utf-16/ascii text block extraction with fallback instruction.
    """
    filename = os.path.basename(file_path)

    # First check if it's actually a docx file with a .doc extension
    try:
        import docx
        doc_test = docx.Document(file_path)
        return extract_from_docx(file_path)
    except Exception:
        pass

    try:
        with open(file_path, "rb") as f:
            content = f.read()

        # Extract ASCII and UTF-16LE printable strings of 5+ chars
        ascii_strings = re.findall(b"[\x20-\x7E\t\n\r]{5,}", content)
        utf16_strings = re.findall(b"(?:[\x20-\x7E]\x00){5,}", content)

        extracted_parts = []
        for s in utf16_strings:
            try:
                decoded = s.decode("utf-16le", errors="ignore").strip()
                if len(decoded.split()) >= 3:
                    extracted_parts.append(decoded)
            except Exception:
                pass

        if not extracted_parts:
            for s in ascii_strings:
                try:
                    decoded = s.decode("latin-1", errors="ignore").strip()
                    # Filter out binary metadata keywords
                    if len(decoded.split()) >= 3 and not re.search(r"Microsoft Word|Normal\.dot|CompObj", decoded):
                        extracted_parts.append(decoded)
                except Exception:
                    pass

        if extracted_parts:
            combined = clean_extracted_text("\n\n".join(extracted_parts))
            if len(combined.split()) >= 15:
                return {
                    "success": True,
                    "text": combined,
                    "title": filename,
                    "source_type": "doc",
                    "source_name": filename,
                    "word_count": len(combined.split()),
                    "error": None
                }

        return {
            "success": False,
            "text": "",
            "title": filename,
            "source_type": "doc",
            "source_name": filename,
            "word_count": 0,
            "error": (
                f"'{filename}' is a legacy binary Word 97-2003 (.doc) file. "
                "Please save/convert it to modern .docx format or paste its text directly."
            )
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "title": filename,
            "source_type": "doc",
            "source_name": filename,
            "word_count": 0,
            "error": f"Failed to read legacy .doc file '{filename}': {str(e)}"
        }


# =============================================================================
# 4. PLAIN TEXT / MARKDOWN PROCESSOR
# =============================================================================

def extract_from_txt(file_path: str) -> Dict[str, Any]:
    """Extract clean readable text from plain text or Markdown files (.txt, .md, .text)."""
    if not file_path or not os.path.exists(file_path):
        return {
            "success": False,
            "text": "",
            "title": "",
            "source_type": "txt",
            "source_name": os.path.basename(file_path) if file_path else "",
            "word_count": 0,
            "error": "Text file not found or path is invalid."
        }

    filename = os.path.basename(file_path)
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

    raw_text = None
    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as f:
                raw_text = f.read()
            break
        except (UnicodeDecodeError, Exception):
            continue

    if raw_text is None:
        return {
            "success": False,
            "text": "",
            "title": filename,
            "source_type": "txt",
            "source_name": filename,
            "word_count": 0,
            "error": f"Could not decode text file '{filename}'. Unsupported file encoding."
        }

    clean = clean_extracted_text(raw_text)
    if not clean:
        return {
            "success": False,
            "text": "",
            "title": filename,
            "source_type": "txt",
            "source_name": filename,
            "word_count": 0,
            "error": f"The file '{filename}' contains no readable text."
        }

    return {
        "success": True,
        "text": clean,
        "title": filename,
        "source_type": "txt",
        "source_name": filename,
        "word_count": len(clean.split()),
        "error": None
    }


# =============================================================================
# 5. ARTICLE URL PROCESSOR
# =============================================================================

def extract_from_url(url: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Fetch an online article from a webpage URL.
    Extracts article title, removes ads/navigation/scripts, and cleans readable body paragraphs.
    """
    url = (url or "").strip()
    if not url:
        return {
            "success": False,
            "text": "",
            "title": "",
            "source_type": "url",
            "source_name": "",
            "word_count": 0,
            "error": "No URL provided."
        }

    # Ensure http/https scheme
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    # Parse and validate domain
    parsed = urllib.parse.urlparse(url)
    if not parsed.netloc or "." not in parsed.netloc:
        return {
            "success": False,
            "text": "",
            "title": "",
            "source_type": "url",
            "source_name": url,
            "word_count": 0,
            "error": f"Invalid URL '{url}'. Please provide a valid web address (e.g., https://example.com/article)."
        }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36 TruthLens/2.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract title
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        if not title:
            title = parsed.netloc

        # Strip undesirable tags
        for unwanted in soup([
            "script", "style", "noscript", "nav", "footer", "header",
            "aside", "form", "iframe", "svg", "button", "figure", "figcaption"
        ]):
            unwanted.decompose()

        # Remove elements with common ad/sidebar class or id names
        ad_patterns = re.compile(
            r"comment|sidebar|advertisement|banner|social-share|related-posts|nav-|footer-|cookie|newsletter",
            re.I
        )
        for element in soup.find_all(attrs={"class": ad_patterns}):
            element.decompose()
        for element in soup.find_all(attrs={"id": ad_patterns}):
            element.decompose()

        # Prioritize main article container
        article_elem = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"})
        target_container = article_elem if article_elem else soup.body or soup

        # Extract paragraphs
        paragraphs = []
        for p in target_container.find_all(["p", "h2", "h3", "h4"]):
            p_text = p.get_text(separator=" ", strip=True)
            # Filter out tiny links/buttons/copyright notes
            if len(p_text.split()) >= 4 and not re.search(r"all rights reserved|cookie policy|sign in", p_text, re.I):
                paragraphs.append(p_text)

        if not paragraphs:
            # Fallback: extract text from target container directly
            raw_text = target_container.get_text(separator="\n", strip=True)
            paragraphs = [line.strip() for line in raw_text.split("\n") if len(line.strip().split()) >= 5]

        cleaned_body = clean_extracted_text("\n\n".join(paragraphs))

        if not cleaned_body or len(cleaned_body.split()) < 10:
            return {
                "success": False,
                "text": "",
                "title": title,
                "source_type": "url",
                "source_name": parsed.netloc,
                "word_count": 0,
                "error": (
                    f"Could not extract article text from '{url}'. "
                    "The page may require JavaScript rendering, a subscription login, or CAPTCHA verification."
                )
            }

        return {
            "success": True,
            "text": cleaned_body,
            "title": title,
            "source_type": "url",
            "source_name": parsed.netloc,
            "url": url,
            "word_count": len(cleaned_body.split()),
            "error": None
        }

    except requests.exceptions.HTTPError as he:
        status_code = getattr(he.response, "status_code", "Unknown")
        return {
            "success": False,
            "text": "",
            "title": "",
            "source_type": "url",
            "source_name": parsed.netloc,
            "word_count": 0,
            "error": f"Failed to access webpage (HTTP {status_code}): {url}"
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "text": "",
            "title": "",
            "source_type": "url",
            "source_name": parsed.netloc,
            "word_count": 0,
            "error": f"Connection timed out while fetching article from '{url}'."
        }
    except requests.exceptions.RequestException as re_err:
        return {
            "success": False,
            "text": "",
            "title": "",
            "source_type": "url",
            "source_name": parsed.netloc,
            "word_count": 0,
            "error": f"Network error fetching article from '{url}': {str(re_err)}"
        }
    except Exception as e:
        return {
            "success": False,
            "text": "",
            "title": "",
            "source_type": "url",
            "source_name": parsed.netloc,
            "word_count": 0,
            "error": f"An unexpected error occurred while parsing URL '{url}': {str(e)}"
        }


# =============================================================================
# 6. UNIFIED ARTICLE INGESTION ROUTER
# =============================================================================

def ingest_content(
    direct_text: str = "",
    file_path: Optional[str] = None,
    url: str = ""
) -> Dict[str, Any]:
    """
    Unified router that extracts clean article text from any supported input method:
    - URL (highest precedence if provided)
    - Uploaded File (.pdf, .docx, .doc, .txt, .md)
    - Direct Text / Paste
    """
    # 1. URL extraction
    if url and url.strip():
        return extract_from_url(url.strip())

    # 2. File extraction
    if file_path and os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return extract_from_pdf(file_path)
        elif ext == ".docx":
            return extract_from_docx(file_path)
        elif ext == ".doc":
            return extract_from_doc(file_path)
        elif ext in (".txt", ".text", ".md"):
            return extract_from_txt(file_path)
        else:
            return {
                "success": False,
                "text": "",
                "title": os.path.basename(file_path),
                "source_type": "unsupported",
                "source_name": os.path.basename(file_path),
                "word_count": 0,
                "error": f"Unsupported file extension '{ext}'. Please upload a PDF, DOCX, DOC, or TXT/MD document."
            }

    # 3. Direct Text
    if direct_text and direct_text.strip():
        return extract_from_direct_text(direct_text)

    # 4. No input provided
    return {
        "success": False,
        "text": "",
        "title": "",
        "source_type": "none",
        "source_name": "None",
        "word_count": 0,
        "error": "No input provided. Please paste an article, upload a document (PDF, DOCX, TXT), or provide an article URL."
    }
