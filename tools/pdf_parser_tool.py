import os
import pdfplumber
from pypdf import PdfReader
from langchain.tools import tool

@tool
def pdf_parser(file_path: str) -> dict:
    """
    Parses a local PDF file and extracts text from all pages.
    Input must be a valid LOCAL FILE PATH to a PDF.
    Returns a dictionary with 'total_pages', 'text', and 'word_count'.
    """
    if not os.path.exists(file_path):
        return {'error': 'File not found'}
        
    text = ""
    total_pages = 0
    
    try:
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception:
        # Fall back to pypdf
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            return {'error': f'Failed to parse PDF: {str(e)}'}

    word_count = len(text.split())
    
    return {
        'total_pages': total_pages,
        'text': text.strip(),
        'word_count': word_count
    }
