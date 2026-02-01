import os
import logs
import hashlib
import traceback
import pymupdf as fitz
import regex as re
from typing import List, Dict, Tuple, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from configuration import Configuration
from rag_scripts.interfaces import IDocumentChunker


class PyMuPDFChunker(IDocumentChunker):
    
    def __init__(self, pdf_path: str, chunk_size: int = Configuration.DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = Configuration.DEFAULT_CHUNK_OVERLAP):
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function =len,
            separators=["\n\n","\n","(?<=\. )\n",
                        "(?<=[a-z0-9]\.)","(?<=\? )", "(?<=\! )",
                        " ","" ] )
        
        logs.logger.info(f"Initialized PyMuPDFChunker for {os.path.basename(pdf_path)} with chunk_size = {chunk_size} and chunk overlap = {chunk_overlap}")

    def _clean_text(self, text: str) -> str:
        try:
            text = text.replace('\u25cf','').replace('\u2022','')
            text = text.replace('\u201c','').replace('\u201d','')
            text = text.replace('\u2013','-').replace('\u2014','-').replace('\u2015','-')
            text = re.sub(r'\n\s*\n',' ',text)
            text = re.sub(r' {2,}', ' ',text)
            text = text.replace('\\nb','')
            text = '\n'.join([line.strip() for line in text.split('\n')])
            text = re.sub(r'[^\x20-\x7E\t\n\r]', '', text)
            return text.strip()
        except Exception as ex:
            logs.logger.info(f"Failed to clean text: {ex}")
            traceback.print_exc()
            logs.logger.info(traceback.print_exc())

            return text.strip()


    def hash_document(self):
        try:
            hasher = hashlib.sha256()
            with open(self.pdf_path, 'rb') as fl:
                while chunk:=fl.read(8192):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as ex:
            logs.logger.info(f"Exception hashing PDF {self.pdf_path}: {ex}")
            logs.logger.info(traceback.print_exc())
            raise ValueError(f"Failed to hash PDF: {ex}")
        
    def chunk_documents(self) -> Tuple[str, List[Dict[str, Any]]]:
        try:
            doc_hash = self.hash_document()
            doc_chunks =[]

            with fitz.open(self.pdf_path) as document:
                for idx, page in enumerate(document):
                    page_text = page.get_text()
                    if page_text.strip():
                        section = self._extract_section(page_text)
                        clause = self._extract_clause(page_text)
                        page_text = self._clean_text(page_text)
                        chunked_page = self.text_splitter.split_text(page_text)
                        for chunk_idx, chunk_content in enumerate(chunked_page):
                            if not chunk_content.strip():
                                continue
                            doc_chunks.append({
                                "content":chunk_content.strip(),
                                "metadata":{
                                    "document_id":doc_hash,
                                    "source_file":os.path.basename(self.pdf_path),
                                    "page_number": idx+1,
                                    "chunk_id":f"{doc_hash} - {idx + 1} -{chunk_idx}",
                                    "section": section or "Unknown Section",
                                    "clause": clause or "Unknown Clause",
                                    "chunk_index_on_page": chunk_idx
                                }
                            })
            if not doc_chunks:
                logs.logger.info(f"No text or chunks extracted from {self.pdf_path} after cleaning")
                return doc_hash,[]
            else:
                logs.logger.info(f"success Document chunked {self.pdf_path} into {len(doc_hash)} chunks")
                return doc_hash,doc_chunks
        
                        

        except Exception as ex:
            logs.logger.info(f"Exception in document chunking {ex}")
            traceback.print_exc()
            return self.hash_document(), []
        

    def _extract_section(self, text: str) -> str:
        match_major = re.search(r'^(?:[IVX]+\.?\s+|[A-Z]\.?\s+|[0-9]+\.?\s+)(.+)', text, re.MULTILINE)
        if match_major:
            return match_major.group(0)

        match_firstline = re.search(r'^\s*([A-Za-z0-9][\w\s,&\'-]+?)\s*$', text, re.MULTILINE)
        if match_firstline:
            return match_firstline.group(1).strip()

        return None


    def _extract_clause(self,text) -> str:
        match = re.search(r'^(?:(?:•|●|-|\*|\d+\.|\([a-z]\)|\([A-Z]\)|\w\))\s*)(.+?)(?=\n(?:•|●|-|\*|\d+\.|\([a-z]\)|\([A-Z]\)|\w\))|\n\n|\Z)', text, re.MULTILINE | re.DOTALL)
        if match:
            clause = match.group(1).strip()
            if len(clause.split()) < 10:
                next_match = re.search(
                    r'^(?:(?:•|●|-|\*|\d+\.|\([a-z]\)|\([A-Z]\)|\w\))\s*)(.+?)(?=\n(?:•|●|-|\*|\d+\.|\([a-z]\)|\([A-Z]\)|\w\))|\n\n|\Z)',
                    text[match.end():], re.MULTILINE | re.DOTALL)
                if next_match:
                    clause += " " + next_match.group(0).strip()
            return clause

        match_para = re.search(r'^(?!#|\s*$).*?\n\n',text,re.DOTALL)
        if match_para:
            return match_para.group(0).strip()
        return None
