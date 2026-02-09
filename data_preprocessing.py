import os
import json
import re
import hashlib
from typing import List, Dict, Optional
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class DataPreprocessor:
    """
    Handles PDF parsing, text cleaning, and corpus generation for RAG system.
    """

    def __init__(self, input_dir: str, output_path: str):
        """
        Initialize the data preprocessor.

        Args:
            input_dir: Directory containing PDF files
            output_path: Path to save the processed corpus (JSONL format)
        """
        self.input_dir = input_dir
        self.output_path = output_path
        self.corpus = []

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing boilerplate content.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        text1 = re.sub(
            r"# In case products are delivered by Henkel Colombiana[\s\S]*$",
            "",
            text,
            flags=re.IGNORECASE
        )
        text2 = re.sub(
            r"for the most direct access to local sales and technical support[\s\S]*?$",
            "",
            text1,
            flags=re.IGNORECASE
        )
        return text2.strip()

    @staticmethod
    def is_cutoff_chunk(text: str) -> bool:
        """
        Check if chunk should be excluded (contains cutoff keywords).

        Args:
            text: Text to check

        Returns:
            True if chunk should be excluded
        """
        return "conversions" in text.lower()

    @staticmethod
    def extract_product_id(filename: str) -> str:
        """
        Extract product ID from filename.
        Example: LOCTITE-AA-332-en_GL.pdf -> LOCTITE-AA-332

        Args:
            filename: PDF filename

        Returns:
            Extracted product ID
        """
        base = os.path.splitext(filename)[0]
        match = re.match(r"([A-Z0-9\-]+?)(?:-[a-z]{2}_[A-Z]{2})?$", base)
        if match:
            return match.group(1)
        return base

    @staticmethod
    def product_hash(product_id: str, length: int = 10) -> str:
        """
        Generate a stable short hash for product_id.
        Example: LOCTITE-454 -> 'ad4f91c2fe'

        Args:
            product_id: Product identifier
            length: Length of hash to return

        Returns:
            Short hash string
        """
        h = hashlib.sha1(product_id.encode("utf-8")).hexdigest()
        return h[:length]

    def chunk_pdf(self, path: str) -> List[str]:
        """
        Extract and chunk text from a PDF file.

        Args:
            path: Path to PDF file

        Returns:
            List of cleaned text chunks
        """
        elements = partition_pdf(
            filename=path,
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=False
        )

        chunks = chunk_by_title(
            elements,
            combine_text_under_n_chars=700,
            max_characters=1000
        )

        cleaned_chunks = []
        for c in chunks:
            if not c.text:
                continue

            text = self.clean_text(c.text)
            if not text:
                continue

            if self.is_cutoff_chunk(text):
                break

            cleaned_chunks.append(text)

        return cleaned_chunks

    def process_pdfs(self) -> List[Dict]:
        """
        Process all PDFs in the input directory and build corpus.

        Returns:
            List of corpus entries
        """
        self.corpus = []

        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        pdf_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".pdf")]

        for filename in tqdm(pdf_files, desc="Processing PDFs"):
            path = os.path.join(self.input_dir, filename)
            product_id = self.extract_product_id(filename)
            product_uuid = self.product_hash(product_id)

            try:
                chunks = self.chunk_pdf(path)
                for idx, chunk in enumerate(chunks):
                    self.corpus.append({
                        "id": f"{product_uuid}::{idx}",
                        "text": chunk,
                        "product_id": product_id,
                        "product_uuid": product_uuid,
                        "source_file": filename,
                        "chunk_index": idx
                    })
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        return self.corpus

    def save_corpus(self) -> None:
        """
        Save the corpus to a JSONL file.
        """
        self.process_pdfs()

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        with open(self.output_path, "w", encoding="utf-8") as f:
            for entry in self.corpus:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Corpus built successfully: {len(self.corpus)} chunks saved to {self.output_path}")



class EmbedProcess:
    """
    Handles embedding and knowledge base in FAISS for RAG system.
    """

    def __init__(
        self,
        metadata: List[Dict] = None,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        auto_generate: bool = True
    ):
        """
        Initialize with metadata and embedding model.

        Args:
            metadata: List of metadata dicts, each containing a 'text' field
            embed_model_name: Name of the sentence transformer model
            auto_generate: Whether to automatically generate embeddings
        """
        self.metadata = metadata
        self.chunks = [entry["text"] for entry in self.metadata] if self.metadata else []
        self.embed_model_name = embed_model_name
        self.embed_model = SentenceTransformer(embed_model_name)
        self.doc_embds = None
        self.index = None

        # Generate embeddings if chunks available and auto_generate is True
        if self.chunks and auto_generate:
            self._generate_embeddings()

    def _generate_embeddings(self) -> None:
        """
        Generate embeddings for all chunks.
        """
        print(f"Generating embeddings for {len(self.chunks)} chunks...")
        self.doc_embds = self.embed_model.encode(
            self.chunks,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        print(f"Embeddings generated with shape: {self.doc_embds.shape}")

    def create_faiss(self) -> None:
        """
        Create FAISS index from embeddings using cosine similarity.
        """
        if self.doc_embds is None:
            raise ValueError("Embeddings not generated. Call _generate_embeddings() first.")

        embed_dim = self.doc_embds.shape[1]
        self.index = faiss.IndexFlatIP(embed_dim) 
        self.index.add(self.doc_embds.astype(np.float32))
        print(f"FAISS index created with {self.index.ntotal} vectors")


    @classmethod
    def from_corpus(cls, corpus_path: str, embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Create EmbedProcess instance from a corpus JSONL file.

        Args:
            corpus_path: Path to corpus.jsonl file
            embed_model_name: Name of the sentence transformer model

        Returns:
            EmbedProcess instance with loaded metadata and chunks
        """
        metadata = []

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                metadata.append(entry)

        print(f"Loaded {len(metadata)} entries from {corpus_path}")
        return cls(metadata=metadata, embed_model_name=embed_model_name)

    def save_embeddings(self, output_dir: str, save_index: bool = True):
        """
        Save embeddings, metadata, and FAISS index to files.

        Args:
            output_dir: Directory to save all files
            save_index: Whether to also save FAISS index (default: True)

        Creates three files:
            - embeddings_{model}.npy
            - embeddings_{model}_meta.jsonl
            - embeddings_{model}.index (if save_index=True)
        """
        if self.doc_embds is None:
            raise ValueError("No embeddings to save. Generate embeddings first.")

        os.makedirs(output_dir, exist_ok=True)

        safe_name = self.embed_model_name.replace("/", "_").replace(".", "-")

        embeddings_path = os.path.join(output_dir, f"embeddings_{safe_name}.npy")
        metadata_path = os.path.join(output_dir, f"embeddings_{safe_name}_meta.jsonl")
        index_path = os.path.join(output_dir, f"embeddings_{safe_name}.index")

        np.save(embeddings_path, self.doc_embds)

        if self.metadata:
            with open(metadata_path, "w", encoding="utf-8") as f:
                for entry in self.metadata:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"Saved metadata to {metadata_path}")

        if save_index:
            if self.index is None:
                self.create_faiss()
            faiss.write_index(self.index, index_path)
            print(f"Saved FAISS index to {index_path}")

    def load_embeddings(self, embeddings_path: str, metadata_path: Optional[str] = None):
        """
        Load pre-computed embeddings from disk.

        Args:
            embeddings_path: Path to .npy embeddings file
            metadata_path: Optional path to metadata JSONL file
        """
        self.doc_embds = np.load(embeddings_path)
        print(f"Loaded embeddings from {embeddings_path} with shape: {self.doc_embds.shape}")

        if metadata_path:
            self.metadata = []
            with open(metadata_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.metadata.append(json.loads(line))
            # Extract chunks from loaded metadata
            self.chunks = [entry["text"] for entry in self.metadata]
            print(f"Loaded {len(self.metadata)} metadata entries from {metadata_path}")



if __name__ == "__main__":
    # Step 1: Process PDFs and create corpus
    preprocessor = DataPreprocessor(
        input_dir="./Data",
        output_path="./corpus.jsonl"
    )
    preprocessor.save_corpus()

    # Step 2: Load corpus and create embeddings
    data = EmbedProcess.from_corpus(
        corpus_path="./corpus.jsonl",
        embed_model_name="intfloat/e5-base-v2"
    )

    # Step 3: Save embeddings, metadata, and FAISS index
    data.save_embeddings(output_dir="./embeddings")


