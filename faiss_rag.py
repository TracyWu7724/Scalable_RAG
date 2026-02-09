from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

from google import genai
from dotenv import load_dotenv
load_dotenv()


class BaseRAG:
    """Base class for RAG systems with common functionality."""

    def __init__(self, metadata_path, faiss_path, embed_model_name="sentence-transformers/all-MiniLM-L6-v2", use_reranker=False, reranker_model_name="Alibaba-NLP/gte-reranker-modernbert-base"):
        """
        Initialize base RAG components.

        Args:
            metadata_path: Path to metadata JSONL file
            faiss_path: Path to FAISS index file
            embed_model_name: Sentence transformer model for embeddings
            use_reranker: Whether to use reranker model
            reranker_model_name: Reranker model name
        """
        # Load embedding model
        if embed_model_name == "nomic-ai/nomic-embed-text-v1":
            self.embed_model = SentenceTransformer(embed_model_name, trust_remote_code=True)
        else:
            self.embed_model = SentenceTransformer(embed_model_name)

        # Load metadata and FAISS index
        self.chunks, self.metadata = self._load_metadata(metadata_path)
        print(f"Loaded {len(self.chunks)} chunks from {metadata_path}")

        self.index = faiss.read_index(faiss_path)
        print(f"Loaded FAISS index from {faiss_path} with {self.index.ntotal} vectors")

        # Build per-product indices
        self._build_product_indices()

        # Initialize reranker if requested
        self.use_reranker = use_reranker
        self.rerank_model = None
        self.rerank_tokenizer = None

        if use_reranker:
            print(f"Loading reranker model: {reranker_model_name}")
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
                reranker_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            self.rerank_model.eval()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.rerank_model.to(self.device)
            print(f"Reranker model loaded on {self.device}")

    @staticmethod
    def _load_metadata(metadata_path):
        """Load chunks and metadata from JSONL file."""
        chunks = []
        metadata = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                chunks.append(entry["text"])
                metadata.append(entry)
        return chunks, metadata

    def _build_product_indices(self):
        """Build separate FAISS indices for each product."""
        # Group chunks and metadata by product_uuid
        product_chunks = {}
        product_metadata = {}
        product_indices_map = {}  # Maps product_uuid to list of chunk indices

        for idx, meta in enumerate(self.metadata):
            product_uuid = meta.get("product_uuid")
            if not product_uuid:
                continue

            if product_uuid not in product_chunks:
                product_chunks[product_uuid] = []
                product_metadata[product_uuid] = []
                product_indices_map[product_uuid] = []

            product_chunks[product_uuid].append(self.chunks[idx])
            product_metadata[product_uuid].append(meta)
            product_indices_map[product_uuid].append(idx)

        # Build FAISS index for each product
        self.product_chunk_faiss = {}
        self.product_chunk_refs = {}
        self.product_ids = {}  # Map product_uuid to product_id

        for product_uuid, chunk_list in product_chunks.items():
            # Get embeddings for this product's chunks from the global index
            indices = product_indices_map[product_uuid]
            embeddings = np.array([self.index.reconstruct(int(i)) for i in indices])

            # Create FAISS index for this product
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype('float32'))

            self.product_chunk_faiss[product_uuid] = index
            self.product_chunk_refs[product_uuid] = chunk_list

            # Store product_id for reference
            if product_metadata[product_uuid]:
                self.product_ids[product_uuid] = product_metadata[product_uuid][0].get("product_id", product_uuid)

        print(f"Built {len(self.product_chunk_faiss)} per-product FAISS indices")

    def retrieve_top_k_faiss(self, query, k=3):
        """Retrieve top-k most similar chunks using FAISS."""
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        _, indices = self.index.search(q_emb, k)
        top_docs = [self.chunks[i] for i in indices[0]]
        return top_docs

    def rerank(self, query, candidate_texts, top_k=3):
        """
        Rerank candidate texts using cross-encoder model.

        Args:
            query: Query string
            candidate_texts: List of candidate text strings
            top_k: Number of top results to return

        Returns:
            List of tuples (text, score) sorted by relevance score
        """
        if not self.use_reranker or self.rerank_model is None:
            # If reranker not available, return texts with dummy scores
            return [(text, 0.0) for text in candidate_texts[:top_k]]

        # Create pairs of [query, text] for the reranker
        pairs = [[query, txt] for txt in candidate_texts]

        # Tokenize pairs
        encoded = self.rerank_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        # Get relevance scores
        with torch.no_grad():
            scores = self.rerank_model(**encoded).logits.view(-1).float()

        scores = scores.cpu().numpy()

        # Sort by score (highest first)
        ranked = sorted(
            zip(candidate_texts, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]

    def retrieve_and_rerank(self, query, initial_k=10, final_k=3):
        """
        Retrieve more candidates with FAISS, then rerank to get top results.

        Args:
            query: Query string
            initial_k: Number of candidates to retrieve initially
            final_k: Number of final results after reranking

        Returns:
            List of top final_k text chunks after reranking
        """
        # Step 1: Retrieve more candidates using FAISS
        candidates = self.retrieve_top_k_faiss(query, k=initial_k)

        # Step 2: Rerank candidates
        if self.use_reranker:
            ranked_results = self.rerank(query, candidates, top_k=final_k)
            # Extract just the texts (without scores)
            top_docs = [text for text, score in ranked_results]
        else:
            # No reranking, just return top final_k
            top_docs = candidates[:final_k]

        return top_docs

    def search_within_product(self, product_uuid, query, k=5):
        """
        Search within a specific product's FAISS index.

        Args:
            product_uuid: UUID of the product to search within
            query: Query string
            k: Number of results to return

        Returns:
            List of top k text chunks from this product
        """
        if product_uuid not in self.product_chunk_faiss:
            print(f"Warning: Product {product_uuid} not found")
            return []

        # Encode query
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)

        # Search within product's index
        index = self.product_chunk_faiss[product_uuid]
        D, I = index.search(q_emb, k)

        # Get chunk references
        refs = self.product_chunk_refs[product_uuid]
        results = [refs[idx] for idx in I[0] if idx < len(refs)]

        return results

    def identify_relevant_products(self, query, top_n=3):
        """
        Identify the most relevant products for a query.

        Args:
            query: Query string
            top_n: Number of top products to return

        Returns:
            List of (product_uuid, product_id, score) tuples
        """
        # Use global FAISS to find relevant chunks
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        D, I = self.index.search(q_emb, min(20, len(self.chunks)))  # Get top 20 chunks

        # Count product occurrences and aggregate scores
        product_scores = {}
        for dist, idx in zip(D[0], I[0]):
            if idx >= len(self.metadata):
                continue
            product_uuid = self.metadata[idx].get("product_uuid")
            if product_uuid:
                if product_uuid not in product_scores:
                    product_scores[product_uuid] = []
                product_scores[product_uuid].append(float(dist))

        # Aggregate scores (use max score for each product)
        product_rankings = []
        for product_uuid, scores in product_scores.items():
            max_score = max(scores)  
            product_id = self.product_ids.get(product_uuid, product_uuid)
            product_rankings.append((product_uuid, product_id, max_score))

        # Sort by score (highest first)
        product_rankings.sort(key=lambda x: x[2], reverse=True)

        return product_rankings[:top_n]

    def retrieve_and_rerank_by_product(self, product_uuid, query, initial_k=10, final_k=3):
        """
        Search within a specific product and rerank results.

        Args:
            product_uuid: UUID of the product to search within
            query: Query string
            initial_k: Number of candidates to retrieve initially
            final_k: Number of final results after reranking

        Returns:
            List of top final_k text chunks after reranking
        """
        # Search within product
        candidates = self.search_within_product(product_uuid, query, k=initial_k)

        if not candidates:
            return []

        # Rerank if enabled
        if self.use_reranker and len(candidates) > 0:
            ranked_results = self.rerank(query, candidates, top_k=final_k)
            top_docs = [text for text, score in ranked_results]
        else:
            top_docs = candidates[:final_k]

        return top_docs

    def build_prompt(self, context, question):
        """Build prompt with context and LOCTITE product information."""
        prompt = f"""
            Use this info to answer: {context}

            Add this manual information to answer based on the product family characteristics:
            LOCTITE product numbers indicate the type of adhesive:

            2XX = Threadlockers
            Used to lock and secure threaded fasteners.
            Threadlockers can have low-strength, medium-strength, high-strength and locking for pre-assembled fasteners.

            4XX and some 3XX products = Instant Adhesives (Cyanoacrylates)
            Fast-curing glues for rubber, plastic, and metal.

            4XXX = Instant Adhesives for special domain such as medical parts
            Very low-viscosity, fast-fixturing for medical parts or other difficult-to-bond materials.

            6XX = Retaining Compounds
            Used to bond cylindrical parts like bearings and shafts.

            5XX = Sealants
            Used to seal surfaces, flanges, and pipe fittings.

            AA = Structural adhesives
            General-purpose, high shear strength, metal-to-metal bonder.

            When a user mentions a LOCTITE number: Look at the first digit(s).
            Match it to the correct category (2XX, 3XX, 4XX, 4XXX, 5XX, 6XX, AA).

            Question: {question}
            Answer:"""
        
        return prompt


    def build_prompt_with_history(self, context, question, history):
        """Build prompt with conversation history, context, and LOCTITE product information.

        Args:
            context: Retrieved document context
            question: Current user question
            history: List of dicts with 'role' and 'content' keys
        """
        history_block = ""
        if history:
            lines = []
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                lines.append(f"{role}: {msg['content']}")
            history_block = "Previous conversation:\n" + "\n".join(lines) + "\n\n"

        prompt = f"""{history_block}Use this info to answer: {context}

            Add this manual information to answer based on the product family characteristics:
            LOCTITE product numbers indicate the type of adhesive:

            2XX = Threadlockers
            Used to lock and secure threaded fasteners.
            Threadlockers can have low-strength, medium-strength, high-strength and locking for pre-assembled fasteners.

            4XX and some 3XX products = Instant Adhesives (Cyanoacrylates)
            Fast-curing glues for rubber, plastic, and metal.

            4XXX = Instant Adhesives for special domain such as medical parts
            Very low-viscosity, fast-fixturing for medical parts or other difficult-to-bond materials.

            6XX = Retaining Compounds
            Used to bond cylindrical parts like bearings and shafts.

            5XX = Sealants
            Used to seal surfaces, flanges, and pipe fittings.

            AA = Structural adhesives
            General-purpose, high shear strength, metal-to-metal bonder.

            When a user mentions a LOCTITE number: Look at the first digit(s).
            Match it to the correct category (2XX, 3XX, 4XX, 4XXX, 5XX, 6XX, AA).

            Question: {question}
            Answer:"""

        return prompt

    def faiss_chain_product_based_with_history(self, question, history=None, initial_k=10, final_k=3):
        """Product-based RAG pipeline with conversation history.

        FAISS retrieval runs on the current question only. History is injected
        into the prompt so the LLM has conversational context.

        Args:
            question: Current user question
            history: List of dicts with 'role' and 'content' keys
            initial_k: Number of candidates to retrieve from product
            final_k: Number of documents to use after reranking

        Returns:
            Generated answer from LLM
        """
        if history is None:
            history = []

        # Identify product from current question only
        relevant_products = self.identify_relevant_products(question, top_n=1)
        if not relevant_products:
            print("Warning: No relevant products found, falling back to global search")
            top_docs = self.retrieve_and_rerank(question, initial_k, final_k)
        else:
            product_uuid, product_id, score = relevant_products[0]
            print(f"Auto-identified product: {product_id} (UUID: {product_uuid}, score: {score:.4f})")
            top_docs = self.retrieve_and_rerank_by_product(product_uuid, question, initial_k, final_k)
            if not top_docs:
                print("Warning: No documents found in product, falling back to global search")
                top_docs = self.retrieve_and_rerank(question, initial_k, final_k)

        context = "\n\n".join(top_docs)
        prompt = self.build_prompt_with_history(context, question, history)
        answer = self.ask_llm(prompt)
        return answer

    def ask_llm(self, prompt):
        """Send prompt to LLM and get response. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement ask_llm method")

    def faiss_chain(self, question, k=3):
        """Full RAG pipeline: retrieve context and generate answer."""
        top_docs = self.retrieve_top_k_faiss(question, k)
        context = "\n\n".join(top_docs)
        prompt = self.build_prompt(context, question)
        answer = self.ask_llm(prompt)
        return answer

    def faiss_chain_with_rerank(self, question, initial_k=10, final_k=3):
        """
        Full RAG pipeline with reranking: retrieve more candidates, rerank, then generate answer.

        Args:
            question: User's question
            initial_k: Number of candidates to retrieve initially (default: 10)
            final_k: Number of documents to use after reranking (default: 3)

        Returns:
            Generated answer from LLM
        """
        top_docs = self.retrieve_and_rerank(question, initial_k=initial_k, final_k=final_k)
        context = "\n\n".join(top_docs)
        prompt = self.build_prompt(context, question)
        answer = self.ask_llm(prompt)
        return answer

    def faiss_chain_product_based(self, question, initial_k=10, final_k=3):
        """
        Product-based RAG pipeline: identify product, search within product, rerank, then generate answer.

        Args:
            question: User's question
            initial_k: Number of candidates to retrieve from product (default: 10)
            final_k: Number of documents to use after reranking (default: 3)

        Returns:
            Generated answer from LLM
        """
        # Identify product if not provided
        relevant_products = self.identify_relevant_products(question, top_n=1)
        if not relevant_products:
            print("Warning: No relevant products found, falling back to global search")
            return self.faiss_chain_with_rerank(question, initial_k, final_k)
        product_uuid, product_id, score = relevant_products[0]
        print(f"Auto-identified product: {product_id} (UUID: {product_uuid}, score: {score:.4f})")


        # Search and rerank within product
        top_docs = self.retrieve_and_rerank_by_product(product_uuid, question, initial_k, final_k)

        if not top_docs:
            print("Warning: No documents found in product, falling back to global search")
            return self.faiss_chain_with_rerank(question, initial_k, final_k)

        # Build prompt and generate answer
        context = "\n\n".join(top_docs)
        prompt = self.build_prompt(context, question)
        answer = self.ask_llm(prompt)
        return answer


class myRAG(BaseRAG):
    """RAG system using local Hugging Face models."""

    def __init__(
        self,
        metadata_path,
        faiss_path,
        llm_model_name="google/gemma-2b-it",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_reranker=False,
        reranker_model_name="Alibaba-NLP/gte-reranker-modernbert-base"
    ):

        # Initialize base class
        super().__init__(metadata_path, faiss_path, embed_model_name, use_reranker, reranker_model_name)

        # Initialize local LLM
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.llm = pipeline(task="text-generation", model=self.llm_model, tokenizer=self.llm_tokenizer, max_new_tokens=256)
        print(f"Initialized local LLM: {llm_model_name}")

    def ask_llm(self, prompt):
        """Send prompt to local LLM and get response."""
        output = self.llm(prompt)
        text = output[0]["generated_text"]
        if "Answer:" in text:
            return text.split("Answer:", 1)[-1].strip()
        return text.strip()


class myRAG_API(BaseRAG):
    """RAG system using Google Gemini API."""

    def __init__(
        self,
        metadata_path,
        faiss_path,
        gemini_model_name="gemini-2.5-flash",
        embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_reranker=False,
        reranker_model_name="Alibaba-NLP/gte-reranker-modernbert-base"
    ):

        # Initialize base class
        super().__init__(metadata_path, faiss_path, embed_model_name, use_reranker, reranker_model_name)

        # Configure Google GenAI client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key must be provided")

        self.client = genai.Client(api_key=api_key)
        self.model_name = gemini_model_name
        print(f"Initialized Google GenAI model: {gemini_model_name}")

    def ask_llm(self, prompt):
        """Send prompt to Google GenAI API and get response."""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            return f"Error calling Google GenAI API: {str(e)}"


if __name__ == "__main__":
    # Example usage 1: use Gemini API with product-based reranking
    MODEL_NAME = "BAAI_bge-base-en-v1-5"

    meta_path = f"./embeddings/embeddings_{MODEL_NAME}_meta_2.jsonl"
    index_path = f"./embeddings/embeddings_{MODEL_NAME}_2.index"

    query = "What is the LOCTITE 243? Can you tell me more about it?"

    rag_api_product = myRAG_API(
        metadata_path=meta_path,
        faiss_path=index_path,
        gemini_model_name="gemini-2.5-flash",
        embed_model_name="BAAI/bge-base-en-v1.5",
        use_reranker=True,
        reranker_model_name="Alibaba-NLP/gte-reranker-modernbert-base"
    )

    result_api_product = rag_api_product.faiss_chain_product_based(question=query, initial_k=10, final_k=3)
    print("Result:", result_api_product)

