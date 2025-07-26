# 📚 Bilingual Multimodal RAG System

This project implements a Multimodal **Retrieval-Augmented Generation (RAG)** pipeline capable of understanding and responding to user queries in both **English** and **Bangla**, with support for **Long-Term and Short-Term Memory**. The system retrieves context from a structured PDF corpus and generates grounded responses.

---

### Working Mechanism

1. **PDF Parsing:** Extract texts, tables and images from the PDF using a `title-based segmentation strategy` with the `Unstructured` library.

2. **Chunking:** Apply `token-aware recursive character chunking` on the extracted texts using `tiktoken` and `LangChain` to maintain semantic boundaries.

3. **Summarization:** Generate summaries for both text chunks and images using the `llama-4-maverick-17b-128e-instruct` model hosted on `Groq Cloud`.

4. **Storage and Embedding:**  
   - Embed the text and image summaries using `jina-embeddings-v3` from `Jina AI`. Store the summaries in `ChromaDB` (vector store).  
   - Store the original texts in `InMemoryDocstore`, linking them to their respective summaries via IDs.

1. **Short-Term Memory:** Maintain conversational `short-term memory` with a sliding window of the last `5 dialogue turns`.

2. **Retrieval via MultiVectorRetriever:** Retrieve relevant documents using `cosine similarity` search. Two retrieval options are supported:
   - Summaries (text + image) with linked `raw text` from `InMemoryDocstore`
   - Summaries (text + image) `only`

3. **Context Construction:** Use the `create_chain` function to build a context by retrieving relevant documents based on the user's query and chosen retrieval strategy.

4. **Prompt and Generation:**  
   - Construct the prompt using `build_prompt`, combining retrieved context, short-term memory and long-term memory.  
   - Generate the final answer using an LLM.

5. **Evaluation:** Evaluate the RAG system’s performance using the `RAGAS` library. Following metrics are used -
    - **Similarity Measurement:** SemanticSimilarity
    - **Groundness Measurement:** ResponseGroundness, Faithfulness, FactualCorrectness
    - **Relevancy Measurement:** LLMContextRecall, ContextRelevance

---

### Tech Stack 

- **Core Programming Language:** Python 
- **LLM Framework:** LangChain
- **Vector Database:** ChromaDB
- **Data Extraction:** Unstructured
- **Chat Model:** llama-4-maverick-17b-128e-instruct / llama-4-scout-17b-16e-instruct (Groq Cloud)
- **Embedding Model:** jina-embeddings-v3 (Jina AI)
- **RAG Evaluation:** RAGAS

---

### Project Structure

```
qna-mm-rag/
├── data/
│   └── book.pdf                # KnowledgeBase for Long-Term memory
│
├── env/                        # Virtual environment 
│
├── notebooks/
│   └── qna-test.ipynb          # Contains the code with intermediate steps and test cases
│
├── .env                        # Environment variables (e.g., API keys)
├── .gitignore                  # Specifies files and folders to ignore in Git
├── README.md                   # Project documentation
├── rag-pipeline-final.ipynb    # Main notebook to run the rag pipeline
└── requirements.txt            # Dependencies
```

---

### Setup Guide

- Python 3.13
- Git installed on your system

1. **Clone the repository**

   ```bash
   git clone https://github.com/arpon-kapuria/QnA-Multimodal-RAG.git
   cd QnA-Multimodal-RAG
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv env
   source env/bin/activate 
   ```

3. **Install project dependencies**

   ```bash
   pip install -r requirements.txt
   ```
   

4. **Set up environment variables**

   Create a `.env` file in the root directory and add necessary environment variables:

   ```
   GROQ_API_KEY="groq_api_key"
   JINA_API_KEY="jina_api_token"
   ```

5. **Run the cells in notebook**

   ```bash
   rag-pipeline-final.ipynb
   ```

6. **To check intermediate steps**

    ```bash
    qna-test.ipynb
    ```

---

### Q&A: Design Choices in the RAG Pipeline

**1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**

**Answer:** `unstructured` was used because it provides multilingual support, handles complex document layouts, and is easy to use. However, image extraction was initially a bit painful and required custom adjustments.

**2. What chunking strategy did you choose (e.g., paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?**

**Answer:** Initially, text was extracted using the `unstructured` library, where a `title-based chunking strategy` was followed. This allowed grouping of content semantically under each header, especially useful in documents with clear section titles (e.g., question sets or academic paragraphs).

Later, LangChain’s `RecursiveCharacterTextSplitter` with a `token-aware chunking strategy`, using `tiktoken` was performed. It aligned with the embedding model's maximum token support (8196 tokens). Chunks were created around paragraph and sentence boundaries using both English and Bangla delimiters. This maintains semantic integrity and aligns with the LLM’s context window.

**3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**

**Answer:** `jina-embeddings-v3`, from Jina AI was used, which supports both English and Bangla. It was chosen for its strong performance on sentence-level similarity and compatibility with multilingual retrieval tasks, while being computationally efficient and free.

**4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**

**Answer:** The query is compared with the stored chunks using `cosine similarity` within `ChromaDB`. Cosine similarity measures the angle between two vectors, making it ideal for determining semantic closeness irrespective of vector magnitude.

`ChromaDB` is chosen for its fast and lightweight approximate nearest neighbor search. It's open-source.

**5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**

**Answer:** To ensure that the question and the document chunks are compared meaningfully -
- The `same embedding model and tokenizer` are used for both queries and chunks.
- Recursive chunking preserves semantic structure.
- Retrieved chunks are embedded in the prompt using a clean prompt template.

If a query is vague, retrieval will `degrade`. We try to mitigate this using short-term memory (chat history) and prompt refinement.

**6. Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?**

**Answer:** Yes, results are generally relevant. To improve further:
- Improve text preprocessing and adjust chunking granularity.
- Replace open-source models with more powerful commercial or cross-lingual embedding/chat models (e.g., OpenAI, Anthropic) for better summary generation and semantic understanding.
- Integrate reranking models (e.g., Cohere Reranker) to refine top-k retrieved results.



