# kvMind

The system enables efficient in-context learning by:
Pre-computing KV caches for text chunks
Using attention mechanisms to identify relevant context
Leveraging cached representations for faster generation

## **kv_retrieve.py**
- Implements batch processing of text chunks to generate KV caches
- Provides various scoring methods (average, top tokens, max) to evaluate relevance
- Features efficient batch retrieval capabilities that calculate attention scores across multiple chunks simultaneously
## **ag_kv.py**
- Implements a RAG (Retrieval-Augmented Generation) pipeline using KV caches
- Processes and indexes text using sentence-based chunking
- Retrieves relevant context chunks based on attention scores
- Generates answers using the retrieved context's KV caches
## **mortise.py**：
Support functions, including saving and reading KV cache (although cache files are large, it's a space-time tradeoff).

## **client 目录**：
- A (demonstration only) frontend demo showcasing text interactions for companion-type products. It emphasizes real-time message sending rather than real-time responses, allowing users to express themselves with less thinking and pressure.
- The backend can calculate KV cache in real-time, but only generates responses when conditions are met.
