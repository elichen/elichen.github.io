# Intentional Retrieval

Transformers have fixed context windows where attention scales quadratically and information outside disappears. These constraints shape how we build with LLMs, yet the default answer—vector databases—isn't always right. Different problems need different retrieval strategies.

## Context Limits

Self-attention costs *O(n²)*, so doubling context quadruples compute. Models handle tens of thousands of tokens, not millions, with no persistent memory—once it's out of context, it's gone.

The response has been to bolt on virtual memory. MemGPT pages data like swap space, Mem0 extracts facts for later, and agent frameworks add buffers, stores, and tool loops. The model stays fixed while the system supplies memory.

## New Primitive

LLMs are probabilistic compute units with no fixed algorithm, just distributions from training. This is why research focuses on orchestration rather than weights.

Teams explore alternatives like state-space models, attention variants, and MoE routing, while memory hierarchies decide what to keep, compress, or forget. LLMs have become the new CPU—scheduled, fed inputs, and wrapped in guardrails.

## Why RAG Won

API models couldn't be fine-tuned and open models were expensive to retrain. RAG offered a solution: keep the model frozen, retrieve passages, and stuff them in the prompt for domain grounding without touching weights. It shipped fast.

The pipeline chunks documents, embeds them, indexes, queries, retrieves top-k, re-ranks, and assembles prompts. While fine-tuning degraded reasoning, RAG sidestepped the problem with stable models, fresh knowledge, and runtime patches.

## Vector Search Reality

Vector search excels at fuzzy matching—ask one way and find documents written another. It's great for discovery, exploration, and support queries.

But it fails at precision. Vectors don't know if a passage contains the exact SKU, legal clause, or API parameter you need. Similarity isn't accuracy.

## Chunking Problems

Chunk too small and facts split across boundaries—miss one and the model guesses. Chunk too large and unrelated content hitchhikes along, diluting relevance. Chunk size alone can swing recall by nine percentage points.

When chunks are missing, models hallucinate. "I don't know" prompts help but aren't enough—if context looks plausible, the model fills gaps.

## When Structure Beats Vectors

Sales records belong in SQL, security policies in ontologies, and engineering wikis in hierarchies.

- **SQL** provides exact answers with joins and aggregations, no hallucination
- **Graphs** let you walk edges instead of embeddings, achieving 93% vs 89% precision
- **Trees** preserve document structure and keep related context together

Use structure when you have it and embeddings when you don't.

## Choosing Retrieval

1. **Check your data**: Is it structured, semi-structured, or unstructured?
2. **Check precision needs**: Legal workflows need exactness while support can tolerate fuzziness
3. **Mix methods**: Try SQL first then embeddings, or embeddings then filters
4. **Measure retrieval**: Track hit rates, chunk overlap, and user overrides
5. **Design memory**: Plan short-term caches, long-term stores, and refresh cycles

Match retrieval to task instead of defaulting to vectors.

## Building with LLMs

LLMs are compute primitives with memory problems, and vector databases aren't always the answer.

When you need precision, use the structures you already have. When you need fuzzy matching, use embeddings but chunk carefully.

The question isn't "RAG or not" but rather "which retrieval for which problem." Choose intentionally.
