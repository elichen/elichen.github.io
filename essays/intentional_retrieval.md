# Intentional Retrieval

When you work with large language models, retrieval is the hard part. The models are flexible, but the information you need has to fit inside a context window that was never meant to hold a whole product catalog or an internal wiki.

Context windows are small by computer standards. Self-attention scales quadratically, so doubling the window more than doubles the compute bill. Models can juggle tens of thousands of tokens, not millions, and once a fact falls out of the prompt it might as well not exist.

To cope, we bolt on memory. People build paging systems like MemGPT, extract key facts the way Mem0 does, and layer agents with buffers, scratchpads, and tool loops. The model stays frozen while the scaffolding around it works to keep the right facts in view.

All this overhead exists because an LLM is a probabilistic compute unit, not a fixed algorithm. It samples from distributions learned during training. We schedule it, feed it inputs, and guard its outputs the way operating systems once treated CPUs.

Retrieval-augmented generation was the first pattern that shipped. API models were closed to fine-tuning, open models were costly to retrain, and RAG let teams ground answers in their own data without touching the weights. Chunk documents, embed them, index, retrieve, stuff the results back into the prompt—good enough to launch.

Vector search became the default because it does fuzzy matching well. Phrase the question one way and it can find passages written in another. That makes it great for discovery, exploration, and the kind of support queries where "close" is usually good enough.

Precision is a different story. Vectors cannot tell whether a passage contains the exact SKU, legal clause, or API parameter you need. Similarity is not accuracy, and the model will happily fill the gaps with something that sounds right.

Chunking makes this worse. If you slice narrowly you split facts across boundaries and miss the answer. If you slice wide you drag along unrelated text that muddies the retrieval score. A tweak in chunk size alone can swing recall by large margins, and the model still hallucinates when the prompt looks plausible but lacks the missing fact.

Structure beats vectors whenever you already have it. Sales records belong in SQL where joins and aggregations come for free. Security policies live comfortably in ontologies that enforce relationships. Engineering wikis often benefit from tree-shaped layouts that keep related context together without embedding every paragraph. Use the structure first, then fall back to embeddings.

Choosing retrieval is a design problem. Start with the shape of the data: structured, semi-structured, or unstructured. Ask how precise the answer must be; legal workflows care more than triaging support tickets. Mix methods in layers—SQL first then embeddings, or the other way around. Instrument the pipeline so you know hit rates, chunk overlap, and when humans override the output. Plan for memory the way you would for caches: short-term context, longer-term stores, and refresh cycles that keep things current.

Intentional retrieval means matching the method to the problem instead of defaulting to the trend. Use vectors when you need fuzziness, structure when you have certainty, and accept that the orchestration matters as much as the model you chose.
