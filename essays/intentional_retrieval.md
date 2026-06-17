# Intentional Retrieval

When you work with large language models, retrieval is the hard part. The models themselves are flexible, but the information you actually need has to fit inside a context window that was never designed to hold a whole product catalog or an entire internal wiki.

Context windows are small by computer standards, and there's a real reason for that, because self-attention scales quadratically, so doubling the window more than doubles the compute bill. Models can juggle tens of thousands of tokens, not millions, and once a fact falls out of the prompt, it might as well not exist at all.

So to cope, we bolt memory onto the outside. People build paging systems like MemGPT, they extract key facts the way Mem0 does, and they layer agents with buffers, scratchpads, and tool loops. The model itself stays frozen while all the scaffolding around it works to keep the right facts in view.

And all of this overhead exists for one reason, which is that an LLM is a probabilistic compute unit rather than a fixed algorithm. It samples from distributions it learned during training. We schedule it, we feed it inputs, and we guard its outputs, much the way operating systems once treated CPUs.

Retrieval-augmented generation was the first pattern that really shipped. The API models were closed to fine-tuning, the open models were costly to retrain, and RAG let teams ground their answers in their own data without ever touching the weights. You chunk the documents, embed them, index them, retrieve, and stuff the results back into the prompt, and it was good enough to launch on.

Vector search became the default because it does fuzzy matching so well. You can phrase the question one way and it'll still find passages written another way, which makes it great for discovery and for the kind of support queries where "close" is usually close enough.

Precision, though, is a different story. Vectors can't tell you whether a passage contains the exact SKU, the exact legal clause, or the exact API parameter you're after, because similarity isn't accuracy, and the model will happily fill any gap with something that merely sounds right.

And chunking only makes this worse. Slice too narrowly and you split a fact across the boundary and miss the answer entirely. Slice too wide and you drag along unrelated text that muddies the retrieval score. A change in chunk size alone can swing recall noticeably, and the model still hallucinates whenever the prompt looks plausible but is quietly missing the fact you needed.

This is why structure beats vectors whenever you already have it. Sales records belong in SQL, where the database already knows how to do the joins and the aggregations. Security policies fit ontologies that enforce the relationships. Engineering wikis often do better in tree-shaped layouts that keep related context together without embedding every paragraph. Use the structure first, and fall back to embeddings only when you have to.

Choosing your retrieval method is really a design problem. Start with the shape of the data, whether it's structured, semi-structured, or unstructured. Ask how precise the answer has to be, since a legal workflow cares far more than triaging a support ticket. Mix the methods in layers, SQL first and then embeddings, or the other way around. Instrument the pipeline so you actually know your hit rates, your chunk overlap, and when a human overrides the output. And plan for memory the way you'd plan for caches, with short-term context, longer-term stores, and refresh cycles that keep everything current.

Intentional retrieval, in the end, just means matching the method to the problem instead of defaulting to whatever's trending. Use vectors when you need fuzziness, use structure when you have certainty, and accept that the orchestration matters just as much as the model you chose.
