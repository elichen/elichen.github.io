const sources = [
    { id: 1, title: "Attention Is All You Need", detail: "Vaswani et al., Transformer architecture, 2017.", url: "https://arxiv.org/abs/1706.03762" },
    { id: 2, title: "Improving Language Understanding by Generative Pre-Training", detail: "Radford et al., GPT-1, OpenAI, 2018.", url: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf" },
    { id: 3, title: "Better Language Models and Their Implications", detail: "OpenAI announcement and staged-release discussion for GPT-2, 2019.", url: "https://openai.com/index/better-language-models" },
    { id: 4, title: "Language Models are Few-Shot Learners", detail: "Brown et al., GPT-3, 2020.", url: "https://arxiv.org/abs/2005.14165" },
    { id: 5, title: "Scaling Laws for Neural Language Models", detail: "Kaplan et al., OpenAI, 2020.", url: "https://openai.com/research/scaling-laws-for-neural-language-models" },
    { id: 6, title: "Training Compute-Optimal Large Language Models", detail: "Hoffmann et al., Chinchilla scaling, 2022.", url: "https://arxiv.org/abs/2203.15556" },
    { id: 7, title: "Aligning Language Models to Follow Instructions", detail: "OpenAI InstructGPT release, 2022.", url: "https://openai.com/index/instruction-following/" },
    { id: 8, title: "Introducing ChatGPT", detail: "OpenAI public chatbot release, November 2022.", url: "https://openai.com/blog/chatgpt/" },
    { id: 9, title: "Evaluating Large Language Models Trained on Code", detail: "OpenAI Codex and HumanEval, 2021.", url: "https://arxiv.org/abs/2107.03374" },
    { id: 10, title: "Competitive Programming with AlphaCode", detail: "Google DeepMind code generation system, 2022.", url: "https://deepmind.google/blog/competitive-programming-with-alphacode" },
    { id: 11, title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", detail: "Wei et al., 2022.", url: "https://arxiv.org/abs/2201.11903" },
    { id: 12, title: "Self-Consistency Improves Chain of Thought Reasoning in Language Models", detail: "Wang et al., 2022.", url: "https://arxiv.org/abs/2203.11171" },
    { id: 13, title: "PaLM: Scaling Language Modeling with Pathways", detail: "Google Research, 540B dense model, 2022.", url: "https://research.google/pubs/palm-scaling-language-modeling-with-pathways/" },
    { id: 14, title: "GPT-4", detail: "OpenAI GPT-4 research release, 2023.", url: "https://openai.com/index/gpt-4-research/" },
    { id: 15, title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", detail: "Lewis et al., 2020.", url: "https://arxiv.org/abs/2005.11401" },
    { id: 16, title: "ReAct: Synergizing Reasoning and Acting in Language Models", detail: "Google Research, 2022.", url: "https://research.google/blog/react-synergizing-reasoning-and-acting-in-language-models/" },
    { id: 17, title: "Toolformer: Language Models Can Teach Themselves to Use Tools", detail: "Schick et al., 2023.", url: "https://arxiv.org/abs/2302.04761" },
    { id: 18, title: "Constitutional AI: Harmlessness from AI Feedback", detail: "Anthropic alignment method, 2022.", url: "https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback" },
    { id: 19, title: "Meta and Microsoft Introduce the Next Generation of Llama", detail: "Meta Llama 2 announcement, 2023.", url: "https://about.fb.com/news/2023/07/llama-2/" },
    { id: 20, title: "Announcing Mistral 7B", detail: "Mistral AI, 2023.", url: "https://mistral.ai/news/announcing-mistral-7b" },
    { id: 21, title: "Mixtral of Experts", detail: "Mistral AI sparse MoE release, 2023.", url: "https://mistral.ai/news/mixtral-of-experts" },
    { id: 22, title: "Everything to Know About Gemini", detail: "Google Gemini multimodal model collection, 2023.", url: "https://blog.google/innovation-and-ai/technology/ai/gemini-collection/" },
    { id: 23, title: "Introducing the Next Generation of Claude", detail: "Anthropic Claude 3 family, 2024.", url: "https://www.anthropic.com/news/claude-3-family" },
    { id: 24, title: "Introducing GPT-4o and More Tools to ChatGPT Free Users", detail: "OpenAI GPT-4o release, 2024.", url: "https://openai.com/index/gpt-4o-and-more-tools-to-chatgpt-free/" },
    { id: 25, title: "Introducing Meta Llama 3", detail: "Meta AI open-weight release, 2024.", url: "https://ai.meta.com/blog/meta-llama-3/" },
    { id: 26, title: "Introducing OpenAI o1", detail: "OpenAI reasoning model release, 2024.", url: "https://openai.com/index/introducing-openai-o1-preview/" },
    { id: 27, title: "DeepSeek-V3 Technical Report", detail: "DeepSeek-AI MoE model, 2024.", url: "https://arxiv.org/abs/2412.19437" },
    { id: 28, title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", detail: "DeepSeek-AI reasoning model, 2025.", url: "https://arxiv.org/abs/2501.12948" },
    { id: 29, title: "Claude 3.7 Sonnet and Claude Code", detail: "Anthropic hybrid reasoning model and coding agent, 2025.", url: "https://www.anthropic.com/news/claude-3-7-sonnet" },
    { id: 30, title: "Introducing OpenAI o3 and o4-mini", detail: "OpenAI tool-using reasoning models, 2025.", url: "https://openai.com/index/introducing-o3-and-o4-mini/" },
    { id: 31, title: "Introducing GPT-4.1 in the API", detail: "OpenAI coding, instruction following, and long-context release, 2025.", url: "https://openai.com/index/gpt-4-1/" },
    { id: 32, title: "Introducing gpt-oss", detail: "OpenAI open-weight reasoning models, 2025.", url: "https://openai.com/index/introducing-gpt-oss" },
    { id: 33, title: "Introducing GPT-5", detail: "OpenAI unified fast and thinking system, 2025.", url: "https://openai.com/index/introducing-gpt-5/" },
    { id: 34, title: "Introducing GPT-5.3-Codex", detail: "OpenAI agentic coding model, 2026.", url: "https://openai.com/index/introducing-gpt-5-3-codex/" },
    { id: 35, title: "Gemini 3.1 Pro", detail: "Google reasoning-focused Gemini update, 2026.", url: "https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/" },
    { id: 36, title: "Introducing GPT-5.4", detail: "OpenAI GPT-5.4 Thinking and Pro, 2026.", url: "https://openai.com/index/introducing-gpt-5-4/" },
    { id: 37, title: "Introducing Claude Opus 4.7", detail: "Anthropic frontier model update, 2026.", url: "https://www.anthropic.com/news/claude-opus-4-7" },
    { id: 38, title: "Introducing GPT-5.5", detail: "OpenAI agentic work model, April 2026.", url: "https://openai.com/index/introducing-gpt-5-5/" },
    { id: 39, title: "Tree of Thoughts", detail: "Yao et al., deliberate search over reasoning traces, 2023.", url: "https://arxiv.org/abs/2305.10601" },
    { id: 40, title: "STaR: Bootstrapping Reasoning With Reasoning", detail: "Self-taught reasoner loop, 2022.", url: "https://arxiv.org/abs/2203.14465" },
    { id: 41, title: "Let's Verify Step by Step", detail: "OpenAI process supervision for mathematical reasoning, 2023.", url: "https://cdn.openai.com/improving-mathematical-reasoning-with-process-supervision/Lets_Verify_Step_by_Step.pdf" },
    { id: 42, title: "GPT-4 Technical Report", detail: "OpenAI technical report, 2023.", url: "https://arxiv.org/abs/2303.08774" },
    { id: 43, title: "GPT-5.4 Thinking System Card", detail: "OpenAI safety and preparedness card, 2026.", url: "https://openai.com/index/gpt-5-4-thinking-system-card/" },
    { id: 44, title: "How does GPT Obtain its Ability? Tracing Emergent Abilities of Language Models to their Sources", detail: "Fu, Peng, and Khot lineage analysis, December 2022.", url: "https://yaofu.notion.site/How-does-GPT-Obtain-its-Ability-Tracing-Emergent-Abilities-of-Language-Models-to-their-Sources-b9a57ac0fcf74f30a1ab9e3e36fa1dc1" },
    { id: 45, title: "To Code, or Not To Code? Exploring Impact of Code in Pre-training", detail: "Aryabumi et al., controlled code pretraining study, 2024.", url: "https://arxiv.org/abs/2408.10914" },
    { id: 46, title: "A Neural Probabilistic Language Model", detail: "Bengio et al., distributed neural language model, 2003.", url: "https://jmlr.org/papers/v3/bengio03a.html" },
    { id: 47, title: "Efficient Estimation of Word Representations in Vector Space", detail: "Mikolov et al., word2vec, 2013.", url: "https://arxiv.org/abs/1301.3781" },
    { id: 48, title: "Sequence to Sequence Learning with Neural Networks", detail: "Sutskever, Vinyals, and Le, encoder-decoder sequence learning, 2014.", url: "https://arxiv.org/abs/1409.3215" },
    { id: 49, title: "Neural Machine Translation by Jointly Learning to Align and Translate", detail: "Bahdanau, Cho, and Bengio, attention for neural translation, 2014.", url: "https://arxiv.org/abs/1409.0473" },
    { id: 50, title: "Deep Contextualized Word Representations", detail: "Peters et al., ELMo contextual embeddings, 2018.", url: "https://arxiv.org/abs/1802.05365" },
    { id: 51, title: "Universal Language Model Fine-tuning for Text Classification", detail: "Howard and Ruder, ULMFiT transfer learning, 2018.", url: "https://arxiv.org/abs/1801.06146" },
    { id: 52, title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", detail: "Devlin et al., masked-language-model pretraining, 2018.", url: "https://arxiv.org/abs/1810.04805" },
    { id: 53, title: "The Llama 4 Herd: The Beginning of a New Era of Natively Multimodal AI Innovation", detail: "Meta AI Llama 4 release, 2025.", url: "https://ai.meta.com/blog/llama-4-multimodal-intelligence/" },
    { id: 54, title: "Qwen3: Think Deeper, Act Faster", detail: "Alibaba Qwen3 hybrid reasoning model family, 2025.", url: "https://qwenlm.github.io/blog/qwen3/" },
    { id: 55, title: "Kimi K2: Open Agentic Intelligence", detail: "Moonshot AI Kimi K2 model repository and technical report, 2025.", url: "https://github.com/MoonshotAI/Kimi-K2" },
    { id: 56, title: "Holistic Evaluation of Language Models", detail: "Liang et al., HELM evaluation framework, 2022.", url: "https://arxiv.org/abs/2211.09110" },
    { id: 57, title: "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?", detail: "Bender et al., data, environmental, and social risk critique, 2021.", url: "https://dl.acm.org/doi/10.1145/3442188.3445922" },
    { id: 58, title: "The 2025 AI Index Report", detail: "Stanford HAI annual report on AI investment, capabilities, and policy, 2025.", url: "https://hai.stanford.edu/ai-index/2025-ai-index-report" },
    { id: 59, title: "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?", detail: "Jimenez et al., real-world software engineering benchmark, 2023.", url: "https://arxiv.org/abs/2310.06770" }
];

const lessons = [
    { id: "pretraining", title: "Pretraining made language a reusable interface", text: "A next-token objective over broad text turned out to create features that could transfer across tasks, replacing many task-specific NLP pipelines." },
    { id: "scale", title: "Scale was a scientific instrument", text: "GPT-2, scaling laws, GPT-3, PaLM, and later frontier systems showed that bigger training runs could expose capabilities before theory could explain them." },
    { id: "data", title: "Data quality changed the slope", text: "Chinchilla and the open-model wave made clear that tokens, deduplication, code, filtering, and synthetic examples could matter as much as raw parameter count." },
    { id: "alignment", title: "Alignment converted models into products", text: "Instruction tuning and RLHF did not invent all capabilities from scratch; they made latent capabilities easier, safer, and more predictable to access." },
    { id: "code", title: "Code made reasoning executable", text: "Training on programs taught models to track variables, satisfy constraints, call tests, and revise outputs against external feedback." },
    { id: "cot", title: "Reasoning improved when models had room to work", text: "Chain of thought, self-consistency, verifiers, and process supervision showed that an answer can improve when inference includes intermediate work." },
    { id: "tools", title: "Tools made language models less trapped in language", text: "Retrieval, calculators, interpreters, browsers, and computer control moved LLMs from answer engines toward systems that can gather evidence and act." },
    { id: "open", title: "Open weights made the field plural", text: "Llama, Mistral, Qwen, DeepSeek, Kimi, and gpt-oss turned frontier ideas into a competitive ecosystem of local, specialized, and audited models." },
    { id: "multimodal", title: "The interface expanded beyond text", text: "Vision, speech, audio, video, and screen control changed the assistant from a text box into an operator over human work surfaces." },
    { id: "testtime", title: "Test-time compute became its own scaling axis", text: "The o-series, DeepSeek-R1, Claude hybrid reasoning, GPT-5 Pro, and GPT-5.5 shifted part of progress from training scale to inference-time search, checking, and persistence." },
    { id: "agents", title: "Agents exposed the reliability gap", text: "Long-horizon work needs memory, tool discipline, state tracking, rollback, verification, and permission boundaries, not just fluent text." },
    { id: "future", title: "The next fork is trustworthy autonomy", text: "The next major systems will be judged by whether they can do valuable work under verification, policy, latency, budget, and audit constraints." }
];

const chapters = [
    {
        id: "before-transformers",
        title: "Before Transformers: Language Modeling Becomes Learnable",
        range: "1950s-2017",
        era: "Prehistory",
        tags: ["Language modeling", "Embeddings", "Seq2seq", "Attention"],
        dek: "Transformers did not appear from nowhere. They inherited decades of work on predicting text, representing meaning as vectors, learning sequences end to end, and deciding which context matters.",
        lessonIds: ["pretraining", "data", "scale"],
        sourceIds: [46, 47, 48, 49, 50, 51],
        sections: [
            {
                heading: "Prediction came before intelligence",
                paragraphs: [
                    "Long before anyone spoke of foundation models, language modeling meant assigning probabilities to text. Statistical systems used counts, smoothing, and n-grams to guess what word should come next. They were narrow, but they established the core interface: a model of language can be tested by how well it predicts language.",
                    "The neural turn changed what could be shared across contexts. Bengio and collaborators' neural probabilistic language model learned distributed word representations while predicting sequences, attacking the curse of dimensionality that made count-based models brittle [[46]]. The key idea was not scale yet; it was that words and contexts could live in a learned continuous space.",
                    "That idea is one of the hidden roots of modern LLMs. A model does not store language as a dictionary of exact phrases. It learns geometry: nearby representations, reusable features, and smooth generalization from seen contexts to unseen ones."
                ]
            },
            {
                heading: "Vectors made meaning portable",
                paragraphs: [
                    "Word2vec made representation learning feel concrete. Its simple architectures produced useful word vectors from large text corpora, and those vectors made semantic regularities visible enough to become a cultural object inside machine learning [[47]]. Meaning was no longer only a symbol table; it was a direction in a space learned from use.",
                    "The limitation was also clear. A single vector for a word cannot fully represent context. The word 'bank' changes with a river, a loan, or a memory. Static embeddings made language portable, but they did not solve composition, long-range dependencies, or ambiguity.",
                    "Modern LLMs inherited both the strength and the limitation. The strength was that self-supervised prediction can discover useful representations without hand labels. The limitation was that language requires context at every step, not only a lookup table at the beginning."
                ]
            },
            {
                heading: "Sequence models exposed the bottleneck",
                paragraphs: [
                    "Neural sequence-to-sequence learning replaced brittle translation pipelines with encoder-decoder models trained end to end [[48]]. It proved that one neural system could map one variable-length sequence to another, which mattered for translation, summarization, dialogue, and every later text-in text-out interface.",
                    "Attention solved a specific failure mode. Fixed-length encodings forced the decoder to compress an entire source sentence into one vector. Bahdanau, Cho, and Bengio instead let the decoder soft-search over source positions while generating each target word, making alignment a learned operation rather than a hand-coded table [[49]].",
                    "ELMo and ULMFiT then pushed contextual pretraining and fine-tuning toward the center of NLP. ELMo showed that representations should vary by context [[50]], while ULMFiT showed that language-model pretraining could transfer strongly to downstream classification with limited labels [[51]]. By the time the Transformer arrived, the field already knew the destination: broad pretraining, reusable representations, and task adaptation. What it lacked was a scalable machine."
                ],
                callout: "The Transformer was a decisive architecture, but the deeper story is older: learn language from prediction, represent meaning in vectors, condition on context, and reuse the result across tasks."
            }
        ]
    },
    {
        id: "transformer-roots",
        title: "The Transformer Bet",
        range: "2017-2018",
        era: "Roots",
        tags: ["Transformers", "Transfer learning", "NLP"],
        dek: "The modern LLM story starts before GPT-1, with a change in architecture and a change in attitude: language models became general-purpose representation learners.",
        lessonIds: ["pretraining", "scale"],
        sourceIds: [1, 2, 52],
        sections: [
            {
                heading: "From sequence models to attention",
                paragraphs: [
                    "Before the transformer, strong NLP systems were usually built around recurrent neural networks, convolutional encoders, hand-designed task heads, and supervised datasets. They could translate, tag, classify, or answer questions, but the field often treated each benchmark as a separate engineering problem. The transformer changed the computational shape of the field: attention let every token directly attend to other tokens, made training much more parallel, and turned long-range dependencies into something the hardware could exploit [[1]].",
                    "The original transformer paper was not primarily a chatbot paper. It was a sequence-to-sequence machine translation paper. Its deeper effect was that it made a simple bet plausible: if a model can repeatedly mix information across positions, and if that operation scales well on accelerators, then one architecture can absorb more data, more tasks, and more modalities than earlier sequence models could comfortably handle [[1]].",
                    "That bet mattered because language is both data and interface. Text contains facts, procedures, style, code, arguments, logs, tables, and instructions. A model trained to predict the next piece of text is not only learning grammar; it is learning compressed regularities of many human activities."
                ]
            },
            {
                heading: "BERT and GPT split the transformer in two directions",
                paragraphs: [
                    "The transformer immediately forked into two major families. Encoder-style models such as BERT used bidirectional context and became excellent at understanding and classification [[52]]. Decoder-only models such as GPT used causal prediction and became excellent at generation. The later public imagination mostly remembers the decoder-only branch because it led to ChatGPT, but the encoder branch proved that pretraining could replace large amounts of task-specific feature engineering.",
                    "The GPT branch kept the training objective brutally simple. Predict the next token. The system would not be handed a database schema for every task or an ontology for every domain. It would learn from broad text and later be steered by examples, instructions, feedback, or tools.",
                    "That simplicity had a strategic advantage. Once the architecture and objective were stable, the field could run a repeated experiment: scale model, scale data, scale compute, measure what breaks, and then fix the next bottleneck."
                ]
            },
            {
                heading: "The first lesson: generality can be trained indirectly",
                paragraphs: [
                    "GPT-1 did not arrive as an isolated miracle. It was part of a larger turn toward unsupervised or self-supervised pretraining. The important conceptual move was that a model could learn a broad language prior from unlabeled text and then adapt to downstream tasks with far less supervised data than a model trained from scratch [[2]].",
                    "That sounds obvious now, but it was not obvious at the time. The dominant question in NLP had long been how to build a model for a task. GPT reframed the question: how do you build a model that has already absorbed enough language structure that the task becomes a thin steering layer?"
                ],
                callout: "The transformer supplied the scalable machinery. Generative pretraining supplied the reusable substrate. Everything after GPT-1 is a story about how to steer, scale, specialize, verify, and productize that substrate."
            }
        ]
    },
    {
        id: "gpt-1",
        title: "GPT-1: Pretraining Becomes a Platform",
        range: "2018",
        era: "GPT",
        tags: ["GPT-1", "Transfer", "Fine-tuning"],
        dek: "GPT-1 was small by later standards, but it named the recipe that would dominate the next decade: generative pretraining followed by task adaptation.",
        lessonIds: ["pretraining", "data"],
        sourceIds: [2],
        sections: [
            {
                heading: "The recipe",
                paragraphs: [
                    "GPT-1 used a decoder-only transformer trained on a large corpus with a language modeling objective, then fine-tuned on supervised NLP tasks. Its key claim was not that the model was already a universal assistant. It was that a single pretraining run could create a transferable representation useful across natural language understanding benchmarks [[2]].",
                    "The model had about 117 million parameters. Later models would make that number look tiny, but scale was not the headline yet. The headline was that unsupervised generative training could initialize a model with enough structure that task-specific fine-tuning became much easier.",
                    "GPT-1 also established a habit that still defines the field: the same model can be evaluated across a suite of tasks that were not individually baked into the architecture. That evaluation style made broad capability visible."
                ]
            },
            {
                heading: "Why generation mattered",
                paragraphs: [
                    "A generative objective forces the model to represent many forms of knowledge in a way that can be used to continue text. It has to learn syntax, entity relations, discourse flow, and patterns of question and answer because all of those improve token prediction.",
                    "That does not mean the model understands in the human sense. It means the objective creates pressure to learn internal structures that are reusable. The later surprise was how far that pressure could go when model size, data, and compute were increased by orders of magnitude.",
                    "GPT-1 therefore sits at the beginning of a pattern: first a capability appears as a fuzzy side effect of prediction, then prompting or fine-tuning exposes it, then product design turns it into a user-visible feature."
                ]
            },
            {
                heading: "What GPT-1 did not solve",
                paragraphs: [
                    "GPT-1 still depended on supervised fine-tuning for each target task. It had no general chat interface, no robust instruction following, no retrieval, no tools, no multimodal input, and no built-in mechanism for extended reasoning. It was a platform seed, not a product.",
                    "The model also inherited the limits of its data and objective. It could imitate patterns, but it did not have a systematic way to ask what the user wanted, distinguish helpfulness from mere continuation, or verify that an answer was true.",
                    "Those gaps became the map for later forks: scale for broader competence, instruction tuning for user intent, code for procedural reliability, retrieval for freshness, tools for action, and test-time compute for harder reasoning."
                ],
                callout: "GPT-1's lasting contribution was not a benchmark score. It was the proof that generative pretraining could be the base layer for many different language tasks."
            }
        ]
    },
    {
        id: "gpt-2",
        title: "GPT-2 and the Zero-Shot Shock",
        range: "2019",
        era: "GPT",
        tags: ["GPT-2", "Zero-shot", "Release policy"],
        dek: "GPT-2 made the field confront two ideas at once: unsupervised language models were becoming surprisingly general, and deployment choices were becoming part of AI research.",
        lessonIds: ["scale", "alignment"],
        sourceIds: [3],
        sections: [
            {
                heading: "From transfer to unsupervised multitask behavior",
                paragraphs: [
                    "GPT-2 scaled the GPT recipe to 1.5 billion parameters and was trained on a much larger web-derived dataset. OpenAI framed the model as an unsupervised multitask learner because it could perform rudimentary translation, summarization, question answering, and reading comprehension from raw text patterns without task-specific training [[3]].",
                    "That result changed expectations. GPT-1 showed that pretraining helped supervised fine-tuning. GPT-2 suggested that enough pretraining could make tasks appear inside the model itself. The task could be expressed as text, and the model could continue the pattern.",
                    "In hindsight, GPT-2 was the first public glimpse of prompt programming. Users were not yet chatting with a polished assistant, but they were starting to discover that the prompt was an executable control surface."
                ]
            },
            {
                heading: "The staged-release moment",
                paragraphs: [
                    "GPT-2 also became famous for its release process. OpenAI initially withheld the largest model and released smaller checkpoints first, citing concerns about misuse. This was controversial, but it foreshadowed a persistent issue: model publication, API access, weights, model cards, red-teaming, and product gates would all become part of the history of LLMs [[3]].",
                    "The staged release was not simply a safety footnote. It marked the beginning of a shift from academic artifacts to deployable systems. A language model could now generate text good enough that release strategy had to be discussed alongside architecture.",
                    "This is one reason the LLM story has so many forks. Some labs pushed closed APIs and controlled deployments. Others pushed open weights, local inference, community fine-tuning, and rapid iteration. Both branches learned from GPT-2."
                ]
            },
            {
                heading: "The lesson: capability arrives before control",
                paragraphs: [
                    "GPT-2 generated coherent passages, but it was not aligned to user intent. It could continue a false premise as fluently as a true one. It could imitate a style without caring about whether the output was useful. It could produce plausible text without grounding.",
                    "That mismatch between fluency and reliability became one of the central tensions of the field. Users respond to fluent language as if it carries authority. Model training rewards prediction. The gap between those two facts created the later demand for RLHF, retrieval, citations, abstention, and calibrated uncertainty.",
                    "GPT-2 therefore taught a hard lesson: scaling a language model can create broad capability, but the model still needs a separate layer that makes it answer the question people actually mean."
                ]
            }
        ]
    },
    {
        id: "gpt-3-scaling",
        title: "Scaling Laws and GPT-3",
        range: "2020",
        era: "Scaling",
        tags: ["GPT-3", "Scaling laws", "In-context learning"],
        dek: "GPT-3 turned prompt examples into a programming interface and scaling laws into a roadmap for industrial AI.",
        lessonIds: ["scale", "pretraining", "data"],
        sourceIds: [4, 5],
        sections: [
            {
                heading: "Scaling becomes predictable enough to budget",
                paragraphs: [
                    "OpenAI's scaling-law work measured how language-model loss changed with model size, dataset size, and training compute across many orders of magnitude [[5]]. The practical implication was enormous: labs could estimate whether a larger run was likely to pay off before spending the money.",
                    "Scaling laws did not say that every downstream behavior would be smooth. They said that the core prediction objective improved in regular ways. The field then discovered that many visible capabilities rode on top of that improvement, sometimes gradually and sometimes with threshold-like behavior.",
                    "The result was a new engineering culture. Training a frontier model became less like crafting a clever architecture for a benchmark and more like planning a capital-intensive system: data pipelines, accelerator clusters, distributed training, evaluations, safety reviews, and serving infrastructure."
                ]
            },
            {
                heading: "GPT-3 and the prompt as a program",
                paragraphs: [
                    "GPT-3 scaled to 175 billion parameters and popularized few-shot learning: put a few examples in the prompt, ask for another, and the model often infers the task without weight updates [[4]]. This made the prompt feel less like a query and more like a tiny program written in natural language.",
                    "The surprising part was not just that GPT-3 knew many facts. It could infer task format from context. A prompt containing examples of translation, sentiment classification, analogy, arithmetic, or style transfer could steer behavior at inference time. The model had not been fine-tuned for that exact task; it was being conditioned.",
                    "That changed the product imagination. If a task could be specified in text, maybe software could become more fluid. Instead of building a separate classifier, write examples. Instead of training a domain model, describe the output. This was powerful, brittle, and addictive."
                ]
            },
            {
                heading: "The cracks in pure scale",
                paragraphs: [
                    "GPT-3 also showed that scale alone was not enough. It hallucinated. It could be sensitive to prompt wording. It struggled with multi-step arithmetic and logic. It could imitate unsafe or biased patterns. It had no durable memory and no access to current facts unless they appeared in training data.",
                    "These failures were not side issues; they defined the next wave of research. The model was capable enough to be useful but unreliable enough to require supervision. The industry learned that a raw base model is not the same thing as an assistant.",
                    "In-context learning gave users control, but it also shifted burden onto users. Prompt engineering became the folk practice of discovering hidden affordances inside a giant conditional generator. InstructGPT and ChatGPT would reduce that burden by training the model to treat user requests as requests rather than text to continue."
                ],
                callout: "GPT-3 made the LLM feel like a general interface. It also made clear that the next bottleneck was not only more parameters; it was steering."
            }
        ]
    },
    {
        id: "instruct-chatgpt",
        title: "Alignment Arrives: InstructGPT and ChatGPT",
        range: "2021-2022",
        era: "Alignment",
        tags: ["RLHF", "InstructGPT", "ChatGPT"],
        dek: "Instruction tuning and RLHF changed the user experience more than a raw benchmark table could capture.",
        lessonIds: ["alignment", "pretraining"],
        sourceIds: [7, 8, 18],
        sections: [
            {
                heading: "Why GPT-3 needed a translator for human intent",
                paragraphs: [
                    "Base GPT-3 was trained to continue text, not to help a user. If a user asked a question, the model might answer, ask another question, continue the transcript, or imitate an internet argument. The next breakthrough was to train the model on what people actually preferred.",
                    "InstructGPT used supervised demonstrations and reinforcement learning from human feedback. Labelers wrote desired answers, compared model outputs, and trained a reward model that could guide a policy toward helpful instruction following [[7]]. OpenAI emphasized that the alignment stage used a small fraction of the compute and data used for pretraining, yet it strongly changed behavior.",
                    "That fact is central. RLHF did not replace pretraining. It acted like a behavioral lens over capabilities that pretraining had already built. A smaller InstructGPT model could be preferred over a much larger base GPT-3 model because it was easier to use and less likely to ignore the user's intent [[7]]."
                ]
            },
            {
                heading: "The chat interface made the capability legible",
                paragraphs: [
                    "ChatGPT, released on November 30, 2022, packaged the InstructGPT lineage into a conversational product [[8]]. The product mattered because it made turn-taking, clarification, refusal, drafting, editing, and follow-up feel natural. The model was no longer just a completion endpoint; it was a collaborator-shaped interface.",
                    "The release also created the largest public feedback loop the field had seen. Users supplied prompts at internet scale, discovered strengths and failures, and pushed labs to improve safety, memory, tools, latency, and model personality. Chat became the default way non-specialists understood LLMs.",
                    "This is a recurring LLM lesson: a model capability becomes socially important only when a product surface makes it repeatable. ChatGPT did for instruction-following what GPT-3 did for prompting: it made an abstract capability visible."
                ]
            },
            {
                heading: "The alignment fork",
                paragraphs: [
                    "Anthropic developed a related but distinct alignment branch with Constitutional AI, using a written set of principles and AI-generated critiques to reduce dependence on human labels for harmlessness training [[18]]. That fork shaped Claude and the broader debate about whether models should be aligned mostly by human preference, explicit principles, automated supervision, or some mixture.",
                    "The alignment era also revealed tradeoffs. RLHF can make models more helpful, but it can also make them deferential, verbose, overconfident, or excessively cautious. A model optimized to please raters can learn style signals that are not identical to truth.",
                    "This is why later frontier systems invested in model specs, system cards, red-team programs, safety classifiers, process supervision, and explicit rules for uncertainty. Alignment began as making GPT-3 follow instructions. It became the broader science of making powerful models behave under pressure."
                ],
                callout: "InstructGPT's lesson is easy to miss: usefulness is not only a matter of intelligence. It is a trained relationship between model behavior and user intent."
            }
        ]
    },
    {
        id: "code-reasoning",
        title: "Code Teaches Models to Think in Procedures",
        range: "2021-2023",
        era: "Code",
        tags: ["Codex", "AlphaCode", "HumanEval"],
        dek: "The code branch of LLM history changed reasoning, evaluation, and product expectations because programs can be executed and tested.",
        lessonIds: ["code", "cot", "tools"],
        sourceIds: [9, 10, 11, 12, 44, 45],
        sections: [
            {
                heading: "Codex made language executable",
                paragraphs: [
                    "OpenAI Codex was a GPT-family model trained on public code and evaluated on HumanEval, a benchmark where generated Python functions are checked by unit tests [[9]]. That mattered because code gave LLMs a training distribution full of procedures, constraints, variable state, API boundaries, and explicit failure.",
                    "Natural language often tolerates vagueness. Code does not. A program either satisfies tests or it does not. That made coding one of the first major domains where LLM outputs could be automatically scored beyond surface plausibility.",
                    "Codex also made the assistant idea more concrete. If a model can turn a docstring into a function, it can act as a pair programmer. GitHub Copilot then showed that the product value was not limited to solving benchmark prompts; it was autocomplete, boilerplate removal, API recall, test drafting, and keeping developers in flow."
                ]
            },
            {
                heading: "Sampling, filtering, and search",
                paragraphs: [
                    "AlphaCode pushed a related idea in competitive programming. It generated many candidate programs, filtered them by behavior, and selected submissions, reaching roughly median human competitor level on Codeforces-style tasks [[10]]. The lesson was not just that the model knew syntax. It was that generation plus search plus filtering could solve problems a single greedy answer would miss.",
                    "This pattern became a bridge to test-time compute. If one sample is unreliable, sample many. If many are available, evaluate them. If evaluation is cheap and objective, the system can improve without changing the base model. Code was the perfect early domain because tests are verifiers.",
                    "The same idea later reappeared in math reasoning, tool-use agents, SWE-bench style issue resolution, and frontier coding models. Modern coding agents do not merely write text; they inspect repos, edit files, run tests, read failures, and iterate."
                ]
            },
            {
                heading: "Did coding teach reasoning?",
                paragraphs: [
                    "It is too strong to say that code alone taught LLMs to reason. Large-scale text, math, dialogue, tool use, and RL all contributed. But code appears to have been unusually useful because it is a dense record of procedural thought. It teaches decomposition, abstraction, reference, invariants, and repair.",
                    "A widely circulated lineage analysis by Yao Fu, Hao Peng, and Tushar Khot traced GPT-3.5-era models through InstructGPT, Codex, and code-davinci-002. It argued that complex reasoning and chain-of-thought behavior were likely side effects of training on code while explicitly labeling the claim as a hypothesis rather than proof [[44]].",
                    "The clue came from several places. Codex dramatically outperformed GPT-3 on code synthesis [[9]]. Code-oriented GPT-3 variants such as code-davinci-002 were unexpectedly strong in reasoning prompting studies. Least-to-most prompting results found especially strong compositional generalization with a code-trained model. Chain-of-thought and self-consistency then made the model's intermediate work explicit [[11]][[12]].",
                    "Later controlled work strengthened the case without making it monocausal: adding code to pretraining improved natural-language reasoning, world knowledge, generative win rates, and code performance in a systematic ablation, but the broader LLM reasoning stack still depends on instruction tuning, math data, RL, scale, tools, and inference-time search [[45]]."
                ],
                callout: "Coding models turned LLM output into something with a feedback loop: write, run, fail, inspect, repair. That loop is now central to frontier agents."
            }
        ]
    },
    {
        id: "reasoning-prompts",
        title: "Reasoning Is Elicited",
        range: "2022-2023",
        era: "Reasoning",
        tags: ["Chain of thought", "Self-consistency", "Verifiers"],
        dek: "The reasoning wave began when researchers realized that models often perform better if inference includes visible intermediate work.",
        lessonIds: ["cot", "testtime"],
        sourceIds: [11, 12, 39, 40, 41],
        sections: [
            {
                heading: "Chain of thought",
                paragraphs: [
                    "Chain-of-thought prompting showed that large models could solve harder math and reasoning tasks when prompted to produce intermediate steps before an answer [[11]]. This did not require changing the model weights. It changed the format of inference.",
                    "The result shifted how people thought about LLM capability. A model's answer quality was no longer just a property of the parameters. It also depended on how much working memory, scratch space, and sampling budget the inference procedure allowed.",
                    "The key phrase is 'large models.' Chain of thought was far more useful once models crossed a capability threshold. Smaller models could generate steps, but the steps were often decorative. Larger models were more likely to make intermediate text do real computational work."
                ]
            },
            {
                heading: "Self-consistency and search over thoughts",
                paragraphs: [
                    "Self-consistency sampled multiple reasoning paths and selected the most common final answer, improving performance on arithmetic and commonsense reasoning benchmarks [[12]]. Tree of Thoughts generalized the idea by exploring branches of intermediate reasoning and allowing search, backtracking, and evaluation [[39]].",
                    "These methods made inference look less like one forward pass and more like a planning algorithm wrapped around a model. The base model proposes. The outer loop samples, scores, compares, or branches. This was an early version of the test-time compute story.",
                    "STaR added another loop: generate rationales, keep ones that lead to correct answers, fine-tune, and repeat [[40]]. That hinted at a self-improvement pattern: models can produce candidate reasoning traces that become training data, as long as some signal can distinguish useful traces from noise."
                ]
            },
            {
                heading: "Process supervision",
                paragraphs: [
                    "OpenAI's process-supervision work argued that supervising intermediate steps can be more informative than only supervising final answers for mathematical reasoning [[41]]. This matters because a final answer can be right for the wrong reason, while step-level feedback can teach a model where a solution went off track.",
                    "The broader lesson is that reasoning systems need supervision at the level where errors occur. In long tasks, the failure might be an early assumption, a missed file, a wrong unit, or an invalid tool result. Outcome-only reward can be too sparse.",
                    "By 2023, the field had assembled many ingredients that later reasoning models would productize: scratchpads, sampled reasoning, verifiers, step supervision, tool calls, and outer-loop search. The public would see those ingredients crystallize in o1, DeepSeek-R1, Claude hybrid reasoning, and GPT-5-series thinking modes."
                ],
                callout: "Reasoning was not a single invention. It was a stack: prompts that make work visible, sampling that explores alternatives, verifiers that select, and training that rewards good process."
            }
        ]
    },
    {
        id: "compute-optimal",
        title: "Chinchilla, Data Quality, and Efficient Scale",
        range: "2022-2024",
        era: "Efficiency",
        tags: ["Chinchilla", "Data", "Specialization"],
        dek: "Once scale worked, the field asked whether it had been scaling the wrong things.",
        lessonIds: ["data", "scale", "open"],
        sourceIds: [6, 13, 20, 21],
        sections: [
            {
                heading: "The Chinchilla correction",
                paragraphs: [
                    "The Chinchilla paper argued that many large language models were undertrained: for a fixed compute budget, it could be better to use a smaller model trained on many more tokens than a larger model trained on fewer tokens [[6]]. DeepMind's Chinchilla was 70B parameters, much smaller than Gopher, but trained with substantially more data.",
                    "This changed the frontier recipe. Parameter count was no longer the only status symbol. Tokens, data mixture, repetition, deduplication, curriculum, and compute allocation became central. A model could be too big for its data.",
                    "The correction also made open models more competitive. If a smaller, better-trained model could rival a larger undertrained one, then labs without the very largest clusters could still produce useful systems by improving data and training efficiency."
                ]
            },
            {
                heading: "PaLM and scale after GPT-3",
                paragraphs: [
                    "Google's PaLM scaled a dense transformer to 540B parameters and showed strong few-shot and chain-of-thought performance [[13]]. PaLM reinforced the idea that scale still mattered, especially when paired with better prompting and infrastructure.",
                    "The important nuance is that PaLM and Chinchilla were not contradictory. PaLM demonstrated the power of very large models. Chinchilla sharpened the question of how to spend compute. Later systems would use both lessons: scale when it pays, but spend aggressively on data quality, post-training, and inference efficiency.",
                    "This period also raised the cost of frontier competition. Training was no longer a single model run; it was a sequence of data experiments, ablations, scaling predictions, checkpoint evaluations, safety reviews, and post-training recipes."
                ]
            },
            {
                heading: "Small models stop being toys",
                paragraphs: [
                    "Mistral 7B showed that a carefully engineered 7B model could outperform older larger models on many benchmarks [[20]]. Mixtral 8x7B then helped popularize sparse mixture-of-experts models in the open ecosystem, activating only part of the model per token while preserving higher total capacity [[21]].",
                    "This was part of a broader efficiency fork. Some labs raced for maximum frontier capability. Others optimized for local deployment, low latency, cheap inference, controllability, and specialization. Both mattered. Consumer hardware, mobile devices, enterprise privacy rules, and edge applications all created demand for smaller capable models.",
                    "By 2024, the field no longer believed in one model size for everything. It believed in a portfolio: tiny models for routing and extraction, medium models for local assistants, large models for complex work, and reasoning models when a task deserved extra inference budget."
                ],
                callout: "The scaling lesson matured from 'make it bigger' to 'allocate compute across parameters, data, post-training, tools, and inference where it buys the most reliability.'"
            }
        ]
    },
    {
        id: "open-weight",
        title: "The Open and Global Model Explosion",
        range: "2023-2026",
        era: "Open",
        tags: ["Llama", "Mistral", "Qwen", "DeepSeek", "Kimi"],
        dek: "Open-weight and permissively available models turned LLM progress from a closed-lab race into a global ecosystem of fine-tunes, local deployments, audits, distillation, and specialization.",
        lessonIds: ["open", "data", "testtime"],
        sourceIds: [19, 20, 21, 25, 27, 28, 32, 53, 54, 55],
        sections: [
            {
                heading: "Llama changes the distribution channel",
                paragraphs: [
                    "Meta's Llama family made strong foundation models widely available to researchers and developers. Llama 2 expanded access for commercial use in 2023 [[19]], Llama 3 continued the push in 2024 with larger training data and stronger code capability [[25]], and Llama 4 moved the family toward natively multimodal mixture-of-experts systems in 2025 [[53]].",
                    "The open-weight ecosystem did something closed APIs could not. It let users fine-tune, quantize, distill, inspect, host privately, and build local products. The community created instruction-tuned variants, domain models, safety-tuned models, and efficient runtimes at a pace no single lab could match.",
                    "This also created governance tension. Meta describes Llama as open, but model licenses, acceptable-use terms, release geography, and downstream restrictions vary across families. The practical split is not simply open versus closed; it is a spectrum from fully open research artifacts to gated weights, custom licenses, commercial APIs, and private frontier systems."
                ]
            },
            {
                heading: "Mistral, Qwen, DeepSeek, and Kimi widen the map",
                paragraphs: [
                    "Mistral showed that smaller open models could be highly competitive [[20]], and Mixtral made sparse expert routing a mainstream open-model design [[21]]. Alibaba's Qwen family, DeepSeek, Moonshot's Kimi line, and other labs then intensified the race around multilingual data, code, math, long context, and low-cost inference.",
                    "DeepSeek-V3 was especially important because it combined a very large total parameter count with sparse activation, reporting 671B total parameters but only 37B active per token [[27]]. That design pointed toward a future where model capacity and inference cost are decoupled.",
                    "Qwen3 made hybrid reasoning more explicit by letting users choose thinking and non-thinking modes across a broad model family [[54]]. Kimi K2 pushed the open agentic branch with a 1T-parameter MoE design, 32B activated parameters, tool-use emphasis, and a technical record centered on agentic capability [[55]]. DeepSeek-R1 then showed that open reasoning models could compete with closed reasoning systems on some public benchmarks, using reinforcement learning to elicit long-form reasoning behavior and distilling that behavior into smaller models [[28]]."
                ]
            },
            {
                heading: "Open is a deployment culture",
                paragraphs: [
                    "In 2025, OpenAI released gpt-oss-120b and gpt-oss-20b as open-weight reasoning models, its first open-weight model release since GPT-2 [[32]]. The release was a sign that open weights had become too strategically important to ignore.",
                    "The open branch changed how frontier labs compete. A closed model can lead on maximum capability, but open models can win on cost, deployment control, community trust, local latency, customization, and ecosystem energy. Quantization, GGUF files, llama.cpp, vLLM, Ollama, LoRA fine-tunes, synthetic data, and distillation became part of model history because they changed who could build.",
                    "The deepest lesson is that LLM history is not a straight line from GPT-1 to the latest OpenAI model. It is a tree. GPT influenced Llama; Llama enabled community tuning; Mistral and DeepSeek pushed efficient architecture and training; Qwen and Kimi made the frontier more geographically plural; open reasoning put pressure back on closed labs; closed labs then integrated better coding, tool use, and inference scaling."
                ],
                callout: "Open weights made the frontier reproducible enough to fork, and forking made the frontier move faster."
            }
        ]
    },
    {
        id: "multimodal-frontier",
        title: "GPT-4, Claude, Gemini, and Multimodal Assistants",
        range: "2023-2024",
        era: "Multimodal",
        tags: ["GPT-4", "Claude", "Gemini", "GPT-4o"],
        dek: "The next major shift was not only smarter text. Models began to see, hear, speak, use longer context, and fit into everyday workflows.",
        lessonIds: ["multimodal", "alignment", "tools"],
        sourceIds: [14, 22, 23, 24, 42],
        sections: [
            {
                heading: "GPT-4 and the closed frontier",
                paragraphs: [
                    "GPT-4 marked a major capability jump in 2023, with strong performance on professional and academic benchmarks and a multimodal direction through image understanding [[14]][[42]]. It also marked a transparency shift: OpenAI released extensive evaluation and safety information, but not the model's full architecture, size, training compute, or dataset details.",
                    "That choice became characteristic of the frontier. As model capability increased, labs shared fewer low-level training details and more system cards, evaluations, and policy frameworks. The artifact was no longer just a paper; it was a deployed system with risk controls.",
                    "GPT-4 also normalized the idea that LLMs could be evaluated as broad intellectual systems. The benchmarks included exams, coding, reasoning, safety behavior, and multimodal tasks. This changed public expectations for what a model release should prove."
                ]
            },
            {
                heading: "Claude and constitutional product design",
                paragraphs: [
                    "Anthropic's Claude line emphasized helpful, honest, harmless behavior and long-context workflows. Claude 3, released in 2024, offered Haiku, Sonnet, and Opus tiers, each trading off speed and capability [[23]]. That family structure became common: one brand, multiple models, different cost and latency points.",
                    "Claude's importance was not only benchmarks. It gave the market another strong closed frontier with a distinct style, a distinct safety philosophy, and a strong following among writers, analysts, and programmers. Competition forced all labs to improve usability, not just raw scores.",
                    "The Claude branch also showed that model personality matters. A model can be technically capable but unpleasant, evasive, sycophantic, or hard to steer. Post-training became a product discipline."
                ]
            },
            {
                heading: "Gemini and native multimodality",
                paragraphs: [
                    "Google's Gemini line pushed a model family designed for multimodal reasoning across text, image, audio, video, and code, with Ultra, Pro, and Nano tiers in the initial release [[22]]. This was a strategic answer to both GPT-4 and the reality that Google controlled enormous multimodal product surfaces.",
                    "GPT-4o then made real-time multimodal interaction feel central rather than auxiliary. OpenAI described GPT-4o as accepting text, audio, image, and video and generating text, audio, and image outputs, with a faster and more natural interaction style [[24]].",
                    "The lesson was that LLMs were leaving the document. They were moving into voice calls, cameras, screenshots, spreadsheets, IDEs, phones, browsers, and operating systems. Once the model can perceive the same surface the user sees, the assistant becomes a participant in the workflow rather than a separate text box."
                ],
                callout: "Multimodality changed the unit of work. The prompt was no longer just text; it could be a screen, a diagram, a voice exchange, or a messy folder of artifacts."
            }
        ]
    },
    {
        id: "tools-agents",
        title: "Tools, Retrieval, and Agents",
        range: "2020-2025",
        era: "Agents",
        tags: ["RAG", "Tools", "Agents"],
        dek: "The model became one component in a larger system: retrieve evidence, call tools, write code, inspect results, and keep state.",
        lessonIds: ["tools", "agents", "alignment"],
        sourceIds: [15, 16, 17, 29, 30, 31],
        sections: [
            {
                heading: "Retrieval fights stale memory",
                paragraphs: [
                    "Retrieval-augmented generation combined a parametric language model with a retrieval system, allowing generated answers to be conditioned on retrieved documents [[15]]. In practice, RAG became one of the most important enterprise patterns because it let companies ground models in private or current knowledge without retraining the base model.",
                    "RAG also clarified a distinction. A model's weights are not a database. They are a compressed statistical memory. Retrieval is useful when the answer depends on freshness, provenance, private data, or exact wording. The model reasons over retrieved context, but the system supplies the evidence.",
                    "This pattern later merged with long context. Some tasks use retrieval to select context. Some stuff massive documents directly into the model. The best systems often do both: retrieve, rank, compress, cite, and then reason."
                ]
            },
            {
                heading: "Tool use becomes an outer loop",
                paragraphs: [
                    "ReAct showed how reasoning traces and actions could be interleaved, letting a model think, call an external source or environment, observe the result, and continue [[16]]. Toolformer explored self-supervised training for deciding when and how to call APIs [[17]].",
                    "These ideas anticipated modern agent frameworks. A tool-using model is not limited to what is in its weights. It can search, calculate, run code, query a database, open a browser, or operate a computer. Each tool call converts language into state change.",
                    "The hard part is not calling a tool once. The hard part is knowing when to call it, checking whether the output is trustworthy, recovering from failure, and not taking actions beyond the user's intent. That is why agentic systems need permissions, logs, sandboxes, and human review."
                ]
            },
            {
                heading: "Coding agents become the proving ground",
                paragraphs: [
                    "By 2025, coding was the clearest agentic domain. Claude 3.7 Sonnet launched with Claude Code as a collaborator that could search and read code, edit files, run tests, and use command-line tools [[29]]. OpenAI's o3 and o4-mini were described as reasoning models that could use and combine ChatGPT tools, including browsing, files, Python, visual input, and image generation [[30]].",
                    "OpenAI's GPT-4.1 family then focused on coding, instruction following, and long context in the API [[31]]. The trend was clear: the model was being surrounded by affordances that let it interact with real artifacts.",
                    "Agents exposed a new reliability bar. A chatbot can be useful with an imperfect answer. An agent editing a repo or operating a browser has to preserve user intent across many steps, avoid damaging state, and produce auditable work. This is why coding agents pushed models toward planning, testing, rollback, and status reporting."
                ],
                callout: "An agent is not a big prompt. It is a control system around a model: state, tools, memory, permissions, verification, and recovery."
            }
        ]
    },
    {
        id: "test-time-compute",
        title: "Test-Time Compute Becomes a Product",
        range: "2024-2026",
        era: "Reasoning",
        tags: ["o1", "DeepSeek-R1", "GPT-5"],
        dek: "The o-series and its successors made a new scaling law feel practical: spend more compute while answering when the problem deserves it.",
        lessonIds: ["testtime", "cot", "code"],
        sourceIds: [26, 28, 29, 30, 33, 36, 38],
        sections: [
            {
                heading: "o1 productizes thinking time",
                paragraphs: [
                    "OpenAI o1, released in preview in September 2024, was presented as a model trained to spend more time thinking before answering, with stronger performance on hard reasoning tasks in science, math, and programming [[26]]. This moved chain-of-thought style ideas from research prompts into a product mode.",
                    "The model's core product claim was not simply a larger base model. It was that the system could allocate more inference work to hard problems. Users experienced this as latency: the model paused, reasoned, and then responded.",
                    "That changed how people judged model quality. A fast answer was no longer always the best answer. For difficult tasks, a slower model that could plan, check, and revise was worth paying for."
                ]
            },
            {
                heading: "Open reasoning changes the race",
                paragraphs: [
                    "DeepSeek-R1 made reasoning-model techniques visible in the open ecosystem. Its paper described reinforcement learning that incentivized reasoning capability, including a R1-Zero path that showed strong reasoning behavior emerging under outcome-focused RL before additional supervised and alignment stages [[28]].",
                    "The release mattered because it reduced mystery. It showed that the reasoning wave was not only closed-lab magic. A capable base model, a reward signal, sampling, reinforcement learning, distillation, and careful post-training could create systems that spend many tokens solving hard problems.",
                    "Anthropic's Claude 3.7 Sonnet then framed reasoning as hybrid: the same model could answer quickly or use extended thinking when needed [[29]]. That anticipated the unified-model direction later seen in GPT-5."
                ]
            },
            {
                heading: "From o3 to GPT-5 and GPT-5.5",
                paragraphs: [
                    "OpenAI o3 and o4-mini, released in April 2025, extended reasoning into multimodal and tool-using contexts, with OpenAI emphasizing the ability to use and combine tools while reasoning [[30]]. GPT-5 then unified a fast model, a deeper thinking model, and a router that decides when to use each [[33]].",
                    "GPT-5 Pro explicitly used scaled parallel test-time compute for harder tasks [[33]]. GPT-5.4 and GPT-5.5 continued that line, emphasizing agentic coding, computer use, knowledge work, scientific workflows, lower token use, and better long-horizon performance [[36]][[38]].",
                    "This is the most important current fork. Pretraining scale still matters. Data still matters. But for many valuable tasks, the system's answer quality now depends on how much inference budget it can spend exploring, verifying, using tools, and persisting through failures."
                ],
                callout: "The field moved from 'what does the model know?' to 'how much work can the system do before it answers, and how can that work be verified?'"
            }
        ]
    },
    {
        id: "industrial-stack",
        title: "The Industrial Stack: Data, Chips, Benchmarks, and Governance",
        range: "2021-2026",
        era: "Industry",
        tags: ["Data labor", "Benchmarks", "Governance", "Economics"],
        dek: "LLM history is not only a sequence of model releases. It is also a history of data pipelines, labor markets, compute supply, benchmark politics, copyright fights, safety institutions, and platform power.",
        lessonIds: ["data", "alignment", "agents", "future"],
        sourceIds: [56, 57, 58, 59],
        sections: [
            {
                heading: "Scale made model building industrial",
                paragraphs: [
                    "The stochastic-parrots critique landed before ChatGPT but named issues that became central afterward: the environmental and financial cost of scale, the hazards of indiscriminate web data, and the social consequences of deploying systems trained on opaque corpora [[57]]. Whether one agrees with every conclusion or not, the paper correctly anticipated that LLM history would become governance history.",
                    "By the mid-2020s, training frontier models required data centers, custom accelerators, networking, power contracts, model-risk teams, red-teaming, policy review, evaluation infrastructure, and inference fleets. Stanford's AI Index tracks this wider industrialization through investment, model releases, benchmark progress, compute, policy activity, and international competition [[58]].",
                    "The cost structure changed product strategy. A lab could have a brilliant model and still lose on serving cost, latency, memory bandwidth, or enterprise trust. That is why mixture-of-experts, quantization, distillation, caching, routing, and smaller specialized models became historical forces rather than implementation details."
                ]
            },
            {
                heading: "Benchmarks became contested infrastructure",
                paragraphs: [
                    "As model claims became more consequential, evaluation had to become broader and more transparent. HELM argued for holistic evaluation across scenarios, metrics, prompts, and models, releasing raw prompts and completions to make comparisons inspectable rather than just leaderboard theater [[56]].",
                    "Coding exposed the same problem in a harder setting. SWE-bench used real GitHub issues and repository tests to ask whether models could repair software, not merely solve toy coding prompts [[59]]. Its influence came from a better unit of work: an issue, a repo, a patch, and a test suite.",
                    "The benchmark lesson is double-edged. Better benchmarks steer progress, but they can also become targets for overfitting, contamination, prompt sensitivity, and narrow optimization. The best model histories therefore should treat benchmark scores as evidence with provenance, not as final truth."
                ]
            },
            {
                heading: "The hidden supply chain became visible",
                paragraphs: [
                    "The public sees chat interfaces. The system depends on people who write demonstrations, rank outputs, label safety data, maintain data filters, evaluate failures, moderate abuse, build benchmarks, operate clusters, and negotiate licenses. RLHF made this obvious: the assistant was not only a neural network, but also a trained social interface.",
                    "Copyright and data provenance became part of technical history because dataset composition affects memorization, bias, factuality, language coverage, and legal exposure. Open models intensified the question: if weights circulate globally, who is accountable for the data, fine-tunes, derivatives, and downstream products?",
                    "Enterprise adoption added a different pressure. Buyers asked about privacy, retention, audit logs, compliance, residency, indemnity, uptime, and cost predictability. Those questions shaped product architectures as much as benchmark tables did."
                ],
                callout: "The model release is the visible artifact. The real historical object is the stack around it: data, labor, chips, evaluations, policy, product, and distribution."
            }
        ]
    },
    {
        id: "frontier-2026",
        title: "The Frontier in 2026 and What Comes Next",
        range: "Current through May 3, 2026",
        era: "Future",
        tags: ["GPT-5.5", "Gemini 3.1", "Claude Opus 4.7", "Future"],
        dek: "By May 2026, the frontier is a competition among agentic work systems, reasoning modes, open-weight ecosystems, and multimodal operating surfaces.",
        lessonIds: ["future", "agents", "testtime", "open"],
        sourceIds: [34, 35, 36, 37, 38, 43],
        sections: [
            {
                heading: "The May 2026 snapshot",
                paragraphs: [
                    "As of May 3, 2026, the latest public OpenAI frontier release is GPT-5.5, announced on April 23, 2026. OpenAI describes it as strongest in agentic coding, computer use, knowledge work, and early scientific research, with GPT-5.5 and GPT-5.5 Pro available in ChatGPT, Codex, and the API under staged safeguards [[38]].",
                    "The competitive frontier around it is broad. Anthropic released Claude Opus 4.7 in April 2026 with a focus on coding and complex work [[37]]. Google released Gemini 3.1 Pro in February 2026 as its most advanced model for complex tasks at the time of its model card [[35]]. OpenAI released GPT-5.4 Thinking and Pro in March 2026, and a GPT-5.4 Thinking system card documented safety mitigations and preparedness framing [[36]][[43]].",
                    "The meaning of 'frontier model' has changed. It no longer means a single chat model with the highest text benchmark. It means a system that can reason, see, use tools, operate software, write code, handle long context, follow policy, and complete multi-step work with fewer human corrections."
                ]
            },
            {
                heading: "The forks that brought us here",
                paragraphs: [
                    "The GPT line supplied the decoder-only scaling template. The BERT and encoder line supplied the proof that self-supervised pretraining could dominate understanding tasks. The alignment line turned raw prediction into instruction following. The code line supplied executable feedback and procedural data. The reasoning line supplied inference-time work. The tool line supplied action. The open line supplied competition, transparency, and specialization.",
                    "Every modern model is a braid of those forks. GPT-5.5 inherits GPT-style pretraining, instruction tuning, code-heavy capability, multimodal interfaces, tool use, reasoning-time allocation, agent loops, and safety systems. Claude Opus 4.7 inherits constitutional post-training, long-context product design, and coding-agent pressure. Gemini 3.1 Pro inherits Google's multimodal and product-surface advantage.",
                    "That is why a linear timeline is misleading. The field looks linear because public releases have dates. Underneath, it is a set of feedback loops: data improves models, models generate data, coding agents improve infrastructure, infrastructure enables bigger runs, open releases distill closed ideas, closed labs respond with better systems."
                ]
            },
            {
                heading: "A peek into the next era",
                paragraphs: [
                    "The next era is likely to be defined by trustworthy autonomy rather than chat fluency. Models will be asked to complete work that spans hours or days: software maintenance, research synthesis, data analysis, security triage, procurement, legal review, tutoring, design, and lab automation. The limiting factor will be reliable stateful execution.",
                    "Expect more explicit budgets. Users and systems will choose between fast answers, careful answers, parallel search, verified answers, and audited actions. Test-time compute will become a knob like temperature once was, but with economic and safety consequences.",
                    "Expect more verification. Code will be run. Math will be checked in proof assistants. Claims will be cited against retrieved sources. Spreadsheets will be recalculated. Browser actions will be logged. Agents that cannot show their work in auditable artifacts will be trusted less for consequential tasks."
                ]
            },
            {
                heading: "What might break the transformer era",
                paragraphs: [
                    "The transformer remains dominant, but the pressure points are clear: inference cost, long-horizon memory, data limits, energy, latency, hallucination, agency risk, and the difficulty of verifying reasoning traces. Future systems may keep transformers as components while adding external memory, symbolic verifiers, search, simulators, world models, or new sequence architectures.",
                    "The near future probably will not be a clean replacement. It will be a hybridization. The model will be one part of a larger cognitive operating system: planner, verifier, retriever, coder, browser, memory store, policy engine, and user interface.",
                    "The core lesson from GPT-1 to GPT-5.5 is that the field advances when a hidden capability becomes controllable. Pretraining made language reusable. Prompting made tasks steerable. RLHF made assistants usable. Code made reasoning testable. Tools made models actionable. Test-time compute made hard thinking scalable. The future belongs to systems that make autonomy verifiable."
                ],
                callout: "The most valuable 2026 models are no longer just models. They are work engines built from models, tools, memory, policy, and verification."
            }
        ]
    }
];

const timeline = [
    { date: "2003", era: "Prehistory", title: "Neural probabilistic language model", text: "Distributed representations enter neural language modeling.", chapter: "before-transformers", source: 46 },
    { date: "2013", era: "Prehistory", title: "Word2vec", text: "Efficient word vectors make learned semantic geometry widely useful.", chapter: "before-transformers", source: 47 },
    { date: "2014", era: "Prehistory", title: "Seq2seq", text: "Encoder-decoder models make sequence-to-sequence learning practical.", chapter: "before-transformers", source: 48 },
    { date: "2014", era: "Prehistory", title: "Neural attention", text: "Attention lets decoders select relevant source context instead of relying on one fixed vector.", chapter: "before-transformers", source: 49 },
    { date: "2017", era: "Roots", title: "Transformer architecture", text: "Attention replaces recurrence as the scalable core architecture for modern LLMs.", chapter: "transformer-roots", source: 1 },
    { date: "2018 Jan", era: "Prehistory", title: "ULMFiT", text: "Language-model pretraining transfers strongly to downstream NLP with limited labels.", chapter: "before-transformers", source: 51 },
    { date: "2018 Feb", era: "Prehistory", title: "ELMo", text: "Contextual word representations show that meaning should change with use.", chapter: "before-transformers", source: 50 },
    { date: "2018", era: "GPT", title: "GPT-1", text: "Generative pretraining plus fine-tuning becomes a transferable NLP recipe.", chapter: "gpt-1", source: 2 },
    { date: "2018 Oct", era: "Roots", title: "BERT", text: "Masked-language-model pretraining makes the encoder branch dominant for understanding tasks.", chapter: "transformer-roots", source: 52 },
    { date: "2019", era: "GPT", title: "GPT-2", text: "Zero-shot multitask behavior and staged release make language models socially visible.", chapter: "gpt-2", source: 3 },
    { date: "2020 Jan", era: "Scaling", title: "Scaling laws", text: "Loss improves predictably with compute, data, and parameters across large ranges.", chapter: "gpt-3-scaling", source: 5 },
    { date: "2020 May", era: "Scaling", title: "GPT-3", text: "Few-shot prompting turns context into a task specification interface.", chapter: "gpt-3-scaling", source: 4 },
    { date: "2020", era: "Agents", title: "RAG", text: "Retrieval-augmented generation separates model memory from evidence retrieval.", chapter: "tools-agents", source: 15 },
    { date: "2021", era: "Code", title: "Codex", text: "Code-trained GPT models make executable evaluation central to LLM progress.", chapter: "code-reasoning", source: 9 },
    { date: "2021 Mar", era: "Industry", title: "Stochastic Parrots", text: "A major critique frames scale, data, environmental cost, and social risk as central to language-model history.", chapter: "industrial-stack", source: 57 },
    { date: "2022 Jan", era: "Alignment", title: "InstructGPT", text: "RLHF makes models follow user intent more reliably than raw base models.", chapter: "instruct-chatgpt", source: 7 },
    { date: "2022 Jan", era: "Reasoning", title: "Chain of thought", text: "Intermediate reasoning steps improve hard tasks for sufficiently large models.", chapter: "reasoning-prompts", source: 11 },
    { date: "2022 Mar", era: "Reasoning", title: "Self-consistency and STaR", text: "Sampling multiple reasoning paths and learning from useful rationales become core ideas.", chapter: "reasoning-prompts", source: 12 },
    { date: "2022 Mar", era: "Efficiency", title: "Chinchilla", text: "Compute-optimal training shifts focus from parameter count to data allocation.", chapter: "compute-optimal", source: 6 },
    { date: "2022 Apr", era: "Scaling", title: "PaLM", text: "A 540B dense model pushes few-shot and reasoning performance.", chapter: "compute-optimal", source: 13 },
    { date: "2022 Nov", era: "Alignment", title: "ChatGPT", text: "The chat interface makes instruction-following capability broadly legible.", chapter: "instruct-chatgpt", source: 8 },
    { date: "2022 Nov", era: "Industry", title: "HELM", text: "Holistic evaluation pushes model comparison beyond single-score leaderboard claims.", chapter: "industrial-stack", source: 56 },
    { date: "2022 Dec", era: "Alignment", title: "Constitutional AI", text: "Anthropic explores principle-guided AI feedback for harmlessness training.", chapter: "instruct-chatgpt", source: 18 },
    { date: "2023 Feb", era: "Agents", title: "Toolformer", text: "A model learns when and how to call simple external APIs.", chapter: "tools-agents", source: 17 },
    { date: "2023 Mar", era: "Multimodal", title: "GPT-4", text: "A large closed frontier model raises expectations for reasoning, coding, and multimodal evaluation.", chapter: "multimodal-frontier", source: 14 },
    { date: "2023 May", era: "Reasoning", title: "Tree of Thoughts", text: "Inference becomes a search over candidate reasoning paths.", chapter: "reasoning-prompts", source: 39 },
    { date: "2023 Jul", era: "Open", title: "Llama 2", text: "Open-weight models become a major commercial and research branch.", chapter: "open-weight", source: 19 },
    { date: "2023 Sep", era: "Efficiency", title: "Mistral 7B", text: "Small, efficient open models pressure larger incumbents.", chapter: "compute-optimal", source: 20 },
    { date: "2023 Oct", era: "Industry", title: "SWE-bench", text: "Real GitHub issues become a more realistic test for coding agents.", chapter: "industrial-stack", source: 59 },
    { date: "2023 Dec", era: "Multimodal", title: "Gemini", text: "Google launches a multimodal model family across Ultra, Pro, and Nano tiers.", chapter: "multimodal-frontier", source: 22 },
    { date: "2023 Dec", era: "Open", title: "Mixtral", text: "Sparse mixture-of-experts design becomes a public open-model pattern.", chapter: "compute-optimal", source: 21 },
    { date: "2024 Mar", era: "Multimodal", title: "Claude 3", text: "Anthropic releases Haiku, Sonnet, and Opus tiers.", chapter: "multimodal-frontier", source: 23 },
    { date: "2024 Apr", era: "Open", title: "Llama 3", text: "Meta expands open-weight competition with stronger data and code training.", chapter: "open-weight", source: 25 },
    { date: "2024 May", era: "Multimodal", title: "GPT-4o", text: "Real-time multimodal interaction moves toward voice, vision, and screen workflows.", chapter: "multimodal-frontier", source: 24 },
    { date: "2024 Sep", era: "Reasoning", title: "OpenAI o1", text: "Reasoning time becomes a product-level capability.", chapter: "test-time-compute", source: 26 },
    { date: "2024 Dec", era: "Open", title: "DeepSeek-V3", text: "Sparse MoE economics make large total capacity cheaper to serve.", chapter: "open-weight", source: 27 },
    { date: "2025 Jan", era: "Reasoning", title: "DeepSeek-R1", text: "Open reasoning models intensify competition around RL and distillation.", chapter: "open-weight", source: 28 },
    { date: "2025 Feb", era: "Agents", title: "Claude 3.7 Sonnet", text: "Hybrid reasoning and Claude Code point toward integrated coding agents.", chapter: "tools-agents", source: 29 },
    { date: "2025 Apr", era: "Reasoning", title: "o3 and o4-mini", text: "Reasoning models combine tools, visual input, and complex task execution.", chapter: "test-time-compute", source: 30 },
    { date: "2025 Apr", era: "Agents", title: "GPT-4.1", text: "OpenAI targets coding, instruction following, and long context in the API.", chapter: "tools-agents", source: 31 },
    { date: "2025 Apr", era: "Open", title: "Llama 4", text: "Meta's Llama line moves toward natively multimodal MoE open-weight models.", chapter: "open-weight", source: 53 },
    { date: "2025 Apr", era: "Open", title: "Qwen3", text: "Alibaba's Qwen3 family makes hybrid thinking and non-thinking modes a major open-model pattern.", chapter: "open-weight", source: 54 },
    { date: "2025 Jul", era: "Open", title: "Kimi K2", text: "Moonshot AI releases a large MoE model optimized for agentic workflows.", chapter: "open-weight", source: 55 },
    { date: "2025 Aug", era: "Open", title: "gpt-oss", text: "OpenAI releases open-weight reasoning models after years of closed frontier releases.", chapter: "open-weight", source: 32 },
    { date: "2025 Aug", era: "Reasoning", title: "GPT-5", text: "OpenAI unifies fast response, deeper thinking, and model routing.", chapter: "test-time-compute", source: 33 },
    { date: "2025", era: "Industry", title: "AI Index 2025", text: "AI progress is tracked as an industrial, economic, and policy system, not only a research field.", chapter: "industrial-stack", source: 58 },
    { date: "2026 Feb", era: "Code", title: "GPT-5.3-Codex", text: "OpenAI emphasizes long-running agentic coding workflows.", chapter: "frontier-2026", source: 34 },
    { date: "2026 Feb", era: "Multimodal", title: "Gemini 3.1 Pro", text: "Google advances Gemini reasoning for complex tasks.", chapter: "frontier-2026", source: 35 },
    { date: "2026 Mar", era: "Reasoning", title: "GPT-5.4", text: "GPT-5.4 Thinking and Pro extend the GPT-5 reasoning line.", chapter: "frontier-2026", source: 36 },
    { date: "2026 Apr", era: "Reasoning", title: "Claude Opus 4.7", text: "Anthropic updates its most capable Opus line for coding and complex work.", chapter: "frontier-2026", source: 37 },
    { date: "2026 Apr", era: "Future", title: "GPT-5.5", text: "OpenAI releases GPT-5.5 with emphasis on agentic coding, computer use, knowledge work, and research.", chapter: "frontier-2026", source: 38 }
];

const glossary = [
    ["Base model", "A pretrained model before instruction tuning, RLHF, or task-specific post-training."],
    ["RLHF", "Reinforcement learning from human feedback; a method for steering outputs toward human preferences."],
    ["RLAIF", "Reinforcement learning from AI feedback; used in Constitutional AI to reduce reliance on human labels."],
    ["In-context learning", "Task adaptation through examples or instructions in the prompt without changing weights."],
    ["Chain of thought", "Intermediate reasoning text generated before a final answer."],
    ["Self-consistency", "Sampling multiple reasoning paths and selecting the answer with the strongest agreement."],
    ["Test-time compute", "Extra compute spent during inference for search, reasoning, tool calls, verification, or parallel attempts."],
    ["Process supervision", "Training feedback on intermediate steps rather than only final answers."],
    ["Verifier", "A model, rule, test, or tool that scores or checks candidate outputs."],
    ["RAG", "Retrieval-augmented generation, where retrieved documents ground the model's answer."],
    ["MoE", "Mixture of experts; a model with multiple expert subnetworks, often activating only a subset per token."],
    ["Distillation", "Training a smaller model to imitate the behavior of a larger or stronger model."],
    ["Post-training", "Instruction tuning, RLHF, safety tuning, and other stages after base pretraining."],
    ["Long context", "The ability to condition on very large amounts of input, such as books, repos, or document sets."],
    ["Agent", "A system that uses a model plus tools, state, permissions, and verification to pursue multi-step tasks."],
    ["Tool use", "Model-driven calls to APIs, browsers, code interpreters, search, databases, or computer actions."],
    ["SWE-bench", "A benchmark family for resolving real software issues in GitHub repositories."],
    ["Open weights", "Publicly released model parameters that can be run, fine-tuned, or inspected outside the originating lab."],
    ["Quantization", "Reducing numerical precision so a model can run with less memory and often lower cost."],
    ["Benchmark contamination", "When benchmark examples or near-duplicates appear in training data, making scores less trustworthy."],
    ["Model card", "A document describing intended use, evaluations, limitations, and safety findings for a model."],
    ["Router", "A system component that chooses which model or mode should handle a request."]
];

const lineageNodes = [
    { id: "prelm", label: "Neural LM", family: "roots", x: 0.06, y: 0.48, chapter: "before-transformers", detail: "Embeddings, seq2seq, and attention made language modeling learnable before 2017." },
    { id: "transformer", label: "Transformer", family: "roots", x: 0.17, y: 0.48, chapter: "transformer-roots", detail: "The scalable attention architecture that made modern LLMs practical." },
    { id: "gpt1", label: "GPT-1", family: "openai", x: 0.29, y: 0.24, chapter: "gpt-1", detail: "Generative pretraining plus fine-tuning." },
    { id: "gpt2", label: "GPT-2", family: "openai", x: 0.40, y: 0.22, chapter: "gpt-2", detail: "Zero-shot multitask behavior and staged release." },
    { id: "gpt3", label: "GPT-3", family: "openai", x: 0.52, y: 0.24, chapter: "gpt-3-scaling", detail: "Few-shot prompting and in-context learning at 175B parameters." },
    { id: "codex", label: "Codex", family: "code", x: 0.56, y: 0.41, chapter: "code-reasoning", detail: "Code-trained GPT branch with executable evaluation." },
    { id: "instruct", label: "InstructGPT", family: "align", x: 0.64, y: 0.18, chapter: "instruct-chatgpt", detail: "RLHF makes GPT-3 follow instructions." },
    { id: "chatgpt", label: "ChatGPT", family: "align", x: 0.74, y: 0.19, chapter: "instruct-chatgpt", detail: "The chat interface makes aligned LLMs mainstream." },
    { id: "gpt4", label: "GPT-4", family: "openai", x: 0.84, y: 0.24, chapter: "multimodal-frontier", detail: "A stronger closed frontier with multimodal direction." },
    { id: "gpt5", label: "GPT-5", family: "reason", x: 0.91, y: 0.34, chapter: "test-time-compute", detail: "Unified fast and thinking system." },
    { id: "gpt55", label: "GPT-5.5", family: "reason", x: 0.95, y: 0.45, chapter: "frontier-2026", detail: "Agentic coding, computer use, and knowledge-work frontier in April 2026." },
    { id: "bert", label: "BERT", family: "google", x: 0.31, y: 0.62, chapter: "transformer-roots", detail: "The encoder branch proved self-supervised pretraining for understanding tasks." },
    { id: "palm", label: "PaLM", family: "google", x: 0.56, y: 0.64, chapter: "compute-optimal", detail: "Large-scale dense model and chain-of-thought substrate." },
    { id: "gemini", label: "Gemini", family: "google", x: 0.76, y: 0.62, chapter: "multimodal-frontier", detail: "Google's multimodal model family." },
    { id: "gemini31", label: "Gemini 3.1", family: "google", x: 0.94, y: 0.62, chapter: "frontier-2026", detail: "Google's February 2026 reasoning-focused Pro model." },
    { id: "llama", label: "Llama", family: "open", x: 0.55, y: 0.82, chapter: "open-weight", detail: "Open-weight branch that powered community fine-tuning." },
    { id: "mistral", label: "Mistral", family: "open", x: 0.68, y: 0.87, chapter: "compute-optimal", detail: "Efficient open models and sparse expert designs." },
    { id: "qwen", label: "Qwen", family: "open", x: 0.78, y: 0.78, chapter: "open-weight", detail: "Alibaba's multilingual open and hybrid-reasoning model family." },
    { id: "deepseek", label: "DeepSeek", family: "open", x: 0.88, y: 0.86, chapter: "open-weight", detail: "Open MoE and reasoning branch trained with RL and distilled into smaller models." },
    { id: "kimi", label: "Kimi", family: "open", x: 0.95, y: 0.78, chapter: "open-weight", detail: "Moonshot's open agentic MoE branch." },
    { id: "claude", label: "Claude", family: "anthropic", x: 0.73, y: 0.44, chapter: "multimodal-frontier", detail: "Anthropic's constitutional and long-context assistant line." },
    { id: "opus47", label: "Opus 4.7", family: "anthropic", x: 0.92, y: 0.52, chapter: "frontier-2026", detail: "Anthropic's April 2026 frontier model." }
];

const lineageEdges = [
    ["prelm", "transformer"], ["transformer", "gpt1"], ["gpt1", "gpt2"], ["gpt2", "gpt3"], ["gpt3", "instruct"], ["instruct", "chatgpt"], ["chatgpt", "gpt4"], ["gpt4", "gpt5"], ["gpt5", "gpt55"],
    ["gpt3", "codex"], ["codex", "gpt5"], ["transformer", "bert"], ["bert", "palm"], ["palm", "gemini"], ["gemini", "gemini31"],
    ["gpt3", "llama"], ["llama", "mistral"], ["llama", "qwen"], ["llama", "deepseek"], ["llama", "kimi"], ["instruct", "claude"], ["claude", "opus47"]
];

const families = {
    roots: { label: "Architecture", color: "--gold" },
    openai: { label: "GPT line", color: "--blue" },
    align: { label: "Alignment", color: "--green" },
    code: { label: "Code", color: "--rust" },
    reason: { label: "Reasoning", color: "--violet" },
    google: { label: "Google", color: "--gold" },
    open: { label: "Open weights", color: "--green" },
    anthropic: { label: "Anthropic", color: "--rust" }
};

const state = {
    chapterIndex: 0,
    activeEra: "All",
    selectedNode: null
};

const bookParts = [
    { start: 0, label: "Part I", title: "Foundations" },
    { start: 4, label: "Part II", title: "Scaling and Steering" },
    { start: 8, label: "Part III", title: "Systems and Ecosystems" },
    { start: 12, label: "Part IV", title: "Frontier and Consequences" }
];

const sourceById = new Map(sources.map((source) => [source.id, source]));
const lessonById = new Map(lessons.map((lesson) => [lesson.id, lesson]));

const els = {
    chapterNav: document.getElementById("chapterNav"),
    chapterCount: document.getElementById("chapterCount"),
    eraBadge: document.getElementById("eraBadge"),
    chapterRange: document.getElementById("chapterRange"),
    chapterTags: document.getElementById("chapterTags"),
    chapterKicker: document.getElementById("chapterKicker"),
    runningChapter: document.getElementById("runningChapter"),
    folio: document.getElementById("folio"),
    chapterTitle: document.getElementById("chapterTitle"),
    chapterDek: document.getElementById("chapterDek"),
    chapterContent: document.getElementById("chapterContent"),
    chapterLessons: document.getElementById("chapterLessons"),
    chapterReferences: document.getElementById("chapterReferences"),
    prevChapter: document.getElementById("prevChapter"),
    nextChapter: document.getElementById("nextChapter"),
    timelineList: document.getElementById("timelineList"),
    timelineCount: document.getElementById("timelineCount"),
    eraFilters: document.getElementById("eraFilters"),
    sourcesList: document.getElementById("sourcesList"),
    sourceCount: document.getElementById("sourceCount"),
    glossaryGrid: document.getElementById("glossaryGrid"),
    glossaryCount: document.getElementById("glossaryCount"),
    searchInput: document.getElementById("searchInput"),
    searchResults: document.getElementById("searchResults"),
    lineageCanvas: document.getElementById("lineageCanvas"),
    lineageFallback: document.getElementById("lineageFallback"),
    modelDetail: document.getElementById("modelDetail"),
    resetMap: document.getElementById("resetMap"),
    readingProgress: document.getElementById("readingProgress"),
    themeToggle: document.getElementById("themeToggle")
};

function citeText(text) {
    return text.replace(/\[\[(\d+)]]/g, (_, id) => {
        const source = sourceById.get(Number(id));
        const label = source ? source.title : `Source ${id}`;
        return `<a class="cite" href="#source-${id}" title="${escapeHtml(label)}">[${id}]</a>`;
    });
}

function stripCites(text) {
    return text.replace(/\[\[(\d+)]]/g, " ");
}

function escapeHtml(text) {
    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function slug(text) {
    return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
}

function sourceKind(source) {
    const url = source.url;
    if (url.includes("arxiv.org") || url.includes("jmlr.org") || url.includes("acm.org")) return "Research";
    if (url.includes("github.com")) return "Repository";
    if (url.includes("stanford.edu") || source.title.includes("Report") || source.title.includes("System Card")) return "Report";
    if (url.includes("openai.com") || url.includes("anthropic.com") || url.includes("meta.com") || url.includes("mistral.ai") || url.includes("qwenlm.github.io") || url.includes("google") || url.includes("deepmind")) return "Lab post";
    return "Reference";
}

function sourceYear(source) {
    return source.detail.match(/\b(19|20)\d{2}\b/)?.[0] || "n.d.";
}

function partForChapter(index) {
    return bookParts.reduce((current, part) => index >= part.start ? part : current, bookParts[0]);
}

function renderChapterNav() {
    els.chapterCount.textContent = `${chapters.length} chapters`;
    let previousPart = "";
    els.chapterNav.innerHTML = chapters.map((chapter, index) => {
        const part = partForChapter(index);
        const partHeading = part.label !== previousPart
            ? `<div class="part-heading">${escapeHtml(part.label)} / ${escapeHtml(part.title)}</div>`
            : "";
        previousPart = part.label;
        return `
            ${partHeading}
            <button class="chapter-link ${index === state.chapterIndex ? "active" : ""}" type="button" data-index="${index}" ${index === state.chapterIndex ? 'aria-current="page"' : ""}>
                <strong>${index + 1}. ${escapeHtml(chapter.title)}</strong>
                <span>${escapeHtml(chapter.range)} / ${escapeHtml(chapter.era)}</span>
            </button>
        `;
    }).join("");
}

function renderChapter() {
    const chapter = chapters[state.chapterIndex];
    const part = partForChapter(state.chapterIndex);
    document.title = `${chapter.title} | The LLM History Book`;
    els.runningChapter.textContent = "The LLM History Book";
    els.folio.textContent = `Chapter ${state.chapterIndex + 1} of ${chapters.length}`;
    els.chapterKicker.textContent = `${part.label}: ${part.title} / Chapter ${state.chapterIndex + 1}`;
    els.eraBadge.textContent = chapter.era;
    els.chapterRange.textContent = chapter.range;
    els.chapterTags.textContent = chapter.tags.join(" / ");
    els.chapterTitle.textContent = chapter.title;
    els.chapterDek.textContent = chapter.dek;
    els.prevChapter.disabled = state.chapterIndex === 0;
    els.nextChapter.disabled = state.chapterIndex === chapters.length - 1;

    els.chapterContent.innerHTML = chapter.sections.map((section, sectionIndex) => {
        const sectionId = `${chapter.id}-${slug(section.heading)}`;
        const paragraphs = section.paragraphs.map((paragraph) => `<p>${citeText(paragraph)}</p>`).join("");
        const bullets = section.bullets ? `<ul>${section.bullets.map((item) => `<li>${citeText(item)}</li>`).join("")}</ul>` : "";
        const callout = section.callout ? `<div class="callout">${citeText(section.callout)}</div>` : "";
        return `
            <section id="${sectionId}">
                <h3>${sectionIndex + 1}. ${escapeHtml(section.heading)}</h3>
                ${paragraphs}
                ${bullets}
                ${callout}
            </section>
        `;
    }).join("");

    decorateGlossary(els.chapterContent);

    els.chapterLessons.innerHTML = chapter.lessonIds.map((id) => {
        const lesson = lessonById.get(id);
        return `<div class="lesson-card"><strong>${escapeHtml(lesson.title)}</strong><span>${escapeHtml(lesson.text)}</span></div>`;
    }).join("");

    els.chapterReferences.innerHTML = chapter.sourceIds.map((id) => {
        const source = sourceById.get(id);
        if (!source) return "";
        return `
            <a class="chapter-reference" href="${source.url}" target="_blank" rel="noreferrer">
                <span>${source.id}</span>
                <strong>${escapeHtml(source.title)}</strong>
                <small>${escapeHtml(source.detail)}</small>
            </a>
        `;
    }).join("");

    renderChapterNav();
    drawLineage();
    if (window.lucide) window.lucide.createIcons();
}

function setChapter(index, push = true) {
    state.chapterIndex = Math.max(0, Math.min(index, chapters.length - 1));
    renderChapter();
    if (push) history.replaceState(null, "", `#${chapters[state.chapterIndex].id}`);
    const reader = document.getElementById("reader");
    if (reader) reader.scrollIntoView({ block: "start" });
}

function renderTimeline() {
    const eras = ["All", ...Array.from(new Set(timeline.map((event) => event.era)))];
    els.eraFilters.innerHTML = eras.map((era) => `
        <button class="era-filter ${state.activeEra === era ? "active" : ""}" type="button" data-era="${escapeHtml(era)}" aria-pressed="${state.activeEra === era}">${escapeHtml(era)}</button>
    `).join("");

    const visible = state.activeEra === "All" ? timeline : timeline.filter((event) => event.era === state.activeEra);
    els.timelineCount.textContent = `${visible.length} events`;
    els.timelineList.innerHTML = visible.map((event) => {
        const source = sourceById.get(event.source);
        return `
            <div class="timeline-event">
                <button class="timeline-jump" type="button" data-chapter="${escapeHtml(event.chapter)}">
                    <span>${escapeHtml(event.date)} / ${escapeHtml(event.era)}</span>
                    <strong>${escapeHtml(event.title)}</strong>
                    <p>${escapeHtml(event.text)}</p>
                </button>
                ${source ? `<a class="timeline-source" href="${source.url}" target="_blank" rel="noreferrer">Source: ${escapeHtml(source.title)}</a>` : ""}
            </div>
        `;
    }).join("");
}

function renderSources() {
    els.sourceCount.textContent = `${sources.length} references`;
    els.sourcesList.innerHTML = sources.map((source) => `
        <div class="source-row" id="source-${source.id}">
            <span>${source.id}</span>
            <div>
                <a href="${source.url}" target="_blank" rel="noreferrer">${escapeHtml(source.title)}</a>
                <small>${escapeHtml(source.detail)}</small>
                <span class="source-meta">${escapeHtml(sourceKind(source))} / ${escapeHtml(sourceYear(source))}</span>
                <span class="source-usage">Used in: ${chapters
                    .map((chapter, index) => chapter.sourceIds.includes(source.id) ? `<a href="#${chapter.id}" data-index="${index}">Ch. ${index + 1}</a>` : "")
                    .filter(Boolean)
                    .join(", ") || "timeline only"}</span>
            </div>
        </div>
    `).join("");
}

function renderLineageFallback() {
    els.lineageFallback.innerHTML = `
        <ul class="lineage-fallback-list">
            ${lineageNodes.map((node) => {
                const index = chapters.findIndex((chapter) => chapter.id === node.chapter);
                return `<li><a href="#${node.chapter}" data-index="${index}">${escapeHtml(node.label)}</a> - ${escapeHtml(node.detail)}</li>`;
            }).join("")}
        </ul>
    `;
}

function renderGlossary() {
    els.glossaryCount.textContent = `${glossary.length} terms`;
    els.glossaryGrid.innerHTML = glossary.map(([term, definition]) => `
        <div class="glossary-term">
            <strong>${escapeHtml(term)}</strong>
            <p>${escapeHtml(definition)}</p>
        </div>
    `).join("");
}

function getCssColor(name) {
    return getComputedStyle(document.body).getPropertyValue(name).trim()
        || getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawLineage() {
    const canvas = els.lineageCanvas;
    const rect = canvas.getBoundingClientRect();
    if (rect.width === 0) return; // canvas inside a closed drawer; redraw on open
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    canvas.height = Math.max(1, Math.floor(rect.height * ratio));

    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const colors = Object.fromEntries(Object.entries(families).map(([key, family]) => [key, getCssColor(family.color)]));
    const muted = getCssColor("--muted");
    const rail = getCssColor("--rail-strong");
    const paper = getCssColor("--paper");
    const ink = getCssColor("--ink");

    ctx.fillStyle = paper;
    ctx.fillRect(0, 0, rect.width, rect.height);

    const position = (node) => ({
        x: node.x * rect.width,
        y: node.y * rect.height
    });

    ctx.lineWidth = 1.4;
    ctx.strokeStyle = rail;
    lineageEdges.forEach(([from, to]) => {
        const a = position(lineageNodes.find((node) => node.id === from));
        const b = position(lineageNodes.find((node) => node.id === to));
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        const midX = (a.x + b.x) / 2;
        ctx.bezierCurveTo(midX, a.y, midX, b.y, b.x, b.y);
        ctx.stroke();
    });

    lineageNodes.forEach((node) => {
        const { x, y } = position(node);
        const selected = state.selectedNode === node.id;
        ctx.beginPath();
        ctx.fillStyle = colors[node.family] || muted;
        ctx.strokeStyle = selected ? ink : paper;
        ctx.lineWidth = selected ? 4 : 2;
        ctx.arc(x, y, selected ? 9 : 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.font = "500 10.5px Geist, ui-sans-serif, system-ui, sans-serif";
        ctx.fillStyle = ink;
        ctx.textAlign = "center";
        ctx.fillText(node.label, x, y - 14);
    });

    const selected = lineageNodes.find((node) => node.id === state.selectedNode) || lineageNodes.find((node) => node.id === "transformer");
    els.modelDetail.innerHTML = `
        <strong>${escapeHtml(selected.label)}</strong>
        <span>${escapeHtml(selected.detail)}</span>
    `;
}

function handleCanvasClick(event) {
    const rect = els.lineageCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const hit = lineageNodes.find((node) => {
        const dx = node.x * rect.width - x;
        const dy = node.y * rect.height - y;
        return Math.sqrt(dx * dx + dy * dy) < 18;
    });

    if (hit) {
        state.selectedNode = hit.id;
        const index = chapters.findIndex((chapter) => chapter.id === hit.chapter);
        if (index >= 0) setChapter(index);
        drawLineage();
    }
}

function buildSearchIndex() {
    const chapterEntries = chapters.flatMap((chapter, chapterIndex) => {
        const chapterEntry = {
            type: "Chapter",
            label: chapter.title,
            body: [chapter.dek, chapter.range, chapter.era, chapter.tags.join(" ")].join(" "),
            chapterIndex
        };
        const sectionEntries = chapter.sections.map((section) => ({
            type: "Section",
            label: `${chapter.title}: ${section.heading}`,
            body: section.paragraphs.map(stripCites).join(" "),
            chapterIndex
        }));
        return [chapterEntry, ...sectionEntries];
    });

    const timelineEntries = timeline.map((event) => ({
        type: "Timeline",
        label: event.title,
        body: `${event.date} ${event.era} ${event.text}`,
        chapterIndex: chapters.findIndex((chapter) => chapter.id === event.chapter)
    }));

    const lessonEntries = lessons.map((lesson) => ({
        type: "Lesson",
        label: lesson.title,
        body: lesson.text,
        chapterIndex: chapters.findIndex((chapter) => chapter.lessonIds.includes(lesson.id))
    }));

    return [...chapterEntries, ...timelineEntries, ...lessonEntries];
}

const searchIndex = buildSearchIndex();

function renderSearchResults(query) {
    const normalized = query.trim().toLowerCase();
    if (!normalized) {
        els.searchResults.hidden = true;
        els.searchResults.innerHTML = "";
        return;
    }

    const terms = normalized.split(/\s+/).filter(Boolean);
    const results = searchIndex
        .map((entry) => {
            const label = entry.label.toLowerCase();
            const body = entry.body.toLowerCase();
            const haystack = `${label} ${body}`;
            let score = haystack.includes(normalized) ? 8 : 0;
            if (label.includes(normalized)) score += 20;
            const matchedTerms = terms.filter((term) => haystack.includes(term));
            score += matchedTerms.length;
            score += terms.filter((term) => label.includes(term)).length * 3;
            if (matchedTerms.length === terms.length) score += 6;
            return { ...entry, score };
        })
        .filter((entry) => entry.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 9);

    els.searchResults.hidden = false;
    if (!results.length) {
        els.searchResults.innerHTML = `<div class="search-result"><strong>No matches</strong><span>${escapeHtml(query)}</span></div>`;
        return;
    }

    els.searchResults.innerHTML = results.map((entry) => `
        <button class="search-result" type="button" data-index="${entry.chapterIndex}">
            <span>${escapeHtml(entry.type)}</span>
            <strong>${escapeHtml(entry.label)}</strong>
        </button>
    `).join("");
}

function updateProgress() {
    const height = document.documentElement.scrollHeight - window.innerHeight;
    const pct = height > 0 ? (window.scrollY / height) * 100 : 0;
    els.readingProgress.style.width = `${Math.min(100, Math.max(0, pct))}%`;
}

function applyInitialTheme() {
    const saved = localStorage.getItem("llm-history-theme");
    if (saved === "dark") document.body.classList.add("dark");
    updateThemeToggleLabel();
}

function updateThemeToggleLabel() {
    els.themeToggle.setAttribute("aria-label", document.body.classList.contains("dark") ? "Use light theme" : "Use dark theme");
}

function toggleTheme() {
    document.body.classList.toggle("dark");
    localStorage.setItem("llm-history-theme", document.body.classList.contains("dark") ? "dark" : "light");
    updateThemeToggleLabel();
    drawLineage();
}

/* --- Inline glossary decoration ----------------------------------------- */

let glossRegex = null;
let glossMap = null;

function buildGlossIndex() {
    const terms = glossary.map(([term]) => term).sort((a, b) => b.length - a.length);
    const escaped = terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    glossRegex = new RegExp(`\\b(${escaped.join("|")})\\b`, "i");
    glossMap = new Map(glossary.map(([term, definition]) => [term.toLowerCase(), definition]));
}

function decorateGlossary(rootEl) {
    if (!rootEl) return;
    if (!glossRegex) buildGlossIndex();
    const seen = new Set();
    const skipParents = new Set(["A", "SCRIPT", "STYLE", "CODE"]);
    const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
            if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
            let parent = node.parentNode;
            while (parent && parent !== rootEl) {
                if (skipParents.has(parent.nodeName)) return NodeFilter.FILTER_REJECT;
                if (parent.classList && (parent.classList.contains("cite") || parent.classList.contains("gloss"))) {
                    return NodeFilter.FILTER_REJECT;
                }
                parent = parent.parentNode;
            }
            return NodeFilter.FILTER_ACCEPT;
        }
    });

    const targets = [];
    let node;
    while ((node = walker.nextNode())) targets.push(node);

    for (const textNode of targets) {
        const text = textNode.nodeValue;
        const match = text.match(glossRegex);
        if (!match) continue;
        const term = match[1];
        const key = term.toLowerCase();
        if (seen.has(key)) continue;
        const definition = glossMap.get(key);
        if (!definition) continue;
        seen.add(key);

        const before = text.slice(0, match.index);
        const after = text.slice(match.index + term.length);
        const frag = document.createDocumentFragment();
        if (before) frag.appendChild(document.createTextNode(before));

        const span = document.createElement("span");
        span.className = "gloss";
        span.tabIndex = 0;
        span.textContent = term;

        const pop = document.createElement("span");
        pop.className = "gloss-pop";
        pop.textContent = definition;
        // Right-align popover when the term is in the right half of the column,
        // so it doesn't clip beyond the prose width.
        if (textNode.parentNode && textNode.parentNode.getBoundingClientRect) {
            const parentRect = textNode.parentNode.getBoundingClientRect();
            const range = document.createRange();
            range.setStart(textNode, match.index);
            range.setEnd(textNode, match.index + term.length);
            const termRect = range.getBoundingClientRect();
            range.detach && range.detach();
            if (parentRect.width && termRect.left - parentRect.left > parentRect.width * 0.55) {
                pop.dataset.popAlign = "right";
            }
        }

        span.appendChild(pop);
        frag.appendChild(span);
        if (after) frag.appendChild(document.createTextNode(after));
        textNode.parentNode.replaceChild(frag, textNode);
    }
}

/* --- Surface controller (drawers + search overlay) --------------------- */

const surfaces = {
    contents: { el: null, opener: null, returnTo: null },
    atlas:    { el: null, opener: null, returnTo: null },
    search:   { el: null, opener: null, returnTo: null }
};
let openSurfaceId = null;

function openSurface(id) {
    if (openSurfaceId === id) return;
    if (openSurfaceId) closeSurface();
    const surface = surfaces[id];
    if (!surface || !surface.el) return;

    surface.returnTo = document.activeElement;
    surface.el.setAttribute("data-open", "");
    surface.el.setAttribute("aria-hidden", "false");

    const backdrop = document.getElementById("surfaceBackdrop");
    if (backdrop) {
        backdrop.hidden = false;
        requestAnimationFrame(() => backdrop.setAttribute("data-open", ""));
    }

    document.body.classList.add("surface-open");
    if (surface.opener) surface.opener.setAttribute("aria-expanded", "true");
    openSurfaceId = id;

    if (id === "atlas") {
        // Canvas was 0×0 while drawer hidden; defer one frame for layout, then draw.
        requestAnimationFrame(() => {
            requestAnimationFrame(drawLineage);
        });
    }

    if (id === "search") {
        setTimeout(() => els.searchInput && els.searchInput.focus(), 30);
    } else {
        const focusable = surface.el.querySelector("button, a, [tabindex]:not([tabindex='-1'])");
        if (focusable) focusable.focus({ preventScroll: true });
    }
}

function closeSurface() {
    if (!openSurfaceId) return;
    const surface = surfaces[openSurfaceId];
    surface.el.removeAttribute("data-open");
    surface.el.setAttribute("aria-hidden", "true");
    if (surface.opener) surface.opener.setAttribute("aria-expanded", "false");

    const backdrop = document.getElementById("surfaceBackdrop");
    if (backdrop) {
        backdrop.removeAttribute("data-open");
        setTimeout(() => { backdrop.hidden = true; }, 240);
    }

    document.body.classList.remove("surface-open");
    if (surface.returnTo && typeof surface.returnTo.focus === "function") {
        surface.returnTo.focus({ preventScroll: true });
    }
    openSurfaceId = null;
}

function setupSurfaces() {
    surfaces.contents.el = document.getElementById("contentsDrawer");
    surfaces.atlas.el = document.getElementById("atlasDrawer");
    surfaces.search.el = document.getElementById("searchOverlay");
    surfaces.contents.opener = document.getElementById("openContents");
    surfaces.atlas.opener = document.getElementById("openAtlas");
    surfaces.search.opener = document.getElementById("openSearch");

    if (surfaces.contents.opener) {
        surfaces.contents.opener.addEventListener("click", () => openSurface("contents"));
    }
    if (surfaces.atlas.opener) {
        surfaces.atlas.opener.addEventListener("click", () => openSurface("atlas"));
    }
    if (surfaces.search.opener) {
        surfaces.search.opener.addEventListener("click", () => openSurface("search"));
    }

    const backdrop = document.getElementById("surfaceBackdrop");
    if (backdrop) backdrop.addEventListener("click", closeSurface);

    document.querySelectorAll("[data-close]").forEach((btn) => {
        btn.addEventListener("click", closeSurface);
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && openSurfaceId) closeSurface();
    });

    // Auto-close after navigation actions inside a drawer/overlay.
    if (els.chapterNav) {
        els.chapterNav.addEventListener("click", (event) => {
            if (event.target.closest("[data-index]")) closeSurface();
        });
    }
    if (els.searchResults) {
        els.searchResults.addEventListener("click", (event) => {
            if (event.target.closest("[data-index]")) closeSurface();
        });
    }
    if (els.timelineList) {
        els.timelineList.addEventListener("click", (event) => {
            if (event.target.closest("[data-chapter]")) closeSurface();
        });
    }
    if (els.lineageFallback) {
        els.lineageFallback.addEventListener("click", (event) => {
            if (event.target.closest("[data-index]")) closeSurface();
        });
    }
}

function bindEvents() {
    els.chapterNav.addEventListener("click", (event) => {
        const button = event.target.closest("[data-index]");
        if (!button) return;
        setChapter(Number(button.dataset.index));
    });

    els.prevChapter.addEventListener("click", () => setChapter(state.chapterIndex - 1));
    els.nextChapter.addEventListener("click", () => setChapter(state.chapterIndex + 1));

    els.eraFilters.addEventListener("click", (event) => {
        const button = event.target.closest("[data-era]");
        if (!button) return;
        state.activeEra = button.dataset.era;
        renderTimeline();
    });

    els.timelineList.addEventListener("click", (event) => {
        const button = event.target.closest("[data-chapter]");
        if (!button) return;
        const index = chapters.findIndex((chapter) => chapter.id === button.dataset.chapter);
        if (index >= 0) setChapter(index);
    });

    els.sourcesList.addEventListener("click", (event) => {
        const link = event.target.closest("[data-index]");
        if (!link) return;
        event.preventDefault();
        setChapter(Number(link.dataset.index));
    });

    els.lineageFallback.addEventListener("click", (event) => {
        const link = event.target.closest("[data-index]");
        if (!link) return;
        event.preventDefault();
        setChapter(Number(link.dataset.index));
    });

    els.searchInput.addEventListener("input", (event) => renderSearchResults(event.target.value));
    els.searchResults.addEventListener("click", (event) => {
        const button = event.target.closest("[data-index]");
        if (!button) return;
        setChapter(Number(button.dataset.index));
        els.searchInput.value = "";
        renderSearchResults("");
    });

    els.lineageCanvas.addEventListener("click", handleCanvasClick);
    els.resetMap.addEventListener("click", () => {
        state.selectedNode = null;
        drawLineage();
    });

    els.themeToggle.addEventListener("click", toggleTheme);
    window.addEventListener("resize", drawLineage);
    window.addEventListener("scroll", updateProgress, { passive: true });
}

function initFromHash() {
    const hash = window.location.hash.replace("#", "");
    if (!hash) return;
    const index = chapters.findIndex((chapter) => chapter.id === hash);
    if (index >= 0) state.chapterIndex = index;
}

function init() {
    applyInitialTheme();
    initFromHash();
    renderChapterNav();
    renderChapter();
    renderTimeline();
    renderSources();
    renderGlossary();
    renderLineageFallback();
    bindEvents();
    setupSurfaces();
    updateProgress();
    if (window.lucide) window.lucide.createIcons();
}

init();
