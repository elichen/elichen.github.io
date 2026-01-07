# Context Management

After fifteen years in engineering management, I've discovered that the principles I learned managing humans at Netflix apply remarkably well to managing AI agents. The philosophy that made Netflix engineering culture influential, "context, not control," turns out to be exactly what's needed to unlock massive productivity gains from frontier AI models.

The goal here is simple: extract maximum productivity from AI. If you want maximum control, you can always write the code yourself. But if you want to compress weeks of work into hours, you need to master context management for these alien intelligences that think differently than we do.

Netflix had a unique approach to engineering management. We hired the best people we could find, then asked ourselves a simple question: how do we create the most effective environment for technical folks to do their best work?

The answer surprised many traditional managers. Technical work is fundamentally creative work, and creativity thrives with freedom. The best environment for innovation wasn't one with detailed instructions and constant oversight; it was one where talented people had the context to make good decisions and the freedom to execute them.

"Context, not control" became our mantra. Provide engineers with rich understanding of the business goals, technical constraints, and user needs. Then get out of their way. The person closest to the problem, armed with proper context, makes better decisions than distant managers working through layers of abstraction.

Today, I'm seeing these same patterns emerge in AI-assisted coding. Frontier AI models deliver huge productivity gains. I've watched project timelines compress from weeks to days to hours. But these models are costly, and getting value from them requires mastering context management all over again.

Just like those Netflix engineers, AI agents do their best work when given freedom within well-defined context. The challenge isn't controlling every line of code they write; it's providing the right context so they make good decisions autonomously.

But here's where AI differs from humans: these systems have alien limitations. They operate within fixed context windows; they can only "see" a certain amount of information at once. They don't have our intuitive understanding of what matters. They need explicit context about things humans would naturally infer. Understanding these constraints is key to productive AI collaboration.

Good context management starts with understanding what context actually means for AI. You need to provide enough information for good decisions, but not so much that you're drowning the model in irrelevant details. It's a delicate balance of "no more, no less."

Sometimes it's more efficient to use dialogue for context alignment rather than typing everything out. You can leverage the AI's existing knowledge, for example, asking it to explain critical components back to you as a trust-but-verify mechanism. This interactive approach often surfaces assumptions and gaps that static documentation misses.

To gain real productivity from context alignment, you must be detailed in your context review. Vague context leads to vague implementations.

The real leverage comes from translating business requirements into concrete engineering decisions. Instead of saying "this needs to scale," you provide engineering choices that achieve scaling goals. Is this a weekend project or an enterprise deployment? What's the maximum number of users? How will traffic spike? These questions translate into concrete decisions: monolithic versus distributed architecture, specific test coverage goals, failsafe mechanisms, database choices based on consistency versus availability tradeoffs.

Good AI agents, like good engineers, excel at "filling in the white space." They can infer UX patterns, handle edge cases, suggest architecture improvements, and anticipate scaling challenges. But they can only do this well when the context clearly defines the boundaries of that white space.

The best agents, whether AI or human, can execute from initial context to completed implementation in a single shot when properly aligned. This single-shot execution is the goal of productivity: set context once, get working implementation without further intervention.

But even great alignments sometimes hide issues that only become apparent during implementation. For larger projects, setting alignment milestones helps catch surprises early and provides natural realignment points. Backend deployed with working API tests before frontend development. RBAC features complete before building the full authentication flow. Mock UI working before implementing the data model. These milestones validate that context and implementation remain aligned as complexity grows.

What separates good AI agents from great ones? The error rate. This distinction matters more than most people realize because errors compound exponentially on long-horizon tasks.

Consider the math: if an agent has a 10% error rate per step, after 10 steps you have only a 35% chance of success. Halve that error rate to 5%, and success probability jumps to 60%. This is why halving the error rate effectively 10x's the complexity of tasks an agent can handle reliably.

This mathematical reality drives a key decision: invest in the AI with the lowest error rate you can afford. The productivity gains from a more capable model far outweigh the additional cost when working on complex, multi-step implementations.

Micromanaging becomes necessary when you don't know an agent's error rate or the quality of your context alignment. Watching the implementation unfold in real-time prevents costly mistakes.

I've caught many off-the-rails failures by simply staying engaged with the output instead of context-switching to other work. This vigilance is particularly important when working with a new AI model, implementing critical system components, dealing with complex architectural decisions, or navigating ambiguous requirements.

Think of micromanagement as calibration. You're learning the agent's capabilities and adjusting context provision accordingly.

One unique advantage of AI-assisted development is the ability to treat failed attempts as learning experiences with minimal cost. If a project goes off track, you always have an option that wasn't practical with human developers: `git restore` and start over.

You're simulating best-of-N sampling, a proven ML technique where you generate multiple attempts and pick the best one. Each fresh start is another sample, another chance at a better outcome. And you're honing your context provision skills. Each attempt refines your understanding of what context the AI needs. The second attempt incorporates lessons from the first failure. The third attempt might nail it perfectly. The ability to reset and retry with improved context turns every mistake into a learning opportunity.

Effective AI-assisted development requires deliberate practice. Pick low-error AI: invest in the best model you can afford, because the productivity multiplier from lower error rates justifies the cost for any serious development work. Master context management: learn what context matters and how to provide it efficiently. Calibrate oversight: know when to let AI run autonomously and when to watch closely.

Context management is becoming the key differentiator in software development productivity. Engineers who master this skill are seeing 10x, sometimes 100x improvements in their output. They're building systems that would have taken teams months with just one person in days.

The lesson from Netflix applies directly: provide rich context, establish clear boundaries, then let intelligence, whether human or artificial, execute. The difference now is that AI can work 24/7, doesn't get tired, and can hold massive codebases in its context window.

But remember: AI thinks differently than we do. It has alien strengths and alien weaknesses. Master the art of context management, and you master the ability to direct these powerful but strange intelligences toward productive ends.