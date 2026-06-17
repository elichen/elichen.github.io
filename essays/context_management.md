# Context Management

After fifteen years in engineering management, I've come to realize that the principles I learned managing humans at Netflix apply remarkably well to managing AI agents. The philosophy at the heart of Netflix's engineering culture, "context, not control," turns out to be almost exactly what you need to pull large productivity gains out of frontier AI models.

The goal here is to extract the maximum productivity from AI, and it's worth being clear about that up front, because if what you want is maximum control, you can always just write the code yourself. But if what you want is to compress weeks of work into hours, then you have to get good at managing context for these alien intelligences that think so differently from us.

Netflix had an unusual approach to engineering management. We hired the best people we could find, and then we asked ourselves a pretty simple question, which was how do we build the most effective environment for technical people to do their best work.

The answer tends to surprise managers who are used to detailed instructions and close oversight, because technical work is creative work, and creativity thrives on freedom. The best environment for innovation turned out to be one where talented people had the context to make good decisions and the freedom to actually act on them.

"Context, not control" became the mantra. You give engineers a rich understanding of the business goals, the technical constraints, and the user needs, and then you get out of their way. The person closest to the problem, armed with the right context, will make better decisions than a distant manager reaching down through layers of abstraction.

And now I'm watching these same patterns show up again in AI-assisted coding. Frontier models deliver real, large productivity gains, and I've watched project timelines fall from weeks to days to hours. But these models are costly to run, and getting value out of them means learning context management all over again.

Just like those Netflix engineers, AI agents do their best work when you give them freedom inside a well-defined context. The whole challenge is providing the right context so that they can make good decisions on their own.

But here's where AI really differs from a human, which is that these systems have alien limitations. They work inside fixed context windows, so they can only "see" a certain amount of information at any one time. They don't carry our intuitive sense of what matters, so they need things spelled out that a person would simply infer. Understanding those constraints is the key to working with them productively.

Good context management starts with understanding what context even means for an AI, which is providing enough information for it to make good decisions without drowning it in detail that doesn't matter.

Sometimes the most efficient path is dialogue rather than typing everything out in advance. You can lean on the model's existing knowledge, for instance by asking it to explain the key components back to you as a kind of trust-but-verify check, and that back-and-forth surfaces the assumptions and gaps that static documentation tends to hide.

But to actually get productivity out of this alignment, you have to be detailed when you review the context, because vague context leads straight to vague implementations.

The real payoff comes from translating business requirements into concrete engineering decisions. Instead of saying "this needs to scale," you give the actual choices that achieve scaling. Is this a weekend project or an enterprise deployment? What's the maximum number of users? How is traffic going to spike? Those questions resolve into concrete decisions, monolithic or distributed architecture, specific test coverage goals, failsafe mechanisms, a database chosen on the consistency-versus-availability tradeoff.

Good agents, like good engineers, are excellent at filling in the white space. They'll infer UX patterns, handle edge cases, suggest architectural improvements, and anticipate scaling challenges. But they can only do that well when the context clearly draws the boundaries of where that white space is.

The best agents, whether human or AI, can run from the initial context all the way to a finished implementation in a single shot when they're properly aligned, and that single-shot execution is really the goal, setting the context once and getting a working implementation back without having to intervene again.

Even so, even a great alignment can hide issues that only surface during implementation, so for larger projects it helps to set alignment milestones that catch surprises early and give you natural points to realign. Backend deployed with working API tests before the frontend starts. RBAC features done before the full authentication flow goes in. A mock UI working before the data model gets built. Each of those checkpoints confirms that the context and the implementation are still in step as complexity grows.

So what separates a good agent from a great one? It comes down to the error rate, and this matters far more than people tend to realize, because errors compound exponentially on long-horizon tasks.

Look at the math for a second. If an agent has a 10% error rate per step, then after ten steps you're down to about a 35% chance of success. Halve that error rate to 5%, and your success probability jumps to 60%. That's why cutting the error rate in half effectively 10x's the complexity of the tasks an agent can handle reliably.

And that reality drives a clear decision, which is to invest in the agent with the lowest error rate you can afford, because on complex multi-step work the productivity from a more capable model easily outweighs the extra cost.

Micromanaging becomes necessary when you don't yet know an agent's error rate or how good your context alignment really is. Watching the implementation unfold in real time is how you catch the costly mistakes before they spread.

I've caught a lot of off-the-rails failures just by staying engaged with the output instead of context-switching away to something else. That kind of vigilance matters most when you're working with a new model, on core system components, on hard architectural decisions, or anywhere the requirements are still ambiguous.

It helps to think of micromanagement as calibration. You're learning what the agent is capable of and adjusting how much context you give it accordingly.

One genuinely unique advantage of AI-assisted development is that you can treat a failed attempt as a cheap learning experience. If a project goes off the rails, you have an option you never had with a human developer, which is to `git restore` and simply start over.

What you're really doing there is simulating best-of-N sampling, the ML technique where you generate many attempts and keep the best one. Each fresh start is another sample, another shot at a better outcome, and each one also sharpens your own context-provision skills. The second attempt carries the lessons from the first failure, the third might land it perfectly, and the ability to reset and retry with better context turns every mistake into something you learn from.

Effective AI-assisted development really comes down to deliberate practice. Pick low-error AI, investing in the best model you can afford, because the multiplier from a lower error rate justifies the cost on any serious work. Master context management, learning what context matters and how to deliver it efficiently. And calibrate your oversight, knowing when to let the agent run on its own and when to watch it closely.

Engineers are reporting 10x and sometimes 100x improvements in their output, building systems that would once have taken a team months, now done by one person in days.

The lesson from Netflix carries straight over. Provide rich context, set clear boundaries, and then let intelligence, human or artificial, execute. The only real difference now is that the AI can work around the clock, never gets tired, and can hold an entire codebase inside its context window.

AI thinks differently than we do, with alien strengths and alien weaknesses, and once you master context management, you can point these strange intelligences toward genuinely productive ends.
