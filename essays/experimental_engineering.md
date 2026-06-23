# Experimental Engineering

Software engineering used to be a discipline of careful planning. You studied the problem, debated the approaches in design reviews, wrote PRDs to assess feasibility, and only then committed to a path, because the cost of being wrong was weeks or months of wasted implementation. So we built up these elaborate processes to keep ourselves from taking a wrong turn, the architecture reviews, the proof-of-concept phases, the technical feasibility assessments, and all of it was really just risk mitigation for the high cost of writing code.

And that cost just collapsed.

I needed to extract images from PDFs recently. Before this shift, I would have gone and researched the libraries, read through the documentation and the Stack Overflow threads, picked the most promising option, implemented it, and then discovered its limitations the hard way, and then I'd have repeated the whole cycle with a second library. That process might have taken a full day, maybe two.

Instead, I just described the problem to an LLM and had it implement five different extraction methods at once, which gave me five working scripts and five sets of output. And I looked at the results, not the code. One method handled embedded images well but missed the vector graphics. Another captured everything but came out blurry. A third just got it right. The total time from problem to validated solution was minutes, and I deleted the code for the four approaches that didn't work.

Right now I'm trying to train a neural network to play a perfect game of Snake with reinforcement learning, and I'm experimenting with different input representations, neural architectures, reward functions, and RL algorithms. In the old world, each one of those would have been a real commitment. You'd read the papers, pick an approach, spend days implementing it, and then find out it didn't converge. Before I had these tools, I once spent two months just reproducing someone else's mechanistic interpretability visualization from scratch, two months of setup before I could even begin to experiment with my own ideas.

Now I can run parallel experiments across multiple dimensions of the problem space at once, and the experiments that fail cost me almost nothing.

And this works even when the LLM has never seen your specific combination of tools. There's no training data for "this particular RL algorithm applied to Snake with this specific neural architecture and reward shaping." The model is just competent enough at each individual piece that you can compose novel combinations and quickly find out whether they work, which means you're searching the solution space faster than any one person ever could.

It's worth remembering how engineering teams used to validate a technical approach. Design documents reviewed by committee. Senior engineers weighing in based on past experience. Gut feel about which database or framework would hold up best. Sometimes those instincts were right, and often they weren't, and either way the team wouldn't find out for weeks.

The instinct to plan carefully before writing code used to be a virtue, and it was, because it was the correct adaptation to a world where writing code was expensive. But that world doesn't exist anymore. The engineers spending a week debating the right approach in a doc could have validated every candidate approach in a single afternoon.

And this changes your whole relationship with code. Exploration code becomes disposable. When the cost of generating it approaches zero, you generate a lot and you keep very little. Write five implementations and keep one. Or keep none and try five more. The old instinct to carefully craft and preserve every line you write is now the thing slowing you down.

Today the real binding constraint is token generation speed and the latency between the AI and your development environment. Faster inference and shorter round-trips mean more experiments per hour, and more experiments per hour means you land on better solutions sooner.

And that gap, the one between what you can imagine and what you can actually test, keeps getting narrower.
