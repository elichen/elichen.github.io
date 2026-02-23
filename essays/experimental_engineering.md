# Software Engineering Is Now an Experimental Science

Software engineering used to be a discipline of careful planning. You studied the problem, debated approaches in design reviews, wrote PRDs to assess feasibility, and then committed to a path. The cost of being wrong was weeks or months of wasted implementation. So we built elaborate processes to avoid wrong turns: architecture reviews, proof-of-concept phases, technical feasibility assessments. All of it was risk mitigation for the high cost of writing code.

That cost just collapsed.

I recently needed to extract images from PDFs. Before this shift, I would have researched libraries, read documentation and Stack Overflow threads, picked the most promising approach, implemented it, and discovered its limitations the hard way. Then I'd repeat the cycle with a second library. This process might take a full day, maybe two.

Instead, I described the problem to an LLM and had it implement five different extraction methods simultaneously. Five working scripts, five sets of output. I looked at the results, not the code. One method handled embedded images well but missed vector graphics. Another captured everything but produced blurry output. A third nailed it. Total time from problem to validated solution: minutes. The code for the four failed approaches? Deleted without a second thought.

I'm currently trying to train a neural network to play a perfect game of Snake using reinforcement learning. I'm experimenting with different input representations, neural architectures, reward functions, and RL algorithms. In the old world, each of these would be a significant commitment. You'd read papers, pick an approach, spend days implementing it, then discover it didn't converge. Before I had these tools, I spent two months just reproducing someone else's mechanistic interpretability visualization from scratch. Two months of setup before I could even begin to experiment with my own ideas.

Now I can run parallel experiments across multiple dimensions of the problem space. The failed experiments cost almost nothing.

This works even when the LLM has never seen your specific combination of tools. There's no training data for "this particular RL algorithm applied to Snake with this specific neural architecture and reward shaping." The LLM is just competent enough at each individual component that you can compose novel combinations and quickly see if they work. You're searching the solution space faster than any individual could.

How did engineering teams previously validate technical approaches? Design documents reviewed by committee. Senior engineers offering opinions based on past experience. Gut feel about which database or framework would work best. Sometimes these instincts were right. Often they weren't, and the team wouldn't find out for weeks.

The instinct to plan carefully before writing code used to be a virtue. It was the correct adaptation to an environment where writing code was expensive. That environment no longer exists. Engineers who spend a week debating the right approach in a doc could have validated all the candidate approaches in an afternoon.

This changes your relationship with code itself. Exploration code becomes disposable. When generation cost approaches zero, you generate a lot and keep very little. Write five implementations, keep one. Or keep none and try five more. The old instinct to carefully craft and preserve every line of code you write actively slows you down.

Today, the binding constraint is token generation speed and the latency between the AI and your development environment. Faster inference and shorter round-trips mean more experiments per hour, which means you land on better solutions sooner.

That gap between what you can imagine and what you can test keeps narrowing.
