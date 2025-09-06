# ThoughtTrim: Anchor-Driven Prompt Compression
Optimize training and inference by discovering “thought anchors,” reinforcing only strong reasoning steps, and compressing prompts so agents follow anchors not meandering chains of thought.
*Inspired by “Thought Anchors: Which LLM Reasoning Steps Matter?” (Bogdan, Macar, Nanda, Conmy, 2025).*

# Why this exists

Reasoning models often produce long chain-of-thought (CoT) traces where only a few sentences causally steer the outcome. Building on evidence that planning/backtracking sentences act as “thought anchors”, ThoughtTrim:

1. Finds anchors (high-influence sentences) in traces
2. Trains policies with RL to follow anchors and avoid weak steps
3. Compresses prompts to the anchor plan so generation is faster, cheaper, and more explainable
4. Trains a “logic-following llm” that plans-execute-checks using only strong steps.
5. Trains a "compression-following agent" that plans-execute-checks using only strong and compressed steps steps.

This submission targets the CBRN × AI Risks Research Sprint with a safety-first design: we improve reliability/faithfulness and reduce unnecessary reasoning sprawl without enabling misuse.
