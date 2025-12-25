# AI Agent Pipelines Documentation

This document describes the various agent workflows and pipelines used in the Prompt Engineering Guide application. Each pipeline demonstrates how different prompts and techniques are chained together to accomplish complex tasks.

---

## Table of Contents

1. [Basic Single-Prompt Pipeline](#basic-single-prompt-pipeline)
2. [Chain-of-Thought Reasoning Pipeline](#chain-of-thought-reasoning-pipeline)
3. [Retrieval Augmented Generation (RAG) Pipeline](#retrieval-augmented-generation-rag-pipeline)
4. [ReAct Agent Pipeline](#react-agent-pipeline)
5. [Prompt Chaining Pipeline](#prompt-chaining-pipeline)
6. [Self-Consistency Voting Pipeline](#self-consistency-voting-pipeline)
7. [Tree of Thoughts Exploration Pipeline](#tree-of-thoughts-exploration-pipeline)
8. [Reflexion Learning Pipeline](#reflexion-learning-pipeline)
9. [Program-Aided Language Model (PAL) Pipeline](#program-aided-language-model-pal-pipeline)
10. [Multi-Stage Information Extraction Pipeline](#multi-stage-information-extraction-pipeline)

---

## Basic Single-Prompt Pipeline

### Overview
The simplest pipeline where a single prompt is sent to the LLM and a response is generated directly.

### Use Cases
- Simple classification tasks
- Direct question answering
- Code generation from descriptions
- Basic text transformations

### Pipeline Flow

```
┌─────────────┐
│   User      │
│   Input     │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│   Prompt Construction               │
│   - Add instructions                │
│   - Format user input               │
│   - Add constraints (optional)      │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   LLM Processing                    │
│   Model: GPT-4 / Mixtral / etc.     │
│   Temperature: 0.7-1.0              │
│   Max Tokens: 256-4000              │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Direct Response                   │
│   - Single output                   │
│   - No intermediate steps           │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────┐
│   Return    │
│   Result    │
└─────────────┘
```

### Example: Sentiment Classification

**Input Data:**
```
Text: "I think the food was okay."
Task: Classify sentiment
```

**Prompt Template:**
```
Classify the text into neutral, negative, or positive
Text: {input}
Sentiment:
```

**LLM Response:**
```
Neutral
```

**Data Flow:**
1. **Input** → User text + classification request
2. **Processing** → LLM analyzes text sentiment
3. **Output** → Single label (Neutral/Negative/Positive)

### Implementation Code

```python
from openai import OpenAI
client = OpenAI()

def single_prompt_pipeline(user_input):
    # Step 1: Construct prompt
    prompt = f"Classify the text into neutral, negative, or positive\nText: {user_input}\nSentiment:"

    # Step 2: Send to LLM
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        max_tokens=256
    )

    # Step 3: Extract result
    result = response.choices[0].message.content

    return result

# Usage
output = single_prompt_pipeline("I think the food was okay.")
print(output)  # "Neutral"
```

### Key Characteristics
- **Simplicity**: Single API call, minimal complexity
- **Speed**: Fastest pipeline type
- **Limitations**: No reasoning steps, no external knowledge, no error correction
- **Best For**: Well-defined, straightforward tasks

---

