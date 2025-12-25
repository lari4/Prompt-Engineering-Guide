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

## Chain-of-Thought Reasoning Pipeline

### Overview
Chain-of-Thought (CoT) pipeline enables complex reasoning by prompting the LLM to generate intermediate reasoning steps before producing the final answer.

### Use Cases
- Mathematical problem solving
- Logical reasoning tasks
- Multi-step analysis
- Complex decision making

### Pipeline Flow

```
┌─────────────┐
│   User      │
│   Problem   │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│   CoT Prompt Construction           │
│   - Add few-shot examples with     │
│     reasoning steps                 │
│   - Add "Let's think step by step" │
│   - Format problem                  │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   LLM Generates Reasoning Chain     │
│   Step 1: [Intermediate thought]   │
│   Step 2: [Intermediate thought]   │
│   Step 3: [Intermediate thought]   │
│   ...                               │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Extract Final Answer              │
│   - Parse reasoning chain           │
│   - Identify conclusion             │
│   - Return answer                   │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────┐
│   Return    │
│   Answer +  │
│   Reasoning │
└─────────────┘
```

### Example: Mathematical Reasoning

**Input Data:**
```
Problem: "The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1."
Task: Determine if statement is true
```

**Prompt with Few-Shot CoT:**
```
Q: The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.
A: Adding all the odd numbers (9, 15, 1) gives 25. The answer is False.

Q: The odd numbers in this group add up to an even number: 17, 10, 19, 4, 8, 12, 24.
A: Adding all the odd numbers (17, 19) gives 36. The answer is True.

Q: The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.
A:
```

**LLM Response (with reasoning):**
```
First, identify the odd numbers: 15, 5, 13, 7, 1.
Now add them: 15 + 5 + 13 + 7 + 1 = 41.
41 is an odd number.
The answer is False.
```

**Data Flow:**
1. **Input** → Problem statement
2. **CoT Examples** → Few-shot demonstrations with reasoning
3. **LLM Processing** → Generates step-by-step reasoning
4. **Reasoning Chain** → Intermediate thoughts: identify odds → calculate sum → determine odd/even
5. **Output** → Final answer with full reasoning trace

### Zero-Shot CoT Variant

```
┌─────────────┐
│   Problem   │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│   Add Magic Phrase                  │
│   "Let's think step by step"        │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   LLM Generates Reasoning           │
│   (No examples needed)              │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────┐
│   Answer    │
└─────────────┘
```

### Implementation Code

```python
from openai import OpenAI
client = OpenAI()

def chain_of_thought_pipeline(problem, examples=None, zero_shot=False):
    # Step 1: Construct prompt with CoT
    if zero_shot:
        prompt = f"{problem}\n\nLet's think step by step."
    else:
        # Use few-shot examples with reasoning
        prompt = ""
        for example in examples:
            prompt += f"Q: {example['question']}\n"
            prompt += f"A: {example['reasoning']}\n\n"
        prompt += f"Q: {problem}\nA:"

    # Step 2: Send to LLM
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )

    # Step 3: Extract reasoning chain and answer
    full_response = response.choices[0].message.content
    
    # Parse response to separate reasoning from answer
    lines = full_response.strip().split('\n')
    reasoning_steps = [line for line in lines if line.strip()]
    final_answer = reasoning_steps[-1] if reasoning_steps else ""

    return {
        "reasoning": reasoning_steps,
        "answer": final_answer,
        "full_response": full_response
    }

# Usage - Few-Shot CoT
examples = [
    {
        "question": "The odd numbers in this group add up to an even number: 4, 8, 9, 15, 12, 2, 1.",
        "reasoning": "Adding all the odd numbers (9, 15, 1) gives 25. The answer is False."
    }
]

result = chain_of_thought_pipeline(
    problem="The odd numbers in this group add up to an even number: 15, 32, 5, 13, 82, 7, 1.",
    examples=examples
)

print("Reasoning:", result["reasoning"])
print("Answer:", result["answer"])

# Usage - Zero-Shot CoT
result_zs = chain_of_thought_pipeline(
    problem="If I have 5 apples and buy 3 more, then give 2 away, how many do I have?",
    zero_shot=True
)
```

### Key Characteristics
- **Transparency**: Shows reasoning process
- **Accuracy**: Better performance on complex reasoning tasks
- **Debugging**: Easy to identify where reasoning fails
- **Flexibility**: Works with both few-shot and zero-shot
- **Limitations**: Requires more tokens, slower than basic pipeline

### Performance Improvements
- **Few-shot**: ~85-90% accuracy on arithmetic reasoning
- **Zero-shot**: ~65-75% accuracy (no examples needed)
- **Significant improvement** over direct answer prompting (~17% → 78% on some benchmarks)

---

