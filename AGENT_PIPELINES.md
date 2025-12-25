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

## Retrieval Augmented Generation (RAG) Pipeline

### Overview
RAG pipeline combines information retrieval with text generation to ground LLM responses in external knowledge sources, reducing hallucination and enabling access to up-to-date information.

### Use Cases
- Knowledge-intensive question answering
- Document-based QA systems
- Fact verification
- Research assistance
- Technical documentation queries

### Pipeline Flow

```
┌─────────────┐
│   User      │
│   Query     │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│   Query Processing                  │
│   - Extract keywords                │
│   - Generate embeddings             │
│   - Expand query (optional)         │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Document Retrieval                │
│   - Vector similarity search        │
│   - Top-K document selection        │
│   - Re-ranking (optional)           │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Retrieved Documents               │
│   Doc 1: [Relevant content]         │
│   Doc 2: [Relevant content]         │
│   Doc 3: [Relevant content]         │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Context Construction              │
│   - Combine query + documents       │
│   - Format as context               │
│   - Add instructions                │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   LLM Generation                    │
│   - Process query + context         │
│   - Generate grounded response      │
│   - Cite sources (optional)         │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────┐
│   Return    │
│   Answer +  │
│   Citations │
└─────────────┘
```

### Example: Science Question Answering

**Input Data:**
```
Question: "What was OKT3 originally sourced from?"
```

**Step 1: Query Embedding**
```
Query → Embedding Vector [0.234, -0.891, 0.456, ...]
```

**Step 2: Document Retrieval**
```
Retrieved Document (Top match):
"Teplizumab traces its roots to a New Jersey drug company called Ortho 
Pharmaceutical. There, scientists generated an early version of the antibody, 
dubbed OKT3. Originally sourced from mice, the molecule was able to bind to 
the surface of T cells and limit their cell-killing potential. In 1986, it 
was approved to help prevent organ rejection after kidney transplants, making 
it the first therapeutic antibody allowed for human use."

Similarity Score: 0.92
```

**Step 3: Context Construction**
```
Answer the question based on the context below. Keep the answer short and concise. 
Respond "Unsure about answer" if not sure about the answer.

Context: Teplizumab traces its roots to a New Jersey drug company called Ortho 
Pharmaceutical. There, scientists generated an early version of the antibody, 
dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the 
surface of T cells and limit their cell-killing potential. In 1986, it was 
approved to help prevent organ rejection after kidney transplants, making it the 
first therapeutic antibody allowed for human use.

Question: What was OKT3 originally sourced from?
Answer:
```

**Step 4: LLM Response**
```
Mice
```

**Data Flow:**
1. **Input** → User question: "What was OKT3 originally sourced from?"
2. **Embedding** → Convert to vector: [0.234, -0.891, ...]
3. **Retrieval** → Find top-k similar documents from knowledge base
4. **Retrieved Context** → "...Originally sourced from mice..."
5. **Prompt Construction** → Question + Context + Instructions
6. **LLM Processing** → Generate answer grounded in context
7. **Output** → "Mice" (with optional citation)

### Advanced RAG Variant: Multi-Step Retrieval

```
┌─────────────┐
│   Query     │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│   Initial Retrieval                 │
│   Top-K documents                   │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   LLM: Identify Missing Info        │
│   "Need more context about X"       │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Second Retrieval                  │
│   Query for missing information     │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Combine All Retrieved Docs        │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Generate Final Answer             │
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
import numpy as np
from typing import List, Dict

client = OpenAI()

class RAGPipeline:
    def __init__(self, knowledge_base: List[Dict]):
        """
        knowledge_base: List of documents with 'text' and 'embedding' fields
        """
        self.knowledge_base = knowledge_base
    
    def embed_query(self, query: str) -> np.ndarray:
        """Step 1: Convert query to embedding"""
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        return np.array(response.data[0].embedding)
    
    def retrieve_documents(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Dict]:
        """Step 2: Retrieve top-k similar documents"""
        similarities = []
        
        for doc in self.knowledge_base:
            doc_embedding = np.array(doc['embedding'])
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((similarity, doc))
        
        # Sort by similarity and get top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in similarities[:top_k]]
    
    def construct_context(self, query: str, documents: List[Dict]) -> str:
        """Step 3: Build context from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}: {doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Answer the question based on the context below. Keep the answer short and concise. 
Respond "Unsure about answer" if not sure about the answer.

Context: {context}

Question: {query}
Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """Step 4: Generate answer using LLM"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for factual answers
            max_tokens=250
        )
        return response.choices[0].message.content
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """Complete RAG pipeline"""
        # Step 1: Embed query
        query_embedding = self.embed_query(question)
        
        # Step 2: Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query_embedding, top_k)
        
        # Step 3: Construct context
        prompt = self.construct_context(question, retrieved_docs)
        
        # Step 4: Generate answer
        answer = self.generate_answer(prompt)
        
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_docs,
            "prompt": prompt
        }

# Usage
knowledge_base = [
    {
        "text": "Teplizumab traces its roots to a New Jersey drug company called Ortho Pharmaceutical. There, scientists generated an early version of the antibody, dubbed OKT3. Originally sourced from mice, the molecule was able to bind to the surface of T cells and limit their cell-killing potential.",
        "embedding": [...]  # Pre-computed embedding
    },
    # More documents...
]

rag = RAGPipeline(knowledge_base)
result = rag.query("What was OKT3 originally sourced from?")

print("Answer:", result["answer"])
print("Sources:", len(result["sources"]), "documents")
```

### Key Characteristics
- **Grounding**: Answers based on retrieved facts, not just model knowledge
- **Reduced Hallucination**: Citations and evidence reduce fabrication
- **Up-to-date**: Can access latest information from knowledge base
- **Transparency**: Can show source documents
- **Complexity**: Requires vector database and embedding models
- **Latency**: Additional retrieval step adds processing time

### Performance Benefits
- **Accuracy**: 20-30% improvement on knowledge-intensive QA
- **Factuality**: Significant reduction in hallucinated facts
- **Coverage**: Can answer questions beyond training data cutoff

### Components Required
1. **Vector Database**: Store document embeddings (Pinecone, Weaviate, FAISS)
2. **Embedding Model**: Convert text to vectors (text-embedding-ada-002, sentence-transformers)
3. **Retrieval System**: Similarity search and ranking
4. **LLM**: Generate answer from context (GPT-4, Claude, etc.)

---

