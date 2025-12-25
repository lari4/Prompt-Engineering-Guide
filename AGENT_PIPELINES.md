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

## ReAct Agent Pipeline

### Overview
ReAct (Reasoning and Acting) pipeline interleaves reasoning traces with task-specific actions, allowing the agent to think, act, and observe results iteratively until solving the task.

### Use Cases
- Multi-step question answering requiring research
- Fact verification with evidence gathering
- Interactive problem solving
- Decision-making tasks with environment interaction

### Pipeline Flow

```
┌─────────────┐
│   User      │
│   Question  │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│   ReAct Prompt with Examples        │
│   - Show Thought-Action-Observation │
│   - Provide tool descriptions       │
└──────┬──────────────────────────────┘
       │
       v
    ┌──▼──────────────────────────────┐
    │   LLM: Generate Thought         │
    │   "I need to find X..."         │
    └──┬──────────────────────────────┘
       │
       v
    ┌──▼──────────────────────────────┐
    │   LLM: Decide Action            │
    │   "Search[topic]"               │
    └──┬──────────────────────────────┘
       │
       v
    ┌──▼──────────────────────────────┐
    │   Execute Action via Tool       │
    │   - Search engine               │
    │   - Wikipedia API               │
    │   - Calculator                  │
    │   - Database query              │
    └──┬──────────────────────────────┘
       │
       v
    ┌──▼──────────────────────────────┐
    │   Observation                   │
    │   Return tool result            │
    └──┬──────────────────────────────┘
       │
       v
    ┌──▼──────────────────────────────┐
    │   Add to Context                │
    │   Thought + Action + Observation│
    └──┬──────────────────────────────┘
       │
       v
    ┌──▼──────────────────────────────┐
    │   Check if Task Complete?       │
    └──┬───────────────┬──────────────┘
       │ No            │ Yes
       │               │
       │               v
       │          ┌────────────┐
       │          │   Return   │
       │          │   Final    │
       │          │   Answer   │
       │          └────────────┘
       │
       └──> Loop back to "Generate Thought"
```

### Example: Multi-Hop Question Answering

**Input Data:**
```
Question: "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?"
```

**Iteration 1:**

**Thought 1:**
```
I need to search Colorado orogeny, find the area that the eastern sector extends into, then find the elevation range of that area.
```

**Action 1:**
```
Search[Colorado orogeny]
```

**Observation 1:**
```
The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas. This took place from 1780 to 1650 million years ago (Mya), during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico. The eastern sector extends into the high plains and is called the Central Plains orogeny.
```

**Iteration 2:**

**Thought 2:**
```
The eastern sector of Colorado orogeny extends into the high plains. I need to search high plains and find its elevation range.
```

**Action 2:**
```
Search[high plains]
```

**Observation 2:**
```
High Plains refers to one of two distinct land regions: the Great Plains, or the High Plains (United States). The High Plains are a subregion of the Great Plains, mainly in the Western United States. The High Plains have an elevation ranging from 1,800 to 7,000 feet (550 to 2,130 m).
```

**Iteration 3:**

**Thought 3:**
```
High Plains have an elevation range of 1,800 to 7,000 feet, so the answer is 1,800 to 7,000 feet.
```

**Action 3:**
```
Finish[1,800 to 7,000 feet]
```

**Final Answer:**
```
1,800 to 7,000 feet
```

**Data Flow:**
1. **Input** → Complex question requiring multiple steps
2. **Thought 1** → Plan first step: search Colorado orogeny
3. **Action 1** → Execute search tool
4. **Observation 1** → Receive: eastern sector = high plains
5. **Thought 2** → Plan second step: search high plains
6. **Action 2** → Execute search tool
7. **Observation 2** → Receive: elevation 1,800-7,000 feet
8. **Thought 3** → Conclude with answer
9. **Action 3** → Finish with result
10. **Output** → "1,800 to 7,000 feet"

### Implementation Code

```python
from openai import OpenAI
import re
from typing import Dict, List, Tuple

client = OpenAI()

class ReActAgent:
    def __init__(self, tools: Dict):
        """
        tools: Dictionary of available tools/functions
        Example: {"Search": search_function, "Calculator": calc_function}
        """
        self.tools = tools
        self.max_iterations = 10
    
    def create_react_prompt(self, question: str, examples: List[str], history: List[Tuple]) -> str:
        """Build ReAct prompt with examples and interaction history"""
        
        # Few-shot examples of Thought-Action-Observation
        prompt = "Solve a question answering task with interleaving Thought, Action, Observation steps.\n\n"
        
        # Add examples
        for example in examples:
            prompt += example + "\n\n"
        
        # Add current question
        prompt += f"Question: {question}\n"
        
        # Add interaction history
        for i, (thought, action, observation) in enumerate(history, 1):
            prompt += f"Thought {i}: {thought}\n"
            prompt += f"Action {i}: {action}\n"
            prompt += f"Observation {i}: {observation}\n"
        
        # Prompt for next thought
        prompt += f"Thought {len(history) + 1}:"
        
        return prompt
    
    def parse_action(self, text: str) -> Tuple[str, str]:
        """Extract action and argument from LLM response"""
        # Pattern: Action[argument]
        match = re.search(r'Action \d+: (\w+)\[(.*?)\]', text)
        if match:
            tool_name = match.group(1)
            argument = match.group(2)
            return tool_name, argument
        return None, None
    
    def execute_tool(self, tool_name: str, argument: str) -> str:
        """Execute the specified tool with argument"""
        if tool_name in self.tools:
            return self.tools[tool_name](argument)
        return f"Error: Tool {tool_name} not found"
    
    def solve(self, question: str, examples: List[str]) -> Dict:
        """Main ReAct loop"""
        history = []
        
        for iteration in range(self.max_iterations):
            # Step 1: Generate Thought
            prompt = self.create_react_prompt(question, examples, history)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
                stop=["\nObservation"]
            )
            
            thought_and_action = response.choices[0].message.content
            
            # Extract thought
            thought_match = re.search(r'Thought \d+: (.+?)(?:\n|$)', thought_and_action)
            thought = thought_match.group(1) if thought_match else ""
            
            # Step 2: Parse Action
            # Continue prompt to get action
            action_prompt = prompt + thought_and_action + "\nAction " + str(iteration + 1) + ":"
            
            action_response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": action_prompt}],
                temperature=0.7,
                max_tokens=50,
                stop=["\n"]
            )
            
            action_text = "Action " + str(iteration + 1) + ": " + action_response.choices[0].message.content
            tool_name, argument = self.parse_action(action_text)
            
            # Check if task is complete
            if tool_name == "Finish":
                return {
                    "answer": argument,
                    "iterations": iteration + 1,
                    "history": history,
                    "final_thought": thought
                }
            
            # Step 3: Execute Action
            observation = self.execute_tool(tool_name, argument)
            
            # Step 4: Add to history
            history.append((thought, action_text, observation))
        
        return {
            "answer": "Max iterations reached",
            "iterations": self.max_iterations,
            "history": history
        }

# Define tools
def search_wikipedia(query: str) -> str:
    """Mock search function - in practice, call Wikipedia API"""
    knowledge_base = {
        "Colorado orogeny": "The Colorado orogeny was an episode of mountain building in Colorado and surrounding areas. The eastern sector extends into the high plains and is called the Central Plains orogeny.",
        "high plains": "The High Plains are a subregion of the Great Plains. The High Plains have an elevation ranging from 1,800 to 7,000 feet (550 to 2,130 m)."
    }
    return knowledge_base.get(query, "No information found")

# Usage
tools = {
    "Search": search_wikipedia,
    "Finish": lambda x: x
}

examples = [
    """Question: What is the capital of France?
Thought 1: I can answer this directly from my knowledge.
Action 1: Finish[Paris]"""
]

agent = ReActAgent(tools)
result = agent.solve(
    "What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?",
    examples
)

print("Answer:", result["answer"])
print("Iterations:", result["iterations"])
for i, (t, a, o) in enumerate(result["history"], 1):
    print(f"\n--- Iteration {i} ---")
    print(f"Thought: {t}")
    print(f"Action: {a}")
    print(f"Observation: {o}")
```

### Key Characteristics
- **Interleaved Reasoning & Action**: Alternates between thinking and acting
- **Tool Integration**: Can use external tools (search, calculator, APIs)
- **Iterative**: Multiple thought-action-observation cycles
- **Transparent**: Full reasoning trace is visible
- **Flexible**: Can adjust plan based on observations
- **Self-Correcting**: Can fix mistakes by gathering more information

### Performance Benefits
- **HotpotQA**: 27% → 34% accuracy improvement over action-only
- **FEVER**: Better fact verification with explicit reasoning
- **Human Interpretability**: Clear decision-making process

### Available Actions/Tools
- **Search[entity]**: Look up information
- **Lookup[keyword]**: Find specific detail in last search
- **Calculator[expression]**: Perform calculations
- **Finish[answer]**: Return final answer
- **Custom tools**: Database queries, API calls, code execution

---

## Prompt Chaining Pipeline

### Overview
Prompt chaining decomposes complex tasks into a sequence of subtasks, where each subtask is handled by a separate prompt, and the output of one becomes the input to the next.

### Use Cases
- Document question answering with extraction + synthesis
- Multi-stage content generation
- Complex data transformations
- Validation and refinement workflows

### Pipeline Flow

```
┌─────────────┐
│   Input     │
│   Document  │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│   Prompt 1: Extract Relevant Quotes│
│   Task: Find quotes related to Q   │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Output 1: List of Quotes          │
│   <quotes>                          │
│   Quote 1: "..."                    │
│   Quote 2: "..."                    │
│   </quotes>                         │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Prompt 2: Synthesize Answer       │
│   Input: Quotes + Original Doc     │
│   Task: Compose helpful answer     │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Output 2: Final Answer            │
│   "Based on the quotes..."          │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Prompt 3: Quality Check (Optional)│
│   Verify accuracy and tone          │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────┐
│   Return    │
│   Final     │
│   Answer    │
└─────────────┘
```

### Example: Document QA with Chaining

**Input:**
```
Document: [Long research paper about climate change]
Question: "What are the main causes of global warming?"
```

**Chain Step 1: Extract Quotes**

**Prompt 1:**
```
You are a helpful assistant. Your task is to extract quotes from the document that are relevant to answering the question.

Document: {document}
Question: {question}

Please output the list of quotes using <quotes></quotes> tags.
```

**Output 1:**
```
<quotes>
"The primary cause of global warming is the increase in greenhouse gas emissions from fossil fuel combustion."
"Deforestation contributes significantly by reducing CO2 absorption capacity."
"Industrial processes and agriculture also emit substantial amounts of methane and nitrous oxide."
</quotes>
```

**Chain Step 2: Synthesize Answer**

**Prompt 2:**
```
Given the relevant quotes extracted from a document and the original question, please compose a comprehensive answer.

Quotes: {quotes_from_step1}
Question: {question}

Ensure the answer is accurate, has a friendly tone, and sounds helpful.
```

**Output 2:**
```
The main causes of global warming include:
1. Greenhouse gas emissions from burning fossil fuels (coal, oil, natural gas)
2. Deforestation, which reduces the Earth's capacity to absorb CO2
3. Industrial processes and agriculture, which emit methane and nitrous oxide

These factors combine to trap heat in the atmosphere, leading to rising global temperatures.
```

**Data Flow:**
1. **Input** → Document + Question
2. **Stage 1** → Extract relevant quotes → Output: Quote list
3. **Stage 2** → Synthesize from quotes → Output: Coherent answer
4. **Stage 3** (Optional) → Verify quality → Output: Validated answer

### Implementation Code

```python
from openai import OpenAI
client = OpenAI()

def prompt_chain_pipeline(document: str, question: str):
    # Chain Step 1: Extract quotes
    prompt1 = f"""You are a helpful assistant. Extract quotes from the document relevant to the question.

Document: {document}
Question: {question}

Output quotes using <quotes></quotes> tags."""

    response1 = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt1}],
        temperature=0.3,
        max_tokens=500
    )
    
    quotes = response1.choices[0].message.content
    
    # Chain Step 2: Synthesize answer
    prompt2 = f"""Compose a comprehensive answer to the question using the extracted quotes.

Quotes: {quotes}
Question: {question}

Be accurate, friendly, and helpful."""

    response2 = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt2}],
        temperature=0.7,
        max_tokens=300
    )
    
    answer = response2.choices[0].message.content
    
    return {
        "question": question,
        "extracted_quotes": quotes,
        "final_answer": answer
    }
```

### Key Characteristics
- **Modularity**: Each step is isolated and testable
- **Transparency**: Intermediate outputs are visible
- **Debuggability**: Easy to identify which step fails
- **Controllability**: Can modify individual prompts
- **Reliability**: Better than single complex prompt

### Benefits
- **Improved Accuracy**: Focused prompts perform better
- **Easier Debugging**: Clear failure points
- **Flexibility**: Can add/remove steps
- **Cost Optimization**: Can use different models for different steps

---

## Self-Consistency Voting Pipeline

### Overview
Self-consistency generates multiple reasoning paths for the same problem and selects the most consistent answer through majority voting.

### Use Cases
- Arithmetic and mathematical reasoning
- Complex problem solving with multiple solution paths
- Improving reliability of Chain-of-Thought

### Pipeline Flow

```
┌─────────────┐
│   Problem   │
└──────┬──────┘
       │
       ├────────┬────────┬────────┬────────┐
       │        │        │        │        │
       v        v        v        v        v
    ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
    │Path 1││Path 2││Path 3││Path 4││Path 5│
    │      ││      ││      ││      ││      │
    │CoT   ││CoT   ││CoT   ││CoT   ││CoT   │
    │      ││      ││      ││      ││      │
    └───┬──┘└───┬──┘└───┬──┘└───┬──┘└───┬──┘
        │       │       │       │       │
        v       v       v       v       v
    ┌───────────────────────────────────────┐
    │   Answer 1  Answer 2  Answer 3       │
    │   "42"      "42"      "41"            │
    │   Answer 4  Answer 5                 │
    │   "42"      "40"                     │
    └───────┬───────────────────────────────┘
            │
            v
    ┌───────────────────┐
    │   Majority Vote   │
    │   42: 3 votes     │
    │   41: 1 vote      │
    │   40: 1 vote      │
    └────────┬──────────┘
             │
             v
    ┌────────────────┐
    │  Final Answer  │
    │     "42"       │
    └────────────────┘
```

### Example: Math Problem

**Input:**
```
Problem: "When I was 6 my sister was half my age. Now I'm 70, how old is my sister?"
```

**Path 1 (Temperature 0.7):**
```
When I was 6, my sister was half my age, so she was 3.
The age difference is 6 - 3 = 3 years.
Now I'm 70, so my sister is 70 - 3 = 67 years old.
Answer: 67
```

**Path 2 (Temperature 0.8):**
```
At age 6, half is 3, so sister was 3 years old.
Age gap: 6 - 3 = 3 years (constant).
Current age: 70 - 3 = 67.
Answer: 67
```

**Path 3 (Temperature 0.9):**
```
Sister was 3 when I was 6.
I'm 3 years older.
70 - 3 = 67.
Answer: 67
```

**Path 4 (Temperature 0.7):**
```
Half of 6 is 3, sister's age then.
Now: 70 - (6-3) = 67.
Answer: 67
```

**Path 5 (Temperature 0.8):**
```
Sister age at my 6: 3
Years passed: 70 - 6 = 64
Sister now: 3 + 64 = 67
Answer: 67
```

**Voting:**
- Answer "67": 5 votes ✓
- **Final Answer: 67** (unanimous)

### Implementation Code

```python
from openai import OpenAI
from collections import Counter
import re

client = OpenAI()

def self_consistency_pipeline(problem: str, num_paths: int = 5):
    """Generate multiple reasoning paths and vote"""
    
    answers = []
    reasoning_paths = []
    
    # Generate multiple paths with different temperatures
    for i in range(num_paths):
        prompt = f"{problem}\n\nLet's think step by step."
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 + (i * 0.05),  # Vary temperature for diversity
            max_tokens=300
        )
        
        reasoning = response.choices[0].message.content
        reasoning_paths.append(reasoning)
        
        # Extract answer (last number or last line)
        # This is simplified - in practice, use more robust parsing
        answer_match = re.findall(r'\d+', reasoning)
        if answer_match:
            answers.append(answer_match[-1])
    
    # Majority voting
    vote_counts = Counter(answers)
    most_common_answer, vote_count = vote_counts.most_common(1)[0]
    
    return {
        "problem": problem,
        "reasoning_paths": reasoning_paths,
        "answers": answers,
        "vote_counts": dict(vote_counts),
        "final_answer": most_common_answer,
        "confidence": vote_count / num_paths
    }

# Usage
result = self_consistency_pipeline(
    "When I was 6 my sister was half my age. Now I'm 70, how old is my sister?",
    num_paths=5
)

print("Final Answer:", result["final_answer"])
print("Confidence:", f"{result['confidence']*100}%")
print("Vote Distribution:", result["vote_counts"])
```

### Key Characteristics
- **Robustness**: More reliable than single path
- **Diversity**: Explores different reasoning approaches
- **Consensus**: Voting reduces impact of errors
- **Cost**: Requires multiple API calls (higher cost)
- **Latency**: Can parallelize for speed

### Performance Improvements
- **Arithmetic**: ~17% → ~74% accuracy on GSM8K
- **Commonsense**: Significant gains on StrategyQA
- **Best with**: Temperature sampling (0.5-1.0) for diversity

---

## Tree of Thoughts Exploration Pipeline

### Overview
Tree of Thoughts (ToT) maintains a tree of reasoning paths, enabling systematic exploration with lookahead and backtracking using search algorithms.

### Use Cases
- Complex strategic planning
- Game-like problems (Game of 24)
- Creative writing with constraints
- Multi-step optimization problems

### Pipeline Flow

```
┌─────────────┐
│   Problem   │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│   Generate Initial Thoughts (Level 1)│
│   Thought A | Thought B | Thought C │
└─┬───────────┬───────────┬───────────┘
  │           │           │
  v           v           v
┌────────┐ ┌────────┐  ┌────────┐
│Evaluate│ │Evaluate│  │Evaluate│
│A: 0.7  │ │B: 0.9  │  │C: 0.4  │
└───┬────┘ └────┬───┘  └────┬───┘
    │           │(best)      │
    │           v            │
    │     ┌────────────────────────────┐
    │     │  Expand Best: Generate     │
    │     │  Thought B1 | B2 | B3      │
    │     └─┬──────────┬──────────┬────┘
    │       v          v          v
    │    ┌────┐     ┌────┐     ┌────┐
    │    │Eval│     │Eval│     │Eval│
    │    │0.8 │     │0.6 │     │0.9 │
    │    └────┘     └────┘     └─┬──┘
    │                            │(best)
    v                            v
[Backtrack if needed]     ┌──────────────┐
                          │  Continue    │
                          │  Expansion   │
                          └──────┬───────┘
                                 │
                                 v
                          ┌──────────────┐
                          │   Solution   │
                          │   Found      │
                          └──────────────┘
```

### Example: Game of 24

**Problem:** "Use 4 numbers (4, 9, 10, 13) and basic arithmetic operations (+, -, *, /) to obtain 24. Each number must be used exactly once."

**Tree Exploration:**

**Level 1 - Generate Initial Thoughts:**
```
Thought 1: (13 - 9) * (10 - 4) = 4 * 6 = 24 ✓
Thought 2: (10 - 4) * 13 - 9 = 6 * 13 - 9 = 69 ✗
Thought 3: 13 * 9 - 10 - 4 = 117 - 14 = 103 ✗
```

**Evaluation:**
- Thought 1: Sure (correct solution found)
- Thought 2: Impossible (too large)
- Thought 3: Impossible (too large)

**Result:** Solution found at Level 1, no need for deeper exploration.

**Complex Example Requiring Backtracking:**

**Problem:** "4, 5, 6, 10 → 24"

**Level 1:**
```
Thought A: (10 - 6) * 5 + 4 = 4 * 5 + 4 = 24 ✓ [Solution]
Thought B: (10 - 4) * 6 - 5 = 6 * 6 - 5 = 31 [Explore]
Thought C: 10 + 6 + 5 + 4 = 25 [Dead end]
```

### Implementation Code

```python
from openai import OpenAI
from typing import List, Tuple
import re

client = OpenAI()

class TreeOfThoughts:
    def __init__(self, max_depth: int = 3, breadth: int = 3):
        self.max_depth = max_depth
        self.breadth = breadth  # Number of thoughts to generate per level
    
    def generate_thoughts(self, problem: str, current_path: List[str]) -> List[str]:
        """Generate possible next thoughts"""
        context = "\n".join(current_path) if current_path else ""
        
        prompt = f"""Problem: {problem}

Current reasoning path:
{context}

Generate {self.breadth} different next steps or thoughts to solve this problem.
Each thought should be on a new line."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=300
        )
        
        thoughts = response.choices[0].message.content.strip().split('\n')
        return thoughts[:self.breadth]
    
    def evaluate_thought(self, problem: str, thought: str) -> Tuple[str, float]:
        """Evaluate if thought is promising (sure/likely/impossible)"""
        prompt = f"""Problem: {problem}
Thought: {thought}

Evaluate if this thought leads toward solving the problem.
Respond with one of: sure (100% correct), likely (promising), unlikely (poor direction), impossible (wrong)

Evaluation:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50
        )
        
        evaluation = response.choices[0].message.content.lower()
        
        # Map evaluation to score
        if "sure" in evaluation:
            return "sure", 1.0
        elif "likely" in evaluation:
            return "likely", 0.7
        elif "unlikely" in evaluation:
            return "unlikely", 0.3
        else:
            return "impossible", 0.0
    
    def bfs_search(self, problem: str) -> dict:
        """Breadth-first search through thought tree"""
        # Queue: (path, depth)
        queue = [([], 0)]
        best_solution = None
        explored = []
        
        while queue:
            current_path, depth = queue.pop(0)
            
            if depth >= self.max_depth:
                continue
            
            # Generate thoughts for current node
            thoughts = self.generate_thoughts(problem, current_path)
            
            for thought in thoughts:
                # Evaluate thought
                evaluation, score = self.evaluate_thought(problem, thought)
                
                new_path = current_path + [thought]
                explored.append({
                    "path": new_path,
                    "thought": thought,
                    "evaluation": evaluation,
                    "score": score,
                    "depth": depth + 1
                })
                
                # Check if solution found
                if evaluation == "sure":
                    best_solution = new_path
                    break
                
                # Add promising paths to queue
                if score >= 0.5:
                    queue.append((new_path, depth + 1))
            
            if best_solution:
                break
        
        return {
            "solution": best_solution,
            "explored_nodes": len(explored),
            "exploration_tree": explored
        }

# Usage
tot = TreeOfThoughts(max_depth=3, breadth=3)
result = tot.bfs_search("Use 4, 9, 10, 13 with +, -, *, / to get 24. Each number used once.")

print("Solution:", result["solution"])
print("Nodes explored:", result["explored_nodes"])
```

### Key Characteristics
- **Systematic Exploration**: BFS or DFS through thought space
- **Evaluation**: Self-assess each thought's promise
- **Backtracking**: Can revisit earlier choices
- **Lookahead**: Evaluate before committing
- **Resource-Intensive**: Multiple LLM calls per level

### Performance Benefits
- **Game of 24**: 74% success rate (vs 7.3% with CoT)
- **Creative Writing**: Better coherence with constraints
- **Strategic Tasks**: Superior planning capabilities

---

## Reflexion Learning Pipeline

### Overview
Reflexion enables agents to learn from mistakes through self-reflection, creating linguistic feedback that improves performance across trials.

### Use Cases
- Code generation with debugging
- Decision-making tasks requiring trial-and-error
- Iterative problem solving
- Learning from failures

### Pipeline Flow

```
┌─────────────┐
│   Task      │
└──────┬──────┘
       │
       v
┌─────────────────────────────────────┐
│   Trial 1                           │
│   Actor: Generate solution          │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Evaluator: Test solution          │
│   Result: Failed (Score: 3/10)      │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Self-Reflection                   │
│   "The solution failed because I    │
│    didn't consider edge case X.     │
│    Next time, I should check Y."    │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Episodic Memory                   │
│   Store: Attempt + Reflection       │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Trial 2 (with memory)             │
│   Actor: Generate improved solution │
│   Consider previous reflection      │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Evaluator: Test solution          │
│   Result: Better (Score: 7/10)      │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────────────────────────────┐
│   Self-Reflection                   │
│   "Much better, but still need to   │
│    optimize the data structure."    │
└──────┬──────────────────────────────┘
       │
       v
       ... (continue until success or max trials)
       │
       v
┌─────────────────────────────────────┐
│   Trial N                           │
│   Result: Success! (Score: 10/10)   │
└──────┬──────────────────────────────┘
       │
       v
┌─────────────┐
│   Return    │
│   Solution  │
└─────────────┘
```

### Example: Code Generation with Debugging

**Task:** "Write a Python function to check if a number is prime."

**Trial 1:**

**Actor Output:**
```python
def is_prime(n):
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
```

**Evaluator Result:**
```
Test cases:
- is_prime(1): Expected False, Got True ✗
- is_prime(2): Expected True, Got True ✓
- is_prime(10): Expected False, Got False ✓
- is_prime(17): Expected True, Got True ✓

Score: 3/4 tests passed
Error: Fails for n=1
```

**Self-Reflection:**
```
The function incorrectly returns True for n=1. The issue is that 1 is not 
considered prime, but my function doesn't check for this edge case. Also, 
checking all numbers up to n is inefficient. I should:
1. Add check for n <= 1
2. Optimize by only checking up to sqrt(n)
```

**Memory Stored:**
```
Trial 1: Failed for n=1
Reflection: Need edge case handling for n <= 1, consider sqrt optimization
```

**Trial 2 (with reflection):**

**Actor Output (improved):**
```python
import math

def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True
```

**Evaluator Result:**
```
All test cases passed: 10/10 ✓
Performance: Optimized with sqrt check
```

**Success!**

### Implementation Code

```python
from openai import OpenAI
from typing import List, Dict

client = OpenAI()

class ReflexionAgent:
    def __init__(self, max_trials: int = 5):
        self.max_trials = max_trials
        self.memory = []  # Episodic memory of reflections
    
    def actor_generate(self, task: str) -> str:
        """Generate solution attempt"""
        # Include past reflections in context
        reflection_context = "\n".join([
            f"Previous attempt {i+1} reflection: {r['reflection']}"
            for i, r in enumerate(self.memory)
        ])
        
        prompt = f"""Task: {task}

Previous reflections:
{reflection_context if reflection_context else "None (first attempt)"}

Generate a solution:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def evaluator_test(self, task: str, solution: str) -> Dict:
        """Evaluate the solution"""
        prompt = f"""Task: {task}
Solution: {solution}

Test this solution and provide:
1. Test results
2. Score (0-10)
3. Specific errors if any

Evaluation:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        
        evaluation = response.choices[0].message.content
        
        # Extract score (simplified parsing)
        import re
        score_match = re.search(r'Score:\s*(\d+)', evaluation)
        score = int(score_match.group(1)) if score_match else 0
        
        return {
            "evaluation": evaluation,
            "score": score,
            "passed": score >= 8
        }
    
    def self_reflect(self, task: str, solution: str, evaluation: Dict) -> str:
        """Generate reflection on failure"""
        prompt = f"""Task: {task}
Your solution: {solution}
Evaluation result: {evaluation['evaluation']}

Reflect on what went wrong and how to improve. Be specific about:
1. What failed and why
2. What to try differently next time

Reflection:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        return response.choices[0].message.content
    
    def solve(self, task: str) -> Dict:
        """Main Reflexion loop"""
        for trial in range(self.max_trials):
            # Actor: Generate solution
            solution = self.actor_generate(task)
            
            # Evaluator: Test solution
            evaluation = self.evaluator_test(task, solution)
            
            # Check if success
            if evaluation['passed']:
                return {
                    "success": True,
                    "trial": trial + 1,
                    "solution": solution,
                    "score": evaluation['score'],
                    "reflections": self.memory
                }
            
            # Self-Reflection: Learn from failure
            reflection = self.self_reflect(task, solution, evaluation)
            
            # Store in memory
            self.memory.append({
                "trial": trial + 1,
                "solution": solution,
                "score": evaluation['score'],
                "reflection": reflection
            })
        
        return {
            "success": False,
            "trials": self.max_trials,
            "best_solution": self.memory[-1]['solution'],
            "best_score": max(m['score'] for m in self.memory),
            "reflections": self.memory
        }

# Usage
agent = ReflexionAgent(max_trials=5)
result = agent.solve("Write a Python function to check if a number is prime")

print("Success:", result["success"])
print("Trials needed:", result.get("trial", result.get("trials")))
if result["success"]:
    print("Solution:", result["solution"])
```

### Key Characteristics
- **Learning from Mistakes**: Explicit reflection mechanism
- **Memory**: Stores past attempts and learnings
- **Iterative Improvement**: Each trial benefits from previous reflections
- **No Fine-Tuning**: Works with frozen LLM
- **Self-Evaluation**: Agent critiques its own work

### Performance Benefits
- **AlfWorld**: 90% → 97% success rate
- **HotpotQA**: Better with each trial
- **HumanEval**: 91% success on code generation
- **Efficiency**: Learns faster than traditional RL

---

