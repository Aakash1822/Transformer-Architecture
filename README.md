## The Transformer Architecture, From Scratch

#### The Problem Transformers Solve

Imaging translating "The cat sat on the mat" to French. To translate "sat" you need to know who is sitting(the cat). Words depend on other words - sometimes far away in the sentence. Earlier models (like RNNs) read one word at a time, left to right, and struggled to things from way back. Transformers solved this by looking at all words simultaneously and letting each word "pay attention" to every other word.

### High-Level Picture

A transformer has two halves:

- Encoder - reads and understand the input
- Decoder - generates the output word by word

Each half is a stack of identical layers. Let's zoom into what's inside one layer

### Step 1: Turning Words Into Numbers - Embeddings

Computers can't work with words directly. Every word gets mapped to a list of numbers called vector(e.g., 512 numbers). Similar words get similar vectors. This is the embedding layer - just a big lookup table learned during training

### Step 2: Telling Words Where They Are - Positional Encoding

Because a transformer reads all words at once (not one by one), it has no built-in sense of order. "Dog bites man" and "Man bites dog" would look same without order info.

So we add a postional encdoing - another vector that encodes position(word1, word2, etc.) - to each word's embedding. Now the model knows both what the word is and where it appears

### Step 3: The Core Idea - Self- Attention

This is the heart of the transformer. For every word, self-attention asks:"which other words in this sentence should I pay attention to when processing this word?"

Here's how it works:

Each word create three vectors:

- Query (Q) - "What I am looking for?"
- Key (K) - "What do I contain"
- Value (V) - "What information do I actually carry?"

Think of it like a search engine: your Query searches against everyone's Keys to find relevance scores, then uses those scores to pull a weighted mix of Values.

Mathematically:

1. Cmpute a score Q.K (dot product=measure of similarity)
2. Scale it down (divide by dimension) so numbers don't get too large
3. Apply **softmax** to turn scores into probabilities (they sum to 1)
4. Multiply by V and sum everything up

The result for each word is a new vector that's a blend of all words, weighted by relevance. "Cat" in "the cat sat" will pull heavily from "sat" to understand the verb relationship.

### Step 4: Multi-Head Attention

One attention pass captures one type of relationship. But words relate in many ways - grammatical role, coreference, semantic meaning, etc.

**Multi-head attention** runse self-attention multiple times in paraller (e.g., 8 or 16 times), each with different Q/K/V weight matrices. Each "head" learns to focus on different kinds of relationships. The results are concatenated and projected back to the original size.

### Step 5: Add & Norm

After attention, we add the original input back to the result - thisis a **residual connection**. It prevents information loss and helps gradients flow during training(a deep network problem).

Then we apply **Layer Normalization** - rescaling the numbers so they have a consistent mean and variance. This stabilizes training.

output = LayerNorm(x + Attention(x))

### Step 6: Feed-Forward Network

After attention, each word's passes through a small **feed-forward neural network** independently (same network applied to each position). It's to linear layers with a ReLU activation in between:

FFN(x) = ReLU(xW1 + b1)W2 + b2

This is where the model does aditional "thinking" per word - pattern matching, feature transformation, etc. Then another Add & Norm follows.

### Step 7: The Decoder's Extra Trick - Cross Attention

The decoder generates output one word at a time. It has two attention layers:

1. **Masked self-attention** - attends to previously generated output words (masked so it can't cheat by looking at future words)

2. **Cross-attention** - attends to the encoder's output, so the decoder can "look at the input" while generating each word


##### **Why Transformers Win** 

- All position processed in parallel
- Every layer has full access to all positions
- Attention connects any two words directly


