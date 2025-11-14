Course: Natural Language Processing
Student Name: Tejaswini Kolluru
Student ID: 700773943

1. Character-level RNN (GRU)
Goal

The goal of this task is to train a simple character-level RNN that predicts the next character in a text sequence using PyTorch.

Architecture Used:
Embedding → GRU → Linear → Softmax

Training Method:
Teacher forcing, Adam optimizer, and cross-entropy loss.

Dataset

The model can be trained on:

A tiny toy corpus (e.g., "hello", "help")

A small public-domain text (100–200 KB)

A custom .txt file uploaded by the user

Model & Training Details

Hidden size: 64–256 (default: 128)

Sequence length: 50–100 (default: 100)

Batch size: 64

Epochs: 5–20

Teacher forcing for stable training

Results Generated

Training and validation loss curves

Three sample text generations at temperatures 0.7, 1.0, 1.2

Reflection discussing sequence length, hidden size, temperature effects, and connections to class slides

Connection to Lecture Slides

Embeddings encode characters as vectors

Sampling loop generates text one character at a time

Teacher forcing improves stability

GRUs balance simplicity and performance

Transformers extend these ideas using attention mechanisms

2. Mini Transformer Encoder (NumPy)

This part of the assignment involves implementing a mini Transformer Encoder from scratch using NumPy. The notebook follows the instructions step-by-step and explains each part clearly.

What the notebook demonstrates:

It begins by selecting a small dataset of 10 short sentences.

It shows how the sentences were tokenized, how the vocabulary was built, and how token IDs were generated.

It demonstrates how sinusoidal positional encoding was added to the word embeddings.

It implements the main components of a Transformer encoder, including:

Self-attention

Multi-head attention (with 2 heads)

Feed-forward layer

Add & LayerNorm around both sublayers

It displays the required outputs:

Input token sequences

Final contextual embeddings

An attention heatmap showing how words attend to each other
