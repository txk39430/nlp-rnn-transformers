# 🌐 Natural Language Processing – Assignment  
**Student Name:** Tejaswini Kolluru  
**Student ID:** 700773943  

---

# 1. Character-level RNN (GRU)

##  Goal  
The task involves training a small **character-level RNN** that predicts the next character in a sequence using **PyTorch**.

**Model Architecture:**  
`Embedding → GRU → Linear → Softmax`

**Training Method:**  
Teacher forcing, Adam optimizer, and cross-entropy loss.

---

##  Dataset  
The model can be trained on any of the following:
- A tiny toy corpus (e.g., "hello", "help")  
- A 100–200 KB public-domain text (e.g., *Pride and Prejudice*)  
- A custom uploaded `.txt` file  

---

##  Model & Training Details  
- Hidden size: **64–256** (default: 128)  
- Sequence length: **50–100** (default: 100)  
- Batch size: **64**  
- Epochs: **5–20**  
- Teacher forcing for stable learning  

---

##  Results  
1. **Training & validation loss curves** plotted after training  
2. **Three sample generations** (length 200-400 chars) at temperatures:  
   - τ = 0.7  
   - τ = 1.0  
   - τ = 1.2  
3. **Reflection** discussing:  
   - Sequence length effects  
   - Hidden size effects  
   - Temperature differences  
   - Connections to class slides  

---

##  Connection to Lecture Slides  
- Embeddings represent characters in dense form  
- Sampling loop generates text one character at a time  
- Teacher forcing stabilizes learning  
- GRUs balance simplicity & performance  
- Transformers extend these ideas via attention  

---

# 2. Mini Transformer Encoder (NumPy)

This notebook implements a **mini Transformer Encoder** using NumPy, following all assignment instructions and explaining each step.

---

##  What the Notebook Demonstrates  

1. It begins by selecting a small dataset of **10 short sentences**.  

2. It shows how the sentences were **tokenized**, how the vocabulary was built, and how token IDs were created.  

3. It demonstrates how **sinusoidal positional encoding** was added to the word embeddings.  

4. It implements the main components of a Transformer encoder, including:  
   - Self-attention  
   - Multi-head attention (with **2 heads**)  
   - Feed-forward layer  
   - Add & LayerNorm around both sublayers  

5. It displays the required outputs, including:  
   - Input token sequences  
   - Final contextual embeddings  
   - An attention heatmap showing how words attend to one another  


