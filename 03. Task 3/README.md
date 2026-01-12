# 📝 Task 3 — Abstractive Text Summarization (XSum)

## 🎯 Overview
Task 3 focuses on **abstractive text summarization** using a **Transformer-based foundation model** on the **XSum dataset**.  
This task demonstrates the complete deep learning pipeline, including data acquisition, preprocessing, fine-tuning, and inference, under realistic computational constraints.

The implementation emphasizes **correct workflow and reasoning**, rather than large-scale training.

---

## 📌 Objectives
- Apply a Transformer-based foundation model for text summarization  
- Use the XSum dataset via **Hugging Face Dataset API**  
- Perform **lightweight fine-tuning** on a sampled subset of data  
- Generate abstractive summaries from unseen text  

---

## 🗂️ Workflow Summary
1. **Dataset Access**
   - XSum data retrieved using Hugging Face Dataset Server API
   - Subset sampling applied to reduce computational cost

2. **Preprocessing**
   - Input text formatted with summarization prompts
   - Tokenization of input text and reference summaries

3. **Model Fine-Tuning**
   - Lightweight fine-tuning using a Transformer-based summarization model
   - Small batch size and limited epochs

4. **Inference**
   - Generate abstractive summaries after fine-tuning
   - Compare model output with reference summaries

---

## ⚙️ Model & Tools
- **Model**: Transformer-based summarization model (e.g., FLAN-T5-Small)
- **Framework**: PyTorch
- **Libraries**:
  - transformers
  - torch
  - requests
- **Platform**: Google Colab (GPU)

---

## 📁 File Structure
```text
Task_3/
├── task_3.ipynb
├── README.md
├── requirements.txt
```

---

## ✅ Notes
- Dataset scripts are not used directly to avoid compatibility issues
- Sampling is applied as permitted by the course instructor
- Fine-tuning is performed at a small scale for demonstration purposes
- The focus is on pipeline correctness and reproducibility

---

✨ This task demonstrates practical application of modern NLP models for abstractive text summarization within academic constraints.
