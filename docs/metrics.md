Evaluating large language models (LLMs) involves several metrics depending on the task—such as text generation, summarization, classification, or question-answering. Here are key metrics, their usage, formulas, and examples for each:

---

### 1. **Exact Match (EM)**

**Usage:** Measures whether the model’s output exactly matches the reference text, useful for tasks like question-answering.

**Formula:** 
$$
\text{EM} = \frac{\text{Number of Exact Matches}}{\text{Total Number of Examples}}
$$

**Example:**  
- Reference answer: "Paris is the capital of France."
- Model answer: "Paris is the capital of France."
- If answers match exactly, count as 1; otherwise, count as 0. Calculate EM over all examples.

---

### 2. **F1 Score**

**Usage:** Measures the overlap of words between the predicted and reference answers. Effective for QA, summarization, and retrieval tasks where partial matches are meaningful.

**Formula:**  
$$
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$
where  
$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$  
$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

**Example:**  
- Reference answer: "Paris is the capital of France."
- Model answer: "Paris is the capital."
- Precision = 3/4, Recall = 3/5, F1 = \(2 \cdot \frac{0.75 \cdot 0.6}{0.75 + 0.6} = 0.67\).

---

### 3. **BLEU (Bilingual Evaluation Understudy)**

**Usage:** Common for translation and generation, measures n-gram overlap between generated and reference texts.

**Formula:**  
$$
\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$  
where \(p_n\) is the precision for n-grams, \(w_n\) is the weight for each n-gram, and BP is the brevity penalty to penalize short sentences.

**Example:**  
- Reference: "The quick brown fox jumps over the lazy dog."
- Model output: "The quick fox jumps over the dog."
- BLEU will be calculated based on 1-grams, 2-grams, etc., with weights like \(w_1 = 0.25\) for each.

---

### 4. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

**Usage:** Primarily used for summarization tasks; measures overlap in n-grams, word sequences, and word pairs.

**Types:**  
- **ROUGE-N**: Precision and recall of n-grams.
- **ROUGE-L**: Longest Common Subsequence (LCS) similarity.
- **ROUGE-W**: Weighted LCS that gives higher weight to longer sequences.

**Example:**  
- Reference: "The quick brown fox jumps over the lazy dog."
- Model output: "The quick brown dog jumps over the fox."
- ROUGE-1, ROUGE-2, and ROUGE-L scores can each be calculated, focusing on 1-grams, 2-grams, and LCS matches.

---

### 5. **METEOR (Metric for Evaluation of Translation with Explicit ORdering)**

**Usage:** Designed for machine translation, but also used for text generation. Focuses on synonyms, stemming, and word order.

**Formula:**  
$$
\text{METEOR} = F \cdot (1 - \text{Penalty})
$$  
where \(F\) is a combination of precision and recall, and Penalty is based on the fragmentation of matched words.

**Example:**  
- Reference: "The boy is playing in the park."
- Model: "A child is playing at the park."
- Precision and recall account for synonyms (e.g., "boy" and "child") and similar word order, yielding a high METEOR score.

---

### 6. **BERTScore**

**Usage:** Evaluates similarity using embeddings, suitable for nuanced similarity detection in translation and paraphrasing.

**Formula:**  
Compute cosine similarity of token embeddings between the reference and generated text, then aggregate scores.

**Example:**  
- Reference: "The cat sat on the mat."
- Model: "A cat rested on the rug."
- BERTScore calculates similarity based on token embeddings, scoring higher for semantically similar texts.

---

### 7. **Perplexity**

**Usage:** Measures the uncertainty of a language model in generating text, commonly used in language modeling tasks.

**Formula:**  
$$
\text{Perplexity} = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 p(x_i)}
$$  
where \(p(x_i)\) is the probability of each word in the sequence.

**Example:**  
If the model predicts high probabilities for the next word, perplexity is low, indicating good model confidence in predicting contextually relevant words.

---

### 8. **Pass@k**

**Usage:** Measures the probability of finding a correct solution within the top \(k\) responses, useful for code generation tasks.

**Formula:**  
$$
\text{Pass@k} = 1 - \prod_{i=1}^{N} \left(1 - p_i\right)
$$  
where \(p_i\) is the probability of the \(i\)-th response being correct among \(k\) generated samples.

**Example:**  
If a coding model outputs three potential solutions for a problem, and at least one is correct, Pass@3 would indicate a successful outcome.

---

These metrics cover a wide range of LLM evaluation needs, from simple exact matching to embedding-based similarity, each with a unique focus suited to specific applications.