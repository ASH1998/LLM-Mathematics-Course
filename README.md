# Full Mathematics and Engineering Coursework for Large Language Models

This curriculum merges the previous versions, expands to **20 modules**, and adds hands‑on coding deliverables for deeper understanding. Follow sequentially or choose modules as needed.

---

## 1. Bird‑Eye Syllabus

| Mod | Focus | Core Topics | Key Coding Deliverables |
|-----|-------|-------------|-------------------------|
| 01 | Linear Algebra | vector spaces, norms, SVD, tensor contractions | batched SVD notebook, einsum attention prototype |
| 02 | Multivariate Calculus and Autodiff | Jacobians, Hessians, reverse‑mode autodiff | build micrograd clone, gradient check transformer layer |
| 03 | Probability and Information Theory | KL divergence, entropy, cross‑entropy | perplexity calculator, masked softmax sampler |
| 04 | Optimization Theory | SGD, Adam, Lion, convexity | optimizer benchmark suite on quadratic and CIFAR‑tiny |
| 05 | Fundamental Learning Theory | PAC, VC dimension, double descent | VC calculator script, bias‑variance simulation |
| 06 | Discrete Math and Graphs | suffix tries, DAGs, attention sparsity graphs | build suffix tree tokenization visualizer |
| 07 | Sequence Models Before Transformers | n‑gram, RNN, LSTM | trigram LM in PyTorch, gradient norm tracker |
| 08 | Attention Mechanics | scaled dot‑product, temperature, masking | single‑head attention from scratch, entropy explorer |
| 09 | Transformer Layer Math | positional encodings, residuals, layer norm | two‑layer transformer training on Tiny Shakespeare |
| 10 | Tokenization and Embeddings | BPE, unigram, cosine similarity | tokenizer trainer CLI, embedding isotropy histogram |
| 11 | Scaling Laws | compute vs params, gradient noise scale | reproduce Kaplan plots, fit scaling exponents |
| 12 | Large‑Batch and Mixed Precision | ZeRO, checkpointing, BF16 | train 70 M parameter LM with ZeRO‑3 on single GPU |
| 13 | RL From Human Feedback | policy gradients, PPO, KL penalty | PPO fine‑tune toy chatbot, KL tracker dashboard |
| 14 | Evaluation and Safety Math | perplexity, MMLU, privacy epsilon | eval harness runner, DP noise budget estimator |
| 15 | Advanced Representation Theory | NTK, lottery tickets, random matrices | NTK spectrum plot, lottery ticket pruning experiment |
| 16 | Retrieval Augmentation and Vector Search | FAISS, HNSW, negative sampling | build RAG pipeline, recall vs latency plot |
| 17 | LLM Engineering Pipeline | data ingestion, sharding, distributed training | end‑to‑end pretraining script, Weights and Biases logging |
| 18 | Inference Optimization and Quantization | INT8, GPTQ, spec‑ulative decoding | Triton or vLLM server with spec‑decoding benchmark |
| 19 | Monitoring, Logging and Observability | perplexity drift, latency SLOs | Prometheus + Grafana dashboard for live model |
| 20 | Responsible AI and Governance | bias metrics, eval cards, licensing | create model card, implement bias evaluation notebook |

---

## 2. Detailed Module Guide with Deliverables

### Module 01 - Linear Algebra
* **Reading:** Axler, *Linear Algebra Done Right*, chapters 1-6
* **Proof Exercise:** orthogonal projection formula, Eckart-Young theorem
* **Coding:** `batched_svd.py` to verify reconstruction error <= 1e-6  
  `einsum_attention.ipynb` implementing Q K V attention using `torch.einsum`

### Module 02 - Multivariate Calculus and Autodiff
* Chain rule in matrix form, forward vs reverse mode
* **Coding:** Build a 100‑line autodiff engine, then gradient check a transformer feed‑forward block on random input

### Module 03 - Probability and Information Theory
* Derive cross‑entropy from likelihood
* **Coding:** `perplexity.py` that computes perplexity on WikiText‑2, plus `temperature_sampler.py` to explore sampling entropy

### Module 04 - Optimization Theory
* SGD vs Adam vs Lion, Polyak‑Lojasiewicz condition
* **Coding:** Benchmark optimizers on quadratic bowl and CIFAR‑tiny classification, output convergence plots

### Module 05 - Fundamental Learning Theory
* PAC bound, VC dimension, bias‑variance, double descent
* **Coding:** `vc_dimension.ipynb` to compute VC for threshold and axis‑aligned splits, plus bias‑variance visualizer on synthetic data

### Module 06 - Discrete Math and Graphs
* Prefix trees, DAG compression, sparse attention graphs
* **Coding:** Build a suffix tree visualizer for given text, export to graphviz

### Module 07 - Sequence Models Before Transformers
* Vanishing/exploding gradients, GRU equations
* **Coding:** Implement trigram LM and GRU LM, track hidden‑state norms

### Module 08 - Attention Mechanics
* Complexity O(n^2 d), temperature effect on entropy
* **Coding:** `attention_entropy.py` plotting entropy vs temperature on toy sequence

### Module 09 - Transformer Layer Math
* Sinusoidal and rotary encodings, residual path analysis
* **Coding:** Two‑layer transformer trained on Tiny Shakespeare with custom positional encodings, visualize gradient flow

### Module 10 - Tokenization and Embeddings
* BPE merge heuristics, isotropy
* **Coding:** Trainer CLI for BPE and unigram; histogram of cosine similarities between random embedding pairs

### Module 11 - Scaling Laws
* Fit power‑law loss surfaces
* **Coding:** Reproduce Kaplan et al. plots for model zoo of 1 M-100 M params, estimate alpha beta gamma

### Module 12 - Large‑Batch and Mixed Precision Training
* ZeRO‑3, BF16, activation checkpointing
* **Coding:** Pretrain 70 M LM on one GPU with bf16 ZeRO‑3, log throughput tokens/sec

### Module 13 - RL From Human Feedback
* KL‑regularized PPO, reward models
* **Coding:** Toy chatbot fine‑tuned with PPO, plot KL divergence and reward per epoch

### Module 14 - Evaluation and Safety Math
* Perplexity vs accuracy, differential privacy
* **Coding:** Eval harness covering MMLU subset, compute DP epsilon for Gaussian noise on gradients

### Module 15 - Advanced Representation Theory
* Neural tangent kernel, random matrix spectrum
* **Coding:** NTK spectrum plot for 1‑layer transformer, lottery ticket pruning to 20 percent sparsity and retrain

### Module 16 - Retrieval Augmentation and Vector Search
* FAISS and HNSW, negative sampling
* **Coding:** Build RAG pipeline indexing 100k Wikipedia passages, measure recall@5 vs latency

### Module 17 - LLM Engineering Pipeline
* Ingestion, sharding, distributed data parallel
* **Coding:** End‑to‑end pretraining script that streams Common Crawl shard, logs metrics to Weights and Biases

### Module 18 - Inference Optimization and Quantization
* GPTQ, INT8, speculative decoding
* **Coding:** Deploy quantized model with vLLM, benchmark tokens/sec and latency p95, implement speculative decoding wrapper

### Module 19 - Monitoring, Logging and Observability
* Perplexity drift, latency SLO violations
* **Coding:** Prometheus exporter scraping Triton metrics, Grafana dashboard visualizing tokens/sec and p95 latency

### Module 20 - Responsible AI and Governance
* Bias metrics, model cards, licenses
* **Coding:** Generate model card MD for trained model, implement bias detection notebook using StereoSet and toxicity classifier

---

## 3. Assessment Structure

| Component | Weight | Details |
|-----------|-------|---------|
| Weekly proof quizzes | 10 % | LaTeX write‑ups of key theorems |
| Hands‑on coding labs | 40 % | Modules 01‑20 deliverables, auto‑graded |
| Paper replication project | 20 % | Recreate a figure from a landmark paper |
| Engineering capstone | 30 % | Train and deploy a compute‑optimal LM with RAG and monitoring |

---

## 4. Primary Resources

* Goodfellow, Bengio, Courville - *Deep Learning*
* Shalev‑Shwartz & Ben‑David - *Understanding Machine Learning*
* Vaswani et al. 2017 - Attention Is All You Need
* Kaplan et al. 2020 - Scaling Laws for Neural Language Models
* DeepMind Chinchilla paper
* Hugging Face Course - Training Transformers at Scale
* OpenAI Spinning‑Up PPO Guide
* FAISS, vLLM, Triton official docs

---

## Study Workflow

1. **Preview** module aims.
2. **Read** core materials.
3. **Prove** the math by hand.
4. **Code** deliverables in PyTorch.
5. **Discuss** in weekly journal club.

Maintain a single LaTeX or Jupyter notebook summarizing proofs, results and reflections.

Enjoy the journey!

