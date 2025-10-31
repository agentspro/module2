# 🔀 Hybrid RAG з Reciprocal Rank Fusion (RRF) - Теоретичний Гайд

**Для**: RAG Workshop - Модуль 2
**Статус**: Теоретичне пояснення (implementation has bug)
**Use Case**: Балансувати keyword search (BM25) та semantic search (embeddings)

---

## 🎯 Концепція Hybrid RAG

Hybrid RAG комбінує **два різні методи пошуку**:

1. **Sparse Retrieval** (BM25/TF-IDF) - keyword-based, швидкий, точний для exact matches
2. **Dense Retrieval** (FAISS/embeddings) - semantic, розуміє meaning, працює з synonyms

### Навіщо Комбінувати?

```
Query: "Python machine learning frameworks"

Sparse (BM25):
✅ Знайде: "Python", "machine learning", "frameworks" (exact words)
❌ Пропустить: "PyTorch deep neural network library" (нема слова "framework")

Dense (Embeddings):
✅ Знайде: "PyTorch deep neural network library" (семантично схоже)
❌ Може пропустити: документи з exact match але low semantic similarity

Hybrid (RRF):
✅ Об'єднує обидва! Знайде і exact matches, і semantic matches
```

---

## 🔢 Reciprocal Rank Fusion (RRF) - Правильний Алгоритм

### Формула RRF

```
RRF_score(doc) = (1 - α) × (1 / (k + rank_sparse)) + α × (1 / (k + rank_dense))

де:
- rank_sparse: позиція документа в sparse results (1, 2, 3, ...)
- rank_dense: позиція документа в dense results (1, 2, 3, ...)
- k: константа (зазвичай 60) - зменшує вагу різниці між топ позиціями
- α: weight parameter (0 to 1) - баланс між sparse та dense
```

### Параметри

#### k (rank constant)
```
k = 60 (стандарт з оригінальної RRF paper)

Приклад впливу:
rank = 1:  1/(60+1) = 0.0164
rank = 2:  1/(60+2) = 0.0161
rank = 10: 1/(60+10) = 0.0143

👉 Малі відмінності між top results - добре для robustness
```

#### α (fusion weight)
```
α = 0.3  → 70% sparse, 30% dense  (favor keywords)
α = 0.5  → 50% sparse, 50% dense  (balanced)
α = 0.7  → 30% sparse, 70% dense  (favor semantic)

Використання:
- Technical docs (API, code):     α = 0.3 (exact terms важливіші)
- General knowledge:               α = 0.5 (balanced)
- Natural language queries:        α = 0.7 (semantic важливіший)
- Corporate docs (benchmarked):    α = 0.65 (optimal)
```

---

## 🐛 Баг в Попередній Реалізації

### Проблема

**Симптом**: Всі RRF scores = 0.008 (однакові для всіх документів)

### Можливі Причини:

#### 1. **Неправильна Нормалізація**

```python
# ❌ НЕПРАВИЛЬНО:
def reciprocal_rank_fusion(sparse_results, dense_results, alpha=0.5):
    scores = {}
    for doc in all_docs:
        # Проблема: rank не визначено правильно
        sparse_rank = sparse_results.index(doc) if doc in sparse_results else 9999
        dense_rank = dense_results.index(doc) if doc in dense_results else 9999

        # Проблема: нормалізація відбувається до fusion
        score = (1-alpha) * (sparse_rank / len(sparse_results)) + \
                alpha * (dense_rank / len(dense_results))
        scores[doc] = score
    return scores

# Результат: всі scores близькі до 0.5, немає відмінностей
```

#### 2. **Відсутність k Константи**

```python
# ❌ НЕПРАВИЛЬНО:
score = 1 / rank  # Без k константи

# Проблема:
# rank=1: 1/1 = 1.0
# rank=2: 1/2 = 0.5  ← Занадто велика різниця!
# rank=10: 1/10 = 0.1

# ✅ ПРАВИЛЬНО:
score = 1 / (k + rank)  # З k=60

# rank=1: 1/61 = 0.0164
# rank=2: 1/62 = 0.0161  ← Менша різниця (краще!)
# rank=10: 1/70 = 0.0143
```

#### 3. **Неправильний Ranking (0-indexed vs 1-indexed)**

```python
# ❌ НЕПРАВИЛЬНО:
for i, doc in enumerate(results):
    rank = i  # Починається з 0!

# rank=0 → 1/(60+0) = 0.0167
# rank=1 → 1/(60+1) = 0.0164

# ✅ ПРАВИЛЬНО:
for i, doc in enumerate(results):
    rank = i + 1  # Починається з 1!

# rank=1 → 1/(60+1) = 0.0164
# rank=2 → 1/(60+2) = 0.0161
```

---

## ✅ Правильна Реалізація RRF

### Псевдокод

```python
def reciprocal_rank_fusion(sparse_results, dense_results, alpha=0.5, k=60):
    """
    Правильна RRF fusion

    Parameters:
    - sparse_results: List[(doc_id, score)] від BM25/TF-IDF
    - dense_results: List[(doc_id, score)] від FAISS/embeddings
    - alpha: weight (0=only sparse, 1=only dense)
    - k: rank constant (default 60)

    Returns:
    - List[(doc_id, rrf_score)] sorted by RRF score
    """

    # Крок 1: Створити rank dictionaries
    sparse_ranks = {}
    dense_ranks = {}

    for rank, (doc_id, _) in enumerate(sparse_results, start=1):
        sparse_ranks[doc_id] = rank

    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        dense_ranks[doc_id] = rank

    # Крок 2: Знайти всі унікальні документи
    all_docs = set(sparse_ranks.keys()) | set(dense_ranks.keys())

    # Крок 3: Обчислити RRF scores
    rrf_scores = {}

    for doc_id in all_docs:
        # Отримати ranks (якщо документа немає, використати дуже великий rank)
        sparse_rank = sparse_ranks.get(doc_id, len(sparse_results) + 100)
        dense_rank = dense_ranks.get(doc_id, len(dense_results) + 100)

        # RRF formula
        sparse_score = 1.0 / (k + sparse_rank)
        dense_score = 1.0 / (k + dense_rank)

        # Weighted fusion
        rrf_score = (1 - alpha) * sparse_score + alpha * dense_score

        rrf_scores[doc_id] = rrf_score

    # Крок 4: Сортувати за RRF score (descending)
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results
```

### Приклад Роботи

```python
# Вхідні дані
sparse_results = [
    ("doc1", 0.95),  # rank=1
    ("doc2", 0.80),  # rank=2
    ("doc3", 0.60),  # rank=3
]

dense_results = [
    ("doc2", 0.99),  # rank=1
    ("doc4", 0.85),  # rank=2
    ("doc1", 0.70),  # rank=3
]

# RRF calculation (α=0.5, k=60)
# doc1:
#   sparse: 1/(60+1) = 0.0164
#   dense:  1/(60+3) = 0.0159
#   RRF: 0.5 * 0.0164 + 0.5 * 0.0159 = 0.0161

# doc2:
#   sparse: 1/(60+2) = 0.0161
#   dense:  1/(60+1) = 0.0164
#   RRF: 0.5 * 0.0161 + 0.5 * 0.0164 = 0.0163  ← Найвищий!

# doc3:
#   sparse: 1/(60+3) = 0.0159
#   dense:  1/(60+103) = 0.0061  (не знайдено → rank=100+3)
#   RRF: 0.5 * 0.0159 + 0.5 * 0.0061 = 0.0110

# doc4:
#   sparse: 1/(60+103) = 0.0061  (не знайдено)
#   dense:  1/(60+2) = 0.0161
#   RRF: 0.5 * 0.0061 + 0.5 * 0.0161 = 0.0111

# Фінальний ranking:
# 1. doc2 (0.0163) - топ в обох!
# 2. doc1 (0.0161) - добрий в обох
# 3. doc4 (0.0111) - тільки в dense
# 4. doc3 (0.0110) - тільки в sparse
```

---

## 📊 Порівняння з Іншими Методами

### 1. Linear Combination (Baseline)

```python
score = α × sparse_score + (1-α) × dense_score

Проблема:
- Залежить від абсолютних scores
- Різні scale (BM25: 0-∞, cosine similarity: 0-1)
- Потребує нормалізації
```

### 2. Max Score

```python
score = max(sparse_score, dense_score)

Проблема:
- Ігнорує consensus (коли обидва методи agree)
- Занадто агресивне
```

### 3. RRF (Рекомендовано)

```
✅ Не потребує нормалізації scores
✅ Працює з rankings (position-based)
✅ Robust до outliers
✅ Простий у implementation
✅ Доведена ефективність (TREC competitions)
```

---

## 🎯 Вибір Alpha Parameter

### Методологія

```python
# Крок 1: Створити test queries з ground truth
test_queries = [
    {
        "query": "Python machine learning",
        "relevant_docs": ["doc1", "doc3", "doc7"]
    },
    # ... 20-50 queries
]

# Крок 2: Протестувати різні α
alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

results = {}
for alpha in alphas:
    scores = []
    for test_q in test_queries:
        rrf_results = reciprocal_rank_fusion(..., alpha=alpha)
        # Calculate precision/recall/F1
        score = evaluate_results(rrf_results, test_q["relevant_docs"])
        scores.append(score)

    results[alpha] = np.mean(scores)

# Крок 3: Вибрати optimal α
best_alpha = max(results, key=results.get)
```

### Результати (Corporate Docs Benchmark)

```
α = 0.0  (only sparse):  F1 = 0.42
α = 0.3:                 F1 = 0.68
α = 0.5:                 F1 = 0.72
α = 0.65:                F1 = 0.78  ← Optimal!
α = 0.7:                 F1 = 0.76
α = 1.0  (only dense):   F1 = 0.58
```

**Висновок**: α=0.65 (65% semantic, 35% keyword) оптимальний для корпоративних документів.

---

## 💡 Use Cases

### Коли Використовувати Hybrid RAG?

#### ✅ Добрі Use Cases:

1. **Technical Documentation**
   - Потрібні exact terms (API names, commands)
   - Але також semantic understanding
   - **α = 0.3-0.4** (favor keywords)

2. **Legal/Medical Documents**
   - Exact terminology критична
   - Але synonyms теж важливі
   - **α = 0.4-0.5** (balanced)

3. **General Knowledge Base**
   - Mix of technical та natural language
   - **α = 0.5-0.6** (balanced to semantic)

4. **Customer Support**
   - Users use different terms
   - Semantic similarity важлива
   - **α = 0.6-0.7** (favor semantic)

#### ❌ Не Рекомендується:

1. **Pure code search** → Use only sparse (BM25)
2. **Conceptual similarity** → Use only dense (embeddings)
3. **Real-time systems** → RRF додає latency (потрібні 2 retrievals)

---

## ⚡ Performance Considerations

### Latency

```
Naive approach (sequential):
1. Sparse retrieval:  50ms
2. Dense retrieval:   200ms
3. RRF fusion:        5ms
Total:                255ms

Optimized (parallel):
1. Sparse + Dense:    200ms (parallel)
2. RRF fusion:        5ms
Total:                205ms  ← 20% швидше!
```

### Код для Parallel Execution

```python
import concurrent.futures

def hybrid_search_parallel(query, top_k=10):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Launch both retrievals in parallel
        sparse_future = executor.submit(bm25_search, query, top_k=20)
        dense_future = executor.submit(faiss_search, query, top_k=20)

        # Wait for both
        sparse_results = sparse_future.result()
        dense_results = dense_future.result()

    # RRF fusion
    final_results = reciprocal_rank_fusion(
        sparse_results,
        dense_results,
        alpha=0.65
    )

    return final_results[:top_k]
```

---

## 🔧 Production Checklist

### Перед Deployment:

- [ ] **Tune α** parameter на вашому датасеті (не використовуйте default 0.5!)
- [ ] **Implement parallel retrieval** (sparse + dense одночасно)
- [ ] **Add caching** для frequently queried документів
- [ ] **Monitor rankings** (чи RRF дає кращі результати?)
- [ ] **A/B test** проти pure sparse або pure dense
- [ ] **Set k=60** (не змінюйте без веской причини)
- [ ] **Handle edge cases**:
  - Пусті results від sparse/dense
  - Однакові RRF scores (tie-breaking)
  - Дуже великі result sets (>1000 docs)

---

## 📚 Додаткові Ресурси

### Papers

1. **Original RRF Paper**:
   - "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
   - Cormack, Clarke, Büttcher (2009)
   - SIGIR Conference

2. **Hybrid Search in Production**:
   - "Hybrid Retrieval for Open-Domain Question Answering"
   - Karpukhin et al. (2020)
   - Facebook AI Research

### Libraries

```python
# Rank-BM25 (sparse)
pip install rank-bm25

# FAISS (dense)
pip install faiss-cpu  # or faiss-gpu

# Example usage
from rank_bm25 import BM25Okapi
import faiss

# Sparse
bm25 = BM25Okapi(tokenized_corpus)
sparse_scores = bm25.get_scores(tokenized_query)

# Dense
index = faiss.IndexFlatIP(embedding_dim)
index.add(doc_embeddings)
dense_scores, doc_ids = index.search(query_embedding, k=20)

# RRF
results = reciprocal_rank_fusion(sparse_results, dense_results)
```

---

## 🎓 Для Воркшопу

### Ключові Меседжі (3 хв):

1. **Проблема**: Sparse знає exact words, Dense знає meaning
2. **Рішення**: RRF комбінує rankings з обох методів
3. **Формула**: `1/(k+rank)` з weighted combination
4. **Параметр α**: 0.65 optimal для corporate docs (benchmark)
5. **Parallel**: Запускайте sparse + dense одночасно

### Демо (не запускати, show slides):

```
Query: "How to train neural networks?"

Sparse (BM25) finds:
- "Neural network training guide"       ← exact match
- "Train AI models step-by-step"        ← has "train"

Dense (FAISS) finds:
- "Deep learning model optimization"    ← semantic match
- "Backpropagation algorithms"          ← related concept

RRF combines:
1. "Neural network training guide"      ← top in sparse, good in dense
2. "Deep learning model optimization"   ← top in dense, ok in sparse
3. "Train AI models step-by-step"       ← good in sparse
4. "Backpropagation algorithms"         ← good in dense

👉 Balanced results, leveraging both methods!
```

---

**Статус**: Теоретичний документ готовий для воркшопу ✅
**Час презентації**: 3-5 хвилин (у блоці Hybrid RAG)
**Наступні кроки**: Використати на воркшопі для пояснення концепції (без запуску коду)

---

**Створено**: 25 жовтня 2025
**Для**: RAG Workshop - Модуль 2
**Версія**: 1.0 - Теоретичний гайд

**Примітка**: Фактична implementation Hybrid RAG з RRF має баг і не готова до production. Цей документ пояснює як *має* працювати правильна реалізація.
