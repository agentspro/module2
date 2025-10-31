# Професійний RAG Evaluation через RAGAS
## ✅ Професійний підхід: Синтетичні датасети

RAGAS може **автоматично згенерувати 50-200+ реалістичних тестових кейсів** з ваших PDF документів.

### Типи згенерованих запитів:

1. **Simple** (40%) - прості фактичні питання
   - "Що таке RAG?"
   - "Які основні компоненти RAG системи?"

2. **Reasoning** (30%) - потребують роздумів
   - "Чому Self-RAG краще ніж Naive RAG?"
   - "Як вибрати оптимальний chunk size?"

3. **Multi-context** (30%) - потребують кількох документів
   - "Порівняйте підходи BM25 та FAISS для retrieval"
   - "Як різні RAG підходи вирішують проблему Lost in the Middle?"

---

## 🚀 Як використовувати

### Крок 1: Встановлення залежностей

```bash
pip install ragas langchain-openai langchain-community pypdf datasets
```

**Важливо:** Потрібен OpenAI API key (RAGAS використовує GPT-4o-mini):

```bash
export OPENAI_API_KEY="your-key-here"
```

### Крок 2: Генерація синтетичного датасету

```bash
cd rag_demos
python generate_synthetic_testset.py
```

**Що робить:**
- Завантажує всі PDF з `data/pdfs/`
- Генерує 50 тестових кейсів (можна змінити через `TEST_SIZE`)
- Зберігає в `data/synthetic_testset.json`

**Час виконання:** 3-5 хвилин для 50 запитів

**Вартість:** ~$0.10-0.20 (GPT-4o-mini дешевий)

**Приклад виводу:**

```
📂 Завантаження документів з data/pdfs...
✅ Завантажено: 91 сторінок з PDF

🧪 Генерація 50 тестових запитів...
   Це займе 3-5 хвилин для 50 запитів

📊 Розподіл типів запитів:
   Simple (факти):          40%
   Reasoning (роздуми):     30%
   Multi-context (складні): 30%

✅ Згенеровано: 50 тестових кейсів за 234.5с

📝 ПРИКЛАДИ ЗГЕНЕРОВАНИХ ТЕСТОВИХ КЕЙСІВ:

Кейс #1
Тип: simple
Запит: What is the main purpose of Retrieval-Augmented Generation?
Ground Truth: RAG combines information retrieval with text generation to provide...
Контексти: 3 документів

Кейс #2
Тип: reasoning
Запит: Why does Self-RAG perform better than standard RAG on complex queries?
Ground Truth: Self-RAG uses adaptive retrieval and self-critique mechanisms...
Контексти: 5 документів
```

### Крок 3: Evaluation RAG системи

```bash
python evaluate_with_synthetic_testset.py
```

**Що робить:**
- Завантажує згенерований датасет
- Запускає RAG систему на всіх 50 запитах
- Оцінює через **4 RAGAS метрики**:
  - **Faithfulness** (0-1) - чи відповідь базується на контексті?
  - **Answer Relevancy** (0-1) - чи релевантна відповідь?
  - **Context Precision** (0-1) - чи точний retrieved контекст?
  - **Context Recall** (0-1) - чи повний retrieved контекст?

**Час виконання:** 5-10 хвилин для 50 запитів

**Вартість:** ~$0.50-1.00 (evaluation через GPT-4o-mini)

**Приклад виводу:**

```
🚀 Запуск RAG системи на тестовому датасеті...
   Обробка 50 запитів...
   Це займе ~2.5 хвилин

   Прогрес: 10/50 (20.0%) - Залишилось ~2.0 хв
   Прогрес: 20/50 (40.0%) - Залишилось ~1.5 хв
   ...

✅ Оброблено: 50 запитів за 2.8 хв
   Середній час на запит: 3.36с

📊 RAGAS Evaluation...
✅ Підготовлено 50 запитів для evaluation

✅ Evaluation завершено за 4.2 хв

======================================================================
📊 RAGAS МЕТРИКИ
======================================================================
Faithfulness:        0.923  ✅ (target > 0.90)
Answer Relevancy:    0.867  ✅ (target > 0.85)
Context Precision:   0.845  ✅ (target > 0.80)
Context Recall:      0.812  ✅ (target > 0.80)
----------------------------------------------------------------------
Average Score:       0.862
======================================================================

✅ EVALUATION ЗАВЕРШЕНО!

🎉 ВІДМІННО! Система готова для production
```

---

##  RAGAS Metrics - Що означають?

### 1. Faithfulness (Вірність контексту)

**Що вимірює:** **Чи базується відповідь на наданому контексті? Виявляє галюцинації.**

**Як працює:**
- LLM розбиває відповідь на твердження (statements)
- Перевіряє кожне твердження чи воно підтверджується контекстом
- Score = кількість підтверджених / загальна кількість

**Production target:** > 0.90 (критично!)

**Приклад:**
```
Question: "What is RAG?"
Context: "RAG combines retrieval with generation"
Answer: "RAG is a method that combines retrieval with generation and was invented in 2025"

Faithfulness: 0.50 (50%)
- ✅ "RAG combines retrieval with generation" - підтверджено
- ❌ "invented in 2025" - НЕ підтверджено (галюцинація!)
```

### 2. Answer Relevancy (Релевантність відповіді)

**Що вимірює:** **Чи відповідь релевантна запиту?**

**Як працює:**
- Генерує зворотні запити з відповіді
- Порівнює їх з оригінальним запитом через cosine similarity
- Високий score = відповідь прямо стосується запиту

**Production target:** > 0.85

**Приклад:**
```
Question: "What is Self-RAG?"
Answer: "Self-RAG is an advanced approach that uses adaptive retrieval"

Relevancy: 0.95 (відмінно!)

Question: "What is Self-RAG?"
Answer: "RAG systems are useful for QA tasks"

Relevancy: 0.40 (погано - загальна відповідь)
```

### 3. Context Precision (Точність контексту)

**Що вимірює:** Чи retrieved документи дійсно релевантні?

**Як працює:**
- Перевіряє кожен retrieved chunk чи він потрібен для відповіді
- Ранжує chunks по релевантності
- Карає за нерелевантні chunks у топі

**Production target:** > 0.80

**Низький score → багато "шуму" в retrieved контексті**

### 4. Context Recall (Повнота контексту)

**Що вимірює:** Чи знайшли ВСІ потрібні документи?

**Як працює:**
- Порівнює retrieved контекст з ground truth
- Перевіряє чи всі факти з ground truth є в контексті

**Production target:** > 0.80

**Низький score → missed relevant documents (треба більше top_k)**

---

##  Інтерпретація результатів

###  Відмінно (Average > 0.85)
```
Faithfulness:      0.92
Answer Relevancy:  0.88
Context Precision: 0.84
Context Recall:    0.81
→ Система готова для production!
```

###  Прийнятно (Average 0.70-0.85)
```
Faithfulness:      0.78
Answer Relevancy:  0.82
Context Precision: 0.71
Context Recall:    0.73
→ Потрібні покращення перед production
```

###  Критично (Average < 0.70)
```
Faithfulness:      0.65
Answer Relevancy:  0.58
Context Precision: 0.62
Context Recall:    0.55
→ Серйозна оптимізація необхідна!
```

---

##  Як покращити метрики?

### Faithfulness низький (< 0.90)
**Проблема:** Галюцинації

**Рішення:**
- Додати re-ranking контексту
- Фільтрувати низько-релевантні chunks
- Використати stronger LLM (GPT-4 замість GPT-3.5)
- Додати prompt "Answer ONLY based on context"

### Answer Relevancy низький (< 0.85)
**Проблема:** Відповіді не по темі

**Рішення:**
- Покращити LLM prompt
- Додати query rewriting
- Використати few-shot examples

### Context Precision низький (< 0.80)
**Проблема:** Багато нерелевантних documents в retrieved

**Рішення:**
- Покращити embeddings (використати better model)
- Додати hybrid search (BM25 + vectors)
- Додати re-ranking stage
- Зменшити top_k

### Context Recall низький (< 0.80)
**Проблема:** Пропускаємо relevant documents

**Рішення:**
- Збільшити top_k
- Покращити chunking strategy (менший chunk_size)
- Додати multi-query retrieval
- Використати parent document retrieval

---

##  Порівняння підходів

Після evaluation можна порівняти різні RAG підходи:

```bash
# Згенерувати датасет (1 раз)
python generate_synthetic_testset.py

# Evaluation різних підходів
python evaluate_rag_approach.py --approach naive
python evaluate_rag_approach.py --approach advanced
python evaluate_rag_approach.py --approach corrective

# Порівняльна таблиця
python compare_all_approaches.py
```

**Типові результати:**

| Підхід | Faithfulness | Relevancy | Precision | Recall | Average |
|--------|--------------|-----------|-----------|--------|---------|
| **Corrective RAG** | 0.93 | 0.88 | 0.86 | 0.84 | **0.88** ✅ |
| **Advanced RAG** | 0.92 | 0.87 | 0.84 | 0.82 | **0.86** ✅ |
| **Hybrid RAG** | 0.88 | 0.83 | 0.81 | 0.78 | **0.82** ⚠️ |
| **Naive RAG** | 0.68 | 0.65 | 0.62 | 0.58 | **0.63** ❌ |

---

##  Вартість

### Генерація датасету (1 раз):
- 50 запитів: ~$0.10-0.20
- 100 запитів: ~$0.20-0.40

### Evaluation:
- 50 запитів: ~$0.50-1.00
- 100 запитів: ~$1.00-2.00

**Всього для 50 тестів:** ~$0.60-1.20

Порівняйте з вартістю 1 години розробника ($50+) - це **pennies** для якісного тестування!

---

##  Production Checklist

Перед deploy RAG системи переконайтесь:

- [ ] Faithfulness > 0.90 (критично для довіри)
- [ ] Answer Relevancy > 0.85
- [ ] Context Precision > 0.80
- [ ] Context Recall > 0.80
- [ ] Тестовий датасет > 50 запитів
- [ ] Покриття всіх типів запитів (simple/reasoning/multi-context)
- [ ] Evaluation на production-like даних

---

##  Додаткові ресурси

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)
- [LangChain + RAGAS Guide](https://python.langchain.com/docs/guides/evaluation/ragas)
- [RAG Evaluation Best Practices](https://www.rungalileo.io/blog/mastering-rag-how-to-evaluate-retrieval-augmented-generation-systems)

---

##  FAQ

**Q: Чому не можна без RAGAS?**
A: RAGAS дає об'єктивні метрики + автоматичну генерацію тестів.

**Q: Чому потрібен OpenAI API key?**
A: RAGAS використовує LLM для evaluation. Можна використати інші LLM, але OpenAI найпростіше.

**Q: Скільки тестів потрібно?**
A: Мінімум 50 для базового висновку. 100-200 для production. 500+ для enterprise.

**Q: Чи можна evaluation локально (Ollama)?**
A: Так, але RAGAS краще працює з GPT-4/GPT-3.5. Ollama може давати менш точні метрики.

**Q: Як часто робити evaluation?**
A: При кожній значній зміні RAG системи (chunking, embeddings, LLM, тощо).

