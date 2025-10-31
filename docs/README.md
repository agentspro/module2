# RAG Демонстраційні Програми

Колекція самодостатніх демонстраційних програм різних підходів до RAG (Retrieval-Augmented Generation), створених на основі матеріалів з Course.md.

## 📁 Структура Проекту

```
rag_demos/
├── data/
│   ├── corporate_docs/      # Корпоративні документи для тестування
│   │   ├── hr_policy.txt
│   │   ├── it_security.txt
│   │   ├── sales_kpi.txt
│   │   └── equipment_policy.txt
│   └── test_queries.json    # Тестові запити
├── utils/                    # Спільні утиліти
│   ├── __init__.py
│   └── data_loader.py
├── naive_rag/               # Базова RAG
│   └── naive_rag_demo.py
├── advanced_rag/            # Покращена RAG
│   └── advanced_rag_demo.py
├── hybrid_rag/              # Гібридний пошук
│   └── hybrid_rag_demo.py
├── corrective_rag/          # Самоперевірка
│   └── corrective_rag_demo.py
├── results/                 # Результати виконання
├── logs/                    # Логи
├── requirements.txt
└── README.md
```

## 🚀 Швидкий Старт

### 1. Встановлення

```bash
cd rag_demos
python -m pip install -r requirements.txt
```

**Примітка**: Демонстрації використовують спрощені реалізації без зовнішніх API. Для production використання розкоментуйте відповідні залежності в `requirements.txt`.

### 2. Запуск Демонстрацій

#### Naive RAG (Базова Реалізація)
```bash
python naive_rag/naive_rag_demo.py
```
- **Точність**: ~25% на складних запитах
- **Техніки**: TF-IDF embeddings, косинусна подібність
- **Обмеження**: Відсутність контексту, немає верифікації

#### Advanced RAG (Покращена Реалізація)
```bash
python advanced_rag/advanced_rag_demo.py
```
- **Точність**: до 90% на складних запитах
- **Техніки**:
  - Query Rewriting (переписування запитів)
  - HyDE (гіпотетичні документи)
  - Hybrid Search (BM25 + векторний)
  - Re-ranking (перерангування)
  - Context Enrichment (збагачення контексту)

#### Hybrid RAG (Гібридний Пошук)
```bash
python hybrid_rag/hybrid_rag_demo.py
```
- **Техніки**:
  - Dense Search (векторний пошук)
  - Sparse Search (BM25)
  - RRF (Reciprocal Rank Fusion)
  - Convex Combination з параметром α
- **Конфігурації**:
  - Технічні документи: α=0.3
  - Змішаний контент: α=0.5
  - Природна мова: α=0.7

#### Corrective RAG (Самоперевірка)
```bash
python corrective_rag/corrective_rag_demo.py
```
- **Техніки**:
  - Оцінка релевантності документів
  - Адаптивні рішення (generate/web_search/rewrite)
  - Перевірка галюцинацій
  - Верифікація відповідей
  - Ітеративне покращення (до 3 ітерацій)

## 📊 Результати

Результати виконання зберігаються в директорії `results/` у форматі JSON:

- `naive_rag_results.json`
- `advanced_rag_results.json`
- `hybrid_rag_*_results.json`
- `corrective_rag_results.json`

Кожен файл містить:
- Запити та відповіді
- Метрики продуктивності
- Час виконання
- Джерела інформації
- Scores релевантності

## 🔬 Тестові Дані

### Корпоративні Документи
Проект включає 4 корпоративні документи:
1. **HR Policy** - політика управління персоналом (відпустки, робочий час, навчання)
2. **IT Security** - політика інформаційної безпеки (кібербезпека, доступ, інциденти)
3. **Sales KPI** - показники ефективності відділу продажів (Q3 2024)
4. **Equipment Policy** - політика надання обладнання (процедури, підтримка)

### Тестові Запити
Файл `data/test_queries.json` містить 4 категорії запитів:
- **simple_queries**: Прості факти (3 запити)
- **medium_queries**: Запити середньої складності (3 запити)
- **complex_queries**: Складні запити з обчисленнями (3 запити)
- **multi_hop_queries**: Multi-hop міркування (2 запити)

## 📚 RAG Test Datasets (Для Розширення)

### Financial Datasets
1. **ConvFinQA** (Chen et al., 2022)
   - 3,892 розмови з числовим міркуванням
   - Складні фінансові розрахунки

2. **TAT-QA** (Zhu et al., 2021)
   - 16,552 запити з фінансових звітів
   - 2,757 гібридних контекстів
   - Джерело: annualreports.com (2019-2020)

3. **FinanceBench** (Islam et al., 2023)
   - 10,231 запити про публічні компанії
   - 40 компаній США
   - 361 звіт (10Ks, 10Qs, 8Ks)
   - GitHub: 150 triplets публічно, 360 PDF
   - Повний доступ: contact@patronus.ai

### Medical Dataset
4. **BioASQ Challenge** (Tsatsaronis et al., 2015)
   - Біомедична семантична індексація
   - 4,720 прикладів
   - 40,200 параграфів
   - Доступ: HuggingFace subset

### Multi-domain Dataset
5. **ToolQA** (Zhuang et al., 2023)
   - 8 доменів
   - 13 типів інструментів
   - Табличні, графові, текстові дані
   - Фокус на зовнішніх інструментах

### Synthetic Dataset
6. **RepLiQA** (Monteiro et al., 2024)
   - 5 тестових наборів
   - Штучні сценарії (без витоку даних)
   - Людські референсні документи

### Колекції
7. **Hugging Face: rag-datasets**
   - Збірка RAG тестових наборів

## 🛠️ Dataset Generation Frameworks

### 1. RAGEval (Zhu et al., 2024)
- Автоматична генерація датасетів
- Метрики: Completeness, Hallucination, Irrelevance
- ⚠️ Код недоступний

### 2. Giskard (Open-source Python)
- Автоматична генерація 6 типів запитів:
  - Complex questions
  - Conversational questions
  - Distracting questions
  - Double questions
  - Simple questions
  - Situational questions
- Вхід: DataFrame з параграфами
- Оцінка: Performance, bias, security

### 3. RAGAS
- LLM-based framework
- Синтетична генерація датасетів
- Метрики оцінки:
  - Faithfulness (>0.90)
  - Answer Relevancy (>0.85)
  - Context Precision (>0.85)
  - Context Recall (>0.85)

## 🎯 Порівняння Підходів

| Підхід | Точність | Час | Складність | Випадки Використання |
|--------|----------|-----|------------|---------------------|
| **Naive RAG** | ~25% | Швидко | Низька | Прототипи, прості задачі |
| **Advanced RAG** | ~90% | Середньо | Висока | Production, складні запити |
| **Hybrid RAG** | ~85% | Середньо | Середня | Технічна документація |
| **Corrective RAG** | ~90%+ | Повільно | Висока | Критичні системи, верифікація |

## 💡 Рекомендації

### Для Початківців
1. Почніть з Naive RAG для розуміння основ
2. Додайте Hybrid Search - найбільший приріст якості
3. Використовуйте RAGAS для оцінки

### Для Production
1. Advanced RAG з re-ranking обов'язковий
2. Corrective RAG для критичних систем
3. Моніторинг в реальному часі (Phoenix/Langsmith)
4. A/B тестування різних конфігурацій

### Оптимізація Параметрів
- **α для Hybrid Search**:
  - 0.3: технічна документація (точні терміни)
  - 0.5: змішаний контент (баланс)
  - 0.7: природна мова (семантика)

- **Chunk Size**:
  - 300-500: короткі відповіді
  - 500-800: збалансовано
  - 800-1200: довгий контекст

- **Top-k**:
  - 3-5: швидкість
  - 5-10: якість
  - 10-20: повнота (з re-ranking)

## 🔧 Розширення

### Додавання Реальних Embeddings
Розкоментуйте в requirements.txt:
```python
# sentence-transformers
# chromadb
```

Замініть SimpleEmbeddings на:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Додавання LLM Генерації
```python
from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
```

### Використання Векторних БД
```python
from chromadb import Client
client = Client()
collection = client.create_collection("rag_docs")
```

## 📈 Метрики Успіху

### Цільові Показники
- **Faithfulness**: >0.90 (фактична точність)
- **Answer Relevancy**: >0.85 (релевантність)
- **Context Precision**: >0.85 (точність контексту)
- **Context Recall**: >0.85 (повнота контексту)
- **Latency**: <3 секунди
- **Cost**: <$0.001 за запит

## 🤝 Contribution

Для додавання нових підходів:
1. Створіть нову директорію `{approach}_rag/`
2. Імплементуйте `{approach}_rag_demo.py`
3. Додайте тестові запити в `data/test_queries.json`
4. Оновіть README.md

## 📝 Ліцензія

MIT License - вільне використання для навчання та комерційних проектів.

## 🔗 Посилання

- **Course.md**: Повний теоретичний матеріал
- **CLAUDE.md**: Інструкції для Claude Code
- **RAG Paper**: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- **RAGAS**: https://github.com/explodinggradients/ragas
- **Giskard**: https://github.com/Giskard-AI/giskard
- **HuggingFace rag-datasets**: https://huggingface.co/collections/rag-datasets

## ⚠️ Примітки

1. **Спрощені Реалізації**: Ці демо використовують TF-IDF замість нейронних embeddings для незалежності від зовнішніх API
2. **Без LLM**: Генерація відповідей спрощена (повертає релевантні фрагменти)
3. **Production**: Для production додайте real embeddings, LLM, та векторні БД
4. **Тестові Дані**: Використовуйте власні документи або завантажте датасети вище

## 🎓 Навчальні Цілі

Після роботи з цими демо ви зрозумієте:
- ✅ Базові принципи RAG
- ✅ Різницю між підходами
- ✅ Query rewriting та HyDE
- ✅ Гібридний пошук (dense + sparse)
- ✅ Re-ranking та context enrichment
- ✅ Самоперевірку та ітеративне покращення
- ✅ Метрики оцінки RAG систем

Успіхів у вивченні RAG! 🚀
