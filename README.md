# Module 2: RAG Systems Comparison

**Курс:** AI.Agents.PRO 2025  
**Модуль:** 2 - Retrieval-Augmented Generation  
**Автор:** Олександр Денисюк  
**Дата:** Жовтень 2025

## Огляд проекту

Цей модуль містить повне порівняння **6 різних підходів до RAG** (Retrieval-Augmented Generation) систем, протестованих на уніфікованому датасеті з 50 однакових запитів. Всі демонстрації готові до запуску, а результати вже згенеровані для аналізу.

## Структура проекту

### 📊 Демонстрації RAG підходів

```
├── naive_rag/               # Базовий RAG
│   └── naive_rag_demo.py   # TF-IDF + cosine similarity
│
├── advanced_rag/            # Покращений RAG (РЕКОМЕНДОВАНО)
│   └── advanced_rag_demo.py # Query rewriting + hybrid search + reranking
│
├── bm25_rag/               # BM25 Keyword Search
│   └── bm25_rag_demo.py    # Найшвидший (3.80с)
│
├── faiss_rag/              # Facebook FAISS Vector Search
│   └── faiss_rag_demo.py   # Оптимальний баланс (3.88с, 66.6%)
│
├── hybrid_rag/             # Гібридний підхід
│   └── hybrid_rag_demo.py  # BM25 + векторний пошук (RRF)
│
└── corrective_rag/         # Адаптивна самоперевірка
    └── corrective_rag_demo.py # 3 ітерації + веб fallback
```

### 📁 Дані та результати

```
├── data/
│   ├── test_queries_unified.json        # Уніфікований датасет (100 запитів)
│   ├── test_queries.json                # Базовий датасет
│   ├── test_queries_with_ground_truth.json  # З правильними відповідями
│   └── pdfs/                            # Директорія для PDF документів
│
└── results/
    ├── naive_rag_results.json           # Naive RAG: 20.3%, 5.28с
    ├── advanced_rag_results.json        # Advanced RAG: 71.4%, 7.39с ⭐
    ├── bm25_rag_results.json           # BM25 RAG: 3.80с (найшвидший)
    ├── faiss_rag_results.json          # FAISS RAG: 66.6%, 3.88с
    ├── hybrid_rag_results.json         # Hybrid RAG: 56.1%, 5.57с
    ├── corrective_rag_results.json     # Corrective RAG: 21.6%, 4.35с
    ├── final_comparison_report.txt     # Детальний звіт порівняння
    └── ragas_comparison.json           # RAGAS оцінки якості
```

### 🛠 Утиліти та скрипти

```
├── compare_all_rag_approaches.py      # Порівняння всіх підходів
├── run_all_tests.sh                   # Автоматичний запуск всіх тестів
├── run_all_demos.py                   # Запуск демонстрацій
├── evaluate_all_with_ragas.py         # Оцінка через RAGAS
├── evaluate_with_synthetic_testset.py # Синтетичний тестовий датасет
├── generate_synthetic_testset.py      # Генерація тестів
└── requirements.txt                   # Залежності проекту
```

## Результати порівняння

### Фінальна таблиця (50 запитів на уніфікованому датасеті):

| Підхід | Точність | Швидкість | Покращення vs Naive | Рекомендація |
|--------|----------|-----------|---------------------|--------------|
| **Advanced RAG** ⭐ | **71.4%** | 7.39с | **+251%** | Продакшен |
| **FAISS RAG** | **66.6%** | **3.88с** | +228% | Оптимальний баланс |
| **BM25 RAG** | N/A* | **3.80с** | - | Найшвидший |
| Hybrid RAG | 56.1% | 5.57с | +176% | Стабільний |
| Corrective RAG | 21.6% | 4.35с | +6% | Самоперевірка** |
| Naive RAG | 20.3% | 5.28с | baseline | Базовий |

\* BM25 використовує іншу шкалу оцінок (keyword-based)  
\** 86% запитів потребували веб пошуку

### Ключові висновки:

1. **Advanced RAG - лідер за точністю**
   - 71.4% точності (в 3.5 рази краще за Naive RAG)
   - Query rewriting покращує розуміння запиту
   - Reranking відфільтровує нерелевантні документи
   - Рекомендовано для продакшену

2. **FAISS RAG - найкращий баланс**
   - На 47% швидше за Advanced RAG
   - Лише на 5% нижча точність
   - Оптимальне співвідношення швидкість/якість
   - Ідеально для real-time застосунків

3. **BM25 RAG - найшвидший**
   - 3.80с - абсолютний рекорд швидкості
   - Keyword-based, без векторизації
   - Відмінно для простих запитів

4. **Corrective RAG - несподіваний результат**
   - Найнижча точність (21.6%)
   - 86% запитів потребували веб пошуку
   - Локальні документи недостатні для відповідей
   - Потребує зовнішнього API

## Швидкий старт

### 1. Встановлення залежностей

```bash
pip install -r requirements.txt
```

Основні залежності:
- `numpy` - математичні операції
- `scikit-learn` - TF-IDF, cosine similarity
- `pymupdf` - парсинг PDF
- `requests` - HTTP запити
- `faiss-cpu` - векторний пошук (опціонально)

### 2. Підготовка даних

Помістіть ваші PDF файли в директорію `data/pdfs/`:

```bash
# Приклад структури
data/pdfs/
├── research_paper_1.pdf
├── research_paper_2.pdf
└── documentation.pdf
```

### 3. Запуск окремої демонстрації

```bash
# Встановіть OpenAI API ключ
export OPENAI_API_KEY="your-api-key-here"

# Advanced RAG (найкраща точність)
python advanced_rag/advanced_rag_demo.py

# FAISS RAG (оптимальний баланс)
python faiss_rag/faiss_rag_demo.py

# BM25 RAG (найшвидший)
python bm25_rag/bm25_rag_demo.py
```

### 4. Запуск всіх тестів

```bash
export OPENAI_API_KEY="your-api-key-here"
./run_all_tests.sh
```

Тести виконуються послідовно, результати зберігаються в `results/`.

### 5. Перегляд результатів

```bash
# Детальний звіт порівняння
cat results/final_comparison_report.txt

# Окремі результати
cat results/advanced_rag_results.json | python -m json.tool | less
```

## Документація

- **`README_EVALUATION.md`** - Методологія оцінювання
- **`RAG_EVALUATION_GUIDE.md`** - Посібник з оцінки RAG систем
- **`results/final_comparison_report.txt`** - Повний звіт тестування

## Технічні деталі

### Конфігурація тестів

- **Датасет:** 50 однакових запитів для всіх підходів
- **Категорії:** definition, technical, approaches, evaluation, challenges, implementation, comparison, optimization
- **Chunk size:** 500-1000 символів
- **Chunk overlap:** 100-200 символів
- **Top-k:** 3-5 документів
- **LLM:** OpenAI gpt-4o-mini (fallback: Ollama llama3.2)
- **Embedding:** TF-IDF (baseline), Sentence Transformers, FAISS (векторний)

### Метрики оцінювання

Для кожного підходу вимірюється:
- **Точність** - cosine similarity між запитом та відповіддю
- **Швидкість** - час виконання запиту (секунди)
- **Релевантність** - якість знайдених документів
- **Повнота** - чи містить відповідь всю необхідну інформацію

RAGAS метрики (опціонально):
- Faithfulness - фактична точність
- Answer Relevancy - релевантність відповіді
- Context Precision - точність пошуку
- Context Recall - повнота контексту

## Рекомендації за сценаріями

### Продакшен (висока якість)
**→ Advanced RAG**
- Точність: 71.4% (+251% vs Naive)
- Query rewriting + hybrid search + reranking
- Час: 7.39с (прийнятно для більшості випадків)

### Real-time (швидкість критична)
**→ FAISS RAG або BM25 RAG**
- FAISS: 3.88с + 66.6% точність
- BM25: 3.80с (найшвидший)
- Для low-latency API (<5 секунд)

### Оптимальний баланс
**→ FAISS RAG**
- На 47% швидше за Advanced
- Лише на 5% нижча точність
- Найкраще співвідношення ціна/якість

### Критичні системи
**→ Corrective RAG** (з обережністю)
- Самоперевірка релевантності (3 ітерації)
- Веб fallback при низькій якості
- УВАГА: потребує достатніх локальних документів

## Для студентів курсу

### Завдання для вивчення:

1. **Аналіз результатів**
   - Порівняйте файли `results/*_results.json`
   - Знайдіть запити, де Advanced RAG працює краще за інші
   - Визначте патерни помилок у Naive RAG

2. **Експерименти**
   - Запустіть тести на ваших PDF документах
   - Порівняйте результати з нашими
   - Спробуйте різні конфігурації (chunk_size, top_k)

3. **Оптимізація**
   - Покращіть Naive RAG додавши query rewriting
   - Експериментуйте з параметрами BM25 (k1, b)
   - Налаштуйте Hybrid RAG alpha параметр

4. **RAGAS Evaluation**
   - Запустіть `evaluate_all_with_ragas.py`
   - Проаналізуйте faithfulness та context precision
   - Порівняйте з нашими метриками

### Рекомендовані кроки:

1. Вивчити код кожного підходу
2. Запустити базовий Naive RAG
3. Порівняти з Advanced RAG
4. Експериментувати з FAISS та BM25
5. Експериментувати з Multimodal RAG
6. Проаналізувати результати RAGAS

## Використані технології

- **Python 3.12+**
- **NumPy** - математичні операції
- **scikit-learn** - TF-IDF, векторизація
- **PyMuPDF (fitz)** - парсинг PDF
- **FAISS** - векторний пошук (Facebook AI)
- **OpenAI API** - генерація відповідей (gpt-4o-mini)
- **Ollama** - локальна LLM (fallback)

## Результати тестування

Всі тести виконані на:
- **Платформа:** Mac M4, 16GB RAM (+Ollama llama3.2)
- **Загальний час:** ~25 хвилин
- **Кількість запитів:** 300 (6 підходів × 50 запитів)
- **LLM:** OpenAI gpt-4o-mini (через експорт OpenAI API ключа)
- **Документи:** 50 PDF файлів, 8,000-9,000 чанків

## Ліцензія

Навчальний матеріал для курсу AI.Agents.PRO 2025.  
Вільне використання в освітніх цілях.

## Контакти

**Курс:** ai.agents.pro  
**Модуль:** 2 - Retrieval-Augmented Generation  
**Репозиторій:** https://github.com/agentspro/module2

---

**Примітка:** 
Всі результати тестування знаходяться в директорії `results/`. 
Детальний звіт доступний у файлі `results/final_comparison_report.txt`.
