# RAG Demos - Порівняння 6 підходів до Retrieval-Augmented Generation

Цей репозиторій містить робочі демонстрації та результати порівняння 6 різних підходів до RAG систем.

## Що знаходиться в репозиторії

### 📊 Готові результати тестування

Всі тести вже виконані на **уніфікованому датасеті з 50 однакових запитів**:

- `results/naive_rag_results.json` - Naive RAG (базовий підхід)
- `results/advanced_rag_results.json` - Advanced RAG (найкраща точність 71.4%)
- `results/bm25_rag_results.json` - BM25 RAG (найшвидший 3.8с)
- `results/faiss_rag_results.json` - FAISS RAG (оптимальний баланс)
- `results/hybrid_rag_results.json` - Hybrid RAG (BM25 + векторний)
- `results/corrective_rag_results.json` - Corrective RAG (самоперевірка)
- **`results/final_comparison_report.txt`** - Детальний звіт з рекомендаціями

### 🚀 Демонстрації (готові до запуску)

```
rag_demos/
├── naive_rag/naive_rag_demo.py          # Базовий підхід
├── advanced_rag/advanced_rag_demo.py    # Покращений (rewriting + reranking)
├── bm25_rag/bm25_rag_demo.py           # Keyword-based пошук
├── faiss_rag/faiss_rag_demo.py         # Векторний індекс Facebook FAISS
├── hybrid_rag/hybrid_rag_demo.py       # Гібридний (BM25 + векторний)
└── corrective_rag/corrective_rag_demo.py # Адаптивна самоперевірка
```

### 📝 Уніфікований датасет

`data/test_queries_unified.json` - 100 стандартизованих запитів:
- 8 категорій (definition, technical, approaches, evaluation, etc.)
- 3 рівні складності (easy, medium, hard)
- Метадані для кожного запиту

### 🛠 Утиліти

- `utils/data_loader.py` - Завантаження документів та датасету
- `compare_all_rag_approaches.py` - Скрипт порівняння (опціональний)
- `run_all_tests.sh` - Автоматичний запуск всіх тестів
- `check_progress.sh` - Моніторинг прогресу виконання

## Швидкий старт

### 1. Встановлення залежностей

```bash
pip install numpy scikit-learn requests pymupdf
```

Опціонально (для FAISS RAG):
```bash
pip install faiss-cpu  # або faiss-gpu для GPU
```

### 2. Підготовка PDF документів

Помістіть ваші PDF файли в `data/pdfs/`

### 3. Налаштування API ключа (для генерації відповідей)

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 4. Запуск окремої демонстрації

```bash
# Naive RAG (базовий)
python naive_rag/naive_rag_demo.py

# Advanced RAG (найкраща точність)
python advanced_rag/advanced_rag_demo.py

# FAISS RAG (оптимальний баланс)
python faiss_rag/faiss_rag_demo.py
```

### 5. Запуск всіх тестів

```bash
# Встановіть API ключ
export OPENAI_API_KEY="your-key"

# Запустіть всі тести послідовно
./run_all_tests.sh

# Або в фоні
./run_all_tests.sh > /tmp/all_tests.log 2>&1 &

# Перевірка прогресу
./check_progress.sh
```

## Результати порівняння

### Основні метрики (50 запитів):

| Підхід | Точність | Швидкість | Рекомендація |
|--------|----------|-----------|--------------|
| **Advanced RAG** | **71.4%** (+251% vs Naive) | 7.39с | Продакшен (найкраща якість) |
| **FAISS RAG** | **66.6%** | **3.88с** | Оптимальний баланс |
| **BM25 RAG** | N/A | **3.80с** | Найшвидший |
| Hybrid RAG | 56.1% | 5.57с | Стабільний |
| Corrective RAG | 21.6% (86% веб) | 4.35с | Самоперевірка |
| Naive RAG | 20.3% (baseline) | 5.28с | Базовий рівень |

### Ключові висновки:

1. **Advanced RAG** - лідер за точністю (+251% vs Naive)
2. **FAISS RAG** - найкращий баланс швидкість/якість
3. **BM25 RAG** - найшвидший (keyword-based, без векторизації)
4. **Corrective RAG** - 86% запитів потребували веб пошуку (локальних документів недостатньо)

## Структура проекту

```
rag_demos/
├── naive_rag/               # Базовий підхід
│   ├── naive_rag_demo.py
│   └── __init__.py
├── advanced_rag/            # Покращений підхід
│   ├── advanced_rag_demo.py
│   └── __init__.py
├── bm25_rag/               # BM25 keyword пошук
├── faiss_rag/              # FAISS векторний пошук
├── hybrid_rag/             # Гібридний підхід
├── corrective_rag/         # Адаптивна самоперевірка
├── utils/                  # Утиліти
│   ├── data_loader.py
│   └── __init__.py
├── data/                   # Дані
│   ├── pdfs/              # PDF документи
│   └── test_queries_unified.json  # Уніфікований датасет
├── results/               # Результати тестів
│   ├── *_results.json
│   └── final_comparison_report.txt
├── run_all_tests.sh       # Запуск всіх тестів
├── check_progress.sh      # Моніторинг прогресу
└── compare_all_rag_approaches.py  # Скрипт порівняння
```

## Використання уніфікованого датасету

```python
from utils.data_loader import DocumentLoader

loader = DocumentLoader()

# Завантаження уніфікованого датасету
queries = loader.load_unified_queries(
    max_queries=50,  # Кількість запитів
    categories=["definition", "technical"]  # Фільтр по категоріях (опційно)
)

# Структура запиту
# {
#   "id": 1,
#   "category": "definition",
#   "question": "What is RAG?",
#   "difficulty": "easy",
#   "expected_concepts": ["retrieval", "generation", "LLM"]
# }
```

## Рекомендації за сценаріями

### Для продакшену (висока якість)
→ **Advanced RAG**
- Точність: 71.4%
- Query rewriting + hybrid search + reranking
- Час: 7.39с (прийнятно)

### Для real-time застосунків (швидкість)
→ **BM25 RAG** або **FAISS RAG**
- BM25: 3.80с (найшвидший)
- FAISS: 3.88с + 66.6% точність
- Для low-latency API (<5 секунд)

### Оптимальний баланс
→ **FAISS RAG**
- На 47% швидше за Advanced
- Лише на 10% нижча точність
- Найкраще співвідношення ціна/якість

### Критичні системи
→ **Corrective RAG**
- Самоперевірка релевантності (3 ітерації)
- Веб fallback при низькій якості
- Для фінансів, медицини, юридичних систем

## Технічні деталі

- **Chunk size**: 500-1000 символів
- **Chunk overlap**: 100-200 символів
- **Top-k**: 3-5 документів
- **LLM**: OpenAI gpt-4o-mini (fallback: Ollama)
- **Embedding**: TF-IDF (baseline) + FAISS (векторний)
- **Загальний час тестування**: ~25 хвилин (300 запитів)

## Документація

- `UNIFIED_DATASET_README.md` - Опис уніфікованого датасету
- `QUICK_START_UNIFIED.md` - Швидкий старт для студентів
- `results/final_comparison_report.txt` - Детальний звіт

## Ліцензія

Навчальний матеріал. Вільне використання в освітніх цілях.

## Автори

Олександр Денисюк - AI.Agents.PRO курс 2024-2025

---

**Примітка**: Всі тести вже виконані, результати знаходяться в `results/`. Для власного тестування просто запустіть відповідний demo скрипт з вашими PDF документами.
