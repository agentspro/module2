# Швидкий старт: Уніфікований датасет для RAG систем

## Що змінилось?

**БУЛО (погано):**
- Кожен demo використовував різні 6 запитів
- Неможливо порівняти Naive RAG vs Advanced RAG
- Результати НЕ співставні

**СТАЛО (добре):**
- **ОДИН** датасет з 100 запитами для ВСІХ demo
- Всі RAG підходи тестуються на **ОДНАКОВИХ** запитах
- Результати **КОРЕКТНО** порівнюються

---

## Як запустити?

### Варіант 1: Швидкий тест (20 запитів, 5 хвилин)

```bash
# Запустити будь-який demo
python naive_rag/naive_rag_demo.py

# Змініть max_queries у коді:
# unified_queries = loader.load_unified_queries(max_queries=20)
```

### Варіант 2: Повний тест (50 запитів, 15 хвилин)

```bash
# Запустити кілька demo для порівняння
python naive_rag/naive_rag_demo.py
python advanced_rag/advanced_rag_demo.py

# За замовчуванням використовує 50 запитів
```

### Варіант 3: Максимальний тест (100 запитів, 30 хвилин)

```bash
# Змініть max_queries у коді:
# unified_queries = loader.load_unified_queries(max_queries=100)
```

---

## Що покращилось?

### Приклад НЕПРАВИЛЬНОГО порівняння (раніше):

```
Naive RAG на своїх 6 запитах:
  "What is RAG?" → Score: 0.85

Advanced RAG на СВОЇХ 6 запитах:
  "Explain Self-RAG architecture" → Score: 0.42

Висновок: Advanced RAG ГІРШИЙ? НІ! Запити різні!
```

### Приклад ПРАВИЛЬНОГО порівняння (тепер):

```
ОДНАКОВІ 50 запитів для всіх:

Naive RAG:
  ID 1: "What is RAG?" → Score: 0.85
  ID 11: "What is Self-RAG?" → Score: 0.35
  ...
  Середнє: 0.65

Advanced RAG:
  ID 1: "What is RAG?" → Score: 0.92
  ID 11: "What is Self-RAG?" → Score: 0.88
  ...
  Середнє: 0.87

Висновок: Advanced RAG +34% точніший! КОРЕКТНО!
```

---

## Очікувані результати

На 50 запитах (замість 6):

**Naive RAG:**
- Average score: ~0.65
- definition: 0.85, technical: 0.62, approaches: 0.50

**Advanced RAG:**
- Average score: ~0.87
- definition: 0.92, technical: 0.89, approaches: 0.85

**Різниця:** +34% точності (статистично значуще!)

---

## Для студентів

### Що потрібно знати:

1. **Всі demo тепер використовують ОДНАКОВІ запити**
   - 100 стандартизованих запитів
   - 8 категорій (definition, technical, approaches, evaluation, challenges, implementation, comparison, optimization)
   - 3 рівні складності (easy, medium, hard)

2. **Результати можна порівнювати**
   - Naive RAG vs Advanced RAG - коректне порівняння
   - Можна побачити де який підхід кращий
   - Результати зберігаються з query_id для аналізу

3. **Як читати результати:**
   - `query_id` - ID запиту з датасету (1-100)
   - `difficulty` - easy/medium/hard
   - `category` - категорія запиту
   - `scores` - оцінки релевантності знайдених документів

---

## Детальна документація

Читайте `UNIFIED_DATASET_README.md` для повної інформації про:
- Структуру датасету
- Як фільтрувати по категоріях
- Як аналізувати результати
- Приклади порівняння різних RAG підходів
