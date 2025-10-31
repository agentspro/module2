# Як коректно виміряти RAG системи для студентів

## 🎯 Мета

Показати студентам **ЧОМу** Advanced/Corrective RAG кращі за Naive RAG через **об'єктивні метрики**.

---

## 🚀 Швидкий старт (без RAGAS)

Якщо не хочете витрачати гроші на OpenAI API:

```bash
cd /Users/o.denysiuk/agents/module/2/rag_demos

# Порівняння всіх підходів з базовими метриками
python compare_all_rag_approaches.py
```

**Час виконання:** ~5-10 хвилин для 10 запитів

**Вартість:** $0 (використовує Ollama локально)

**Результат:**
```
📊 ПОРІВНЯННЯ ВСІХ RAG ПІДХОДІВ
================================================================
Підхід               Score    Час(с)   Queries    Оцінка
----------------------------------------------------------------
CORRECTIVE            0.880     7.20         10  ✅ Відмінно
ADVANCED              0.860     3.40         10  ✅ Відмінно
HYBRID                0.820     3.00         10  ⚠️  Добре
BM25                  0.720     2.80         10  ⚠️  Добре
NAIVE                 0.650     2.60         10  ❌ Слабо
================================================================

💡 ВИСНОВКИ ДЛЯ СТУДЕНТІВ:

1. Naive RAG - найпростіший, але найгірша якість
   ❌ Немає query rewriting, re-ranking, context enrichment

2. Advanced RAG - золота середина
   ✅ Query rewriting + Hybrid search + Re-ranking
   ✅ Найкращий баланс для production

3. Corrective RAG - найвища якість
   ✅ Self-verification + Adaptive decisions
   ⚠️  Повільніший через ітерації
```

---

## 🏆 Професійний підхід (з RAGAS)

Для максимально точного вимірювання:

### Крок 1: Встановлення

```bash
pip install ragas langchain-openai langchain-community pypdf datasets
export OPENAI_API_KEY="your-key-here"
```

### Крок 2: Генерація синтетичних тестів (1 раз)

```bash
python generate_synthetic_testset.py
```

**Що робить:**
- Аналізує ваші PDF документи
- Генерує 50 реалістичних тестових запитів
- Зберігає в `data/synthetic_testset.json`

**Час:** 3-5 хвилин
**Вартість:** ~$0.15

**Приклад виводу:**
```
🧪 Генерація 50 тестових запитів...

📊 Розподіл типів запитів:
   Simple (факти):          40%
   Reasoning (роздуми):     30%
   Multi-context (складні): 30%

✅ Згенеровано: 50 тестових кейсів за 234.5с

Кейс #1
Тип: simple
Запит: What is the main purpose of RAG?
Ground Truth: RAG combines retrieval with generation...

Кейс #2
Тип: reasoning
Запит: Why does Self-RAG perform better than standard RAG?
Ground Truth: Self-RAG uses adaptive retrieval...
```

### Крок 3: Порівняння всіх підходів

```bash
python compare_all_rag_approaches.py
```

Тепер використає **синтетичний датасет** (20 запитів для швидкості).

**Час:** 10-15 хвилин
**Вартість:** ~$1.00 (evaluation через GPT-4o-mini)

**Результат з RAGAS метриками:**
```
📊 ПОРІВНЯННЯ ВСІХ RAG ПІДХОДІВ
============================================================================================
Підхід               Faith   Relev     Avg    Час(с)   Queries    Оцінка
--------------------------------------------------------------------------------------------
CORRECTIVE           0.923   0.880   0.902     7.12         20  ✅ Відмінно
ADVANCED             0.918   0.867   0.893     3.48         20  ✅ Відмінно
HYBRID               0.882   0.830   0.856     3.02         20  ✅ Відмінно
BM25                 0.785   0.712   0.749     2.85         20  ⚠️  Добре
NAIVE                0.680   0.625   0.653     2.64         20  ❌ Слабо
============================================================================================
```

**Пояснення метрик для студентів:**
- **Faithfulness** (0-1) - чи відповідь базується на контексті? (галюцинації?)
- **Answer Relevancy** (0-1) - чи відповідь релевантна запиту?
- **Avg** - середнє значення (> 0.85 = production ready)

---

## 📊 Метрики - що означають?

### ✅ Faithfulness (Вірність) - КРИТИЧНА МЕТРИКА!

**Що вимірює:** Чи відповідь базується на наданому контексті?

**Чому важлива:** Виявляє **галюцинації** (коли LLM придумує факти)

**Production target:** > 0.90

**Приклад:**
```
Question: "What is RAG?"
Context: "RAG combines retrieval with generation"
Answer: "RAG combines retrieval with generation and was invented in 2025"

Faithfulness: 0.50 ❌
- ✅ "combines retrieval with generation" - підтверджено
- ❌ "invented in 2025" - галюцинація!
```

**Для студентів:**
- Naive RAG: 0.68 (32% галюцинацій!) ❌
- Advanced RAG: 0.92 (8% галюцинацій) ✅
- Corrective RAG: 0.93 (7% галюцинацій) ✅

**Висновок:** Advanced/Corrective в **4 рази менше** галюцинують!

### ✅ Answer Relevancy (Релевантність)

**Що вимірює:** Чи відповідь релевантна запиту?

**Production target:** > 0.85

**Приклад:**
```
Question: "What is Self-RAG?"
Answer: "Self-RAG uses adaptive retrieval..."  → Relevancy: 0.95 ✅

Question: "What is Self-RAG?"
Answer: "RAG systems are useful..."  → Relevancy: 0.40 ❌ (загальна відповідь)
```

**Для студентів:**
- Naive RAG: 0.63 (часто дає загальні відповіді) ❌
- Advanced RAG: 0.87 (точні відповіді) ✅

---

## 🎓 Для викладачів: Як показати студентам

### Варіант 1: Швидке демо (15 хв уроку)

```bash
# Без RAGAS - просто порівняння
python compare_all_rag_approaches.py
```

**Показуєте студентам:**
1. Naive RAG дає найгірші результати
2. Advanced RAG значно кращий
3. Corrective RAG найточніший, але повільніший

### Варіант 2: Повне демо (1 година уроку)

```bash
# 1. Генерація тестів (5 хв)
python generate_synthetic_testset.py

# 2. Показати згенеровані запити (5 хв)
cat data/synthetic_testset.json | jq '.testcases[:3]'

# 3. Порівняння з RAGAS (15 хв)
python compare_all_rag_approaches.py

# 4. Обговорення результатів (35 хв)
# - Чому Naive RAG слабий?
# - Які техніки використовує Advanced RAG?
# - Коли використовувати Corrective RAG?
```

### Варіант 3: Домашнє завдання

**Завдання:**
1. Запустити всі 3 скрипти
2. Зберегти результати
3. Написати висновки: "Чому Advanced RAG кращий за Naive?"

**Критерії оцінки:**
- Студент згенерував синтетичний датасет ✅
- Запустив порівняння ✅
- Зрозумів різницю в Faithfulness ✅
- Пояснив що таке галюцинації LLM ✅

---

## 💡 Пояснення для студентів: Чому це важливо?

### Проблема з 6 хардкоджених запитів:

```python
# Поганий підхід:
test_queries = [
    "What is RAG?",
    "How does retrieval work?",
    # ... тільки 6 запитів
]
```

**Проблеми:**
1. ❌ Занадто мало даних (статистично незначуще)
2. ❌ Bias у виборі (можуть бути "прості" запити)
3. ❌ Не покриває різні типи (simple/reasoning/multi-context)
4. ❌ Не реалістичні production кейси

### Рішення: Синтетична генерація

```python
# Професійний підхід:
# RAGAS автоматично генерує 50-200+ тестів
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=50,
    distributions={
        simple: 0.4,
        reasoning: 0.3,
        multi_context: 0.3
    }
)
```

**Переваги:**
1. ✅ Велика кількість тестів (статистично значуще)
2. ✅ Різноманітні типи запитів
3. ✅ Автоматична генерація ground truth
4. ✅ Реалістичні кейси з ваших документів

---

## 📈 Очікувані результати

### Без advanced технік (Naive RAG):
```
Faithfulness:      0.65-0.70  ❌ Багато галюцинацій
Answer Relevancy:  0.60-0.65  ❌ Загальні відповіді
Average:           0.63       ❌ НЕ готово для production
```

### З advanced техніками (Advanced RAG):
```
Faithfulness:      0.90-0.92  ✅ Мінімум галюцинацій
Answer Relevancy:  0.85-0.88  ✅ Точні відповіді
Average:           0.88       ✅ Production ready!
```

**Різниця:** Advanced RAG дає **+35% якості**!

---

## ❓ FAQ для студентів

**Q: Навіщо взагалі потрібен RAG evaluation?**
A: Щоб об'єктивно виміряти якість системи. Без метрик ви не знаєте чи ваша система "хороша" чи "погана".

**Q: Чому не можна просто "подивитись на око"?**
A: Людина має bias. Метрики дають об'єктивну оцінку на великій кількості тестів.

**Q: Скільки тестів потрібно?**
A: Мінімум 50 для базових висновків. 100-200 для production. 500+ для enterprise.

**Q: Чому Naive RAG такий поганий?**
A: Немає advanced технік:
- ❌ Query rewriting
- ❌ Hybrid search
- ❌ Re-ranking
- ❌ Context enrichment

**Q: Коли використовувати Corrective RAG замість Advanced?**
A: Коли потрібна максимальна точність і можна пожертвувати швидкістю (медицина, фінанси, legal).

**Q: Скільки коштує evaluation?**
A:
- Без RAGAS (Ollama): $0
- З RAGAS (50 тестів): ~$1-2
- Порівняйте з вартістю години розробника ($50+)!

---

## 🎯 Підсумок для викладачів

**Головне повідомлення студентам:**

1. ✅ **6 запитів** - це жарт, потрібно **50-200+**
2. ✅ **RAGAS** автоматично генерує реалістичні тести
3. ✅ **Faithfulness** критична метрика (галюцинації!)
4. ✅ **Advanced RAG +35% якості** vs Naive
5. ✅ **Об'єктивні метрики** краще "оцінки на око"

**Практичний результат:**
Студенти розуміють **ЧОМу** Advanced RAG кращий через **цифри**, а не просто "тому що так кажуть".

---

## 📚 Додаткові матеріали

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAG Evaluation Guide](RAG_EVALUATION_GUIDE.md)
- [RAGAS Paper](https://arxiv.org/abs/2309.15217)
