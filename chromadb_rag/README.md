# ChromaDB RAG

Advanced RAG система з використанням ChromaDB для векторного зберігання.

## Основні відмінності від інших підходів

### ChromaDB vs In-Memory (TF-IDF)

**ChromaDB RAG:**
- Persistent storage (зберігає вектори на диску)
- Dense embeddings (Sentence-Transformers: all-MiniLM-L6-v2)
- Швидкий пошук навіть для великих датасетів
- Не потребує повторного індексування після перезапуску

**Naive/Advanced RAG (TF-IDF):**
- In-memory storage (втрачається після перезапуску)
- Sparse vectors (TF-IDF)
- Потребує переіндексації при кожному запуску

### ChromaDB vs FAISS

**ChromaDB:**
- Persistent out-of-the-box
- Вбудована підтримка метаданих
- Проста інтеграція
- Краща для production (автоматичне збереження)

**FAISS:**
- Потребує ручного збереження індексу
- Швидший для дуже великих датасетів (мільйони векторів)
- Більше налаштувань (IndexFlatIP, IndexIVFFlat, etc.)

## Архітектура

```python
ChromaDBRAG
├── ChromaDB PersistentClient      # Persistent storage на диску
├── Sentence-Transformers          # all-MiniLM-L6-v2 embeddings
├── BM25Retriever                  # Keyword search для hybrid
└── Advanced техніки:
    ├── Query Rewriting            # Альтернативні формулювання
    ├── Hybrid Search              # ChromaDB vector + BM25
    ├── Re-ranking                 # Бонуси за релевантність
    └── Context Enrichment         # Сусідні чанки
```

## Встановлення

```bash
# Основні залежності
pip install chromadb sentence-transformers

# Або всі разом
pip install -r requirements.txt
```

## Швидкий старт

```bash
# Встановіть OpenAI API ключ
export OPENAI_API_KEY="your-api-key-here"

# Запустіть демонстрацію
python chromadb_rag/chromadb_rag_demo.py
```

При першому запуску:
1. Створюється директорія `chromadb_storage/`
2. Завантажуються документи з `data/pdfs/`
3. Створюються embeddings та зберігаються в ChromaDB
4. Виконуються тести на 50 запитах

При наступних запусках:
- ChromaDB завантажує існуючі вектори з диску
- Набагато швидший старт (не потрібно переіндексувати)

## Очікувані результати

**Метрики (очікується):**
- Точність: 65-70%
- Швидкість: 3-4 секунди на запит
- Storage: Persistent (зберігається на диску)

**Порівняння з іншими підходами:**

| Підхід | Точність | Швидкість | Storage | Коментар |
|--------|----------|-----------|---------|----------|
| Advanced RAG | 71.4% | 7.39с | In-memory | Найвища точність |
| FAISS RAG | 66.6% | 3.88с | Manual save | Швидкий |
| **ChromaDB RAG** | **~68%** | **~4с** | **Persistent** | **Краще для production** |
| BM25 RAG | N/A | 3.80с | In-memory | Найшвидший |

## Переваги ChromaDB RAG

1. **Persistent Storage**
   - Вектори зберігаються на диску автоматично
   - Не потрібно переіндексувати при перезапуску
   - Готово для production

2. **Dense Embeddings**
   - Sentence-Transformers (all-MiniLM-L6-v2)
   - Краще розуміння семантики ніж TF-IDF
   - Стандарт для RAG систем

3. **Hybrid Search**
   - Комбінує ChromaDB vector + BM25 keyword
   - Кращий recall ніж окремі підходи

4. **Advanced техніки**
   - Query Rewriting - альтернативні формулювання
   - Re-ranking - бонуси за релевантність
   - Context Enrichment - додає сусідні чанки

## Конфігурація

```python
rag = ChromaDBRAG(
    documents_path="data/pdfs",          # Директорія з PDF
    chunk_size=500,                      # Розмір чанку
    chunk_overlap=100,                   # Перекриття чанків
    persist_directory="chromadb_storage" # Де зберігати вектори
)
```

**Параметри hybrid search:**
```python
results = rag.hybrid_search(
    query="Your question",
    top_k=10,     # Скільки документів повернути
    alpha=0.5     # Баланс: 0=BM25, 1=vector
)
```

**Оптимальні значення alpha:**
- `alpha=0.3` - технічна документація (більше keywords)
- `alpha=0.5` - збалансовано (рекомендовано)
- `alpha=0.7` - природна мова (більше semantic)

## Структура даних

**ChromaDB collection:**
```python
{
    "ids": ["chunk_0", "chunk_1", ...],
    "documents": ["text content", ...],
    "embeddings": [[0.1, 0.2, ...], ...],  # 384-dimensional vectors
    "metadatas": [
        {
            "source": "paper.pdf",
            "chunk_index": 0,
            "chunk_id": 0
        },
        ...
    ]
}
```

**Результат запиту:**
```python
{
    "question": "What is RAG?",
    "answer": "RAG stands for...",
    "techniques_used": ["Query Rewriting", "Hybrid Search", ...],
    "relevant_chunks": 5,
    "sources": ["paper1.pdf", "paper2.pdf"],
    "scores": [0.85, 0.78, ...],
    "execution_time": 3.45,
    "storage": "ChromaDB (persistent)"
}
```

## Використання в production

### Перший запуск (індексація):
```python
from chromadb_rag_demo import ChromaDBRAG

# Створюємо систему
rag = ChromaDBRAG(persist_directory="production_storage")

# Завантажуємо документи
rag.load_and_process_documents()

# Створюємо індекси (зберігається автоматично)
rag.create_embeddings()

# Готово! Вектори на диску
```

### Наступні запуски (швидкий старт):
```python
# ChromaDB автоматично завантажує вектори
rag = ChromaDBRAG(persist_directory="production_storage")

# Можна одразу робити запити!
result = rag.query("Your question")
```

### Оновлення документів:
```python
# Перезавантажуємо документи
rag.load_and_process_documents()

# Переіндексуємо (стара collection видаляється автоматично)
rag.create_embeddings()
```

## Порівняння з іншими RAG

### Коли використовувати ChromaDB RAG:

**Використовуйте ChromaDB RAG коли:**
- Потрібен persistent storage (production)
- Документи змінюються рідко
- Важлива швидкість старту
- Потрібна стандартна RAG архітектура

**Використовуйте Advanced RAG (TF-IDF) коли:**
- Потрібна максимальна точність (71.4%)
- In-memory storage прийнятний
- Документи малі (<1000 чанків)

**Використовуйте FAISS RAG коли:**
- Дуже великий датасет (мільйони векторів)
- Потрібна максимальна швидкість пошуку
- Готові налаштовувати FAISS індекси

**Використовуйте BM25 RAG коли:**
- Потрібна максимальна швидкість (3.80с)
- Keyword search достатній
- Точність не критична

## Моніторинг та debug

### Перевірка ChromaDB collection:
```python
print(f"Documents in DB: {rag.collection.count()}")
print(f"First 5 IDs: {rag.collection.get(limit=5)['ids']}")
```

### Перевірка hybrid search:
```python
results = rag.hybrid_search("test query", top_k=5)
for chunk_id, score in results:
    chunk = rag.chunks[chunk_id]
    print(f"Score: {score:.3f} | {chunk['content'][:100]}")
```

### Очистка storage:
```bash
# Видалити persistent storage
rm -rf chromadb_rag/chromadb_storage/
```

## Troubleshooting

### ChromaDB не встановлено
```bash
pip install chromadb sentence-transformers
```

### Помилка "Collection already exists"
```python
# Видаліть стару collection:
rag.chroma_client.delete_collection("rag_documents")

# Або видаліть директорію:
rm -rf chromadb_storage/
```

### Повільний перший запуск
- Нормально! Sentence-Transformers завантажує модель (350MB)
- Наступні запуски швидкі

### Out of memory
```python
# Зменшіть batch_size
batch_size = 50  # замість 100
```

## Подальші покращення

1. **Кращі embeddings:**
   - `all-mpnet-base-v2` (768-dim, точніше)
   - `multi-qa-mpnet-base-dot-v1` (оптимізовано для Q&A)

2. **Реальний reranker:**
   - `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Покращує точність на 5-10%

3. **Metadata filtering:**
   ```python
   results = collection.query(
       query_texts=["question"],
       where={"source": "specific.pdf"}
   )
   ```

4. **Batch queries:**
   ```python
   results = collection.query(
       query_texts=["q1", "q2", "q3"],
       n_results=5
   )
   ```

## Ресурси

- **ChromaDB Docs:** https://docs.trychroma.com/
- **Sentence-Transformers:** https://www.sbert.net/
- **RAG Guide:** https://www.pinecone.io/learn/retrieval-augmented-generation/

---

**Створено для курсу:** AI.Agents.PRO 2025
**Модуль:** 2 - Retrieval-Augmented Generation
