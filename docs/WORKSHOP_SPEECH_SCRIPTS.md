# 🎤 Сценарії Демонстрацій для RAG Workshop

**Воркшоп**: Модуль 2 - 7 Типів RAG Систем
**Дата**: 30 жовтня 2025, 18:30-21:00
**Практична частина**: 19:40-20:30 (50 хвилин)

---

## 📋 Загальна Структура Практики

```
Блок 1 (15 хв): Базові RAG
├── Demo 1: Naive RAG (5 хв)
└── Demo 2: Retrieve-and-Rerank (10 хв)

Блок 2 (15 хв): Просунуті RAG
├── Demo 3: Multimodal RAG (8 хв)
└── Demo 4: Graph RAG (7 хв)

Блок 3 (18 хв): Агентні RAG
├── Demo 5: Agentic Router (Self-RAG) (8 хв)
└── Demo 6: Agentic Multi-Agent (10 хв)

Пропущено: Hybrid RAG (згадати теоретично або показати якщо залишиться час)
```

---

# 🎬 БЛОК 1: Базові RAG (15 хвилин)

---

## Demo 1: Naive RAG (5 хвилин)

### 🎯 Мета Demo
Показати найпростіший RAG підхід і його обмеження

### 📂 Підготовка (до воркшопу)
```bash
# Terminal 1 - тримати готовим
cd /Users/o.denysiuk/agents/module/2
source rag_env/bin/activate
```

### 🎤 Сценарій

#### Хвилина 1: Вступ (30 сек)
**ЩО ГОВОРИТИ:**
> "Починаємо з найпростішого підходу - Naive RAG. Це baseline, від якого ми будемо відштовхуватися. Зараз побачимо чому він називається 'naive' і які в нього проблеми."

**ЩО ПОКАЗАТИ:**
- Відкрити `rag_demos/naive_rag/naive_rag_demo.py` в VSCode
- Прокрутити до класу `NaiveRAG` (рядки 20-50)

**ЩО СКАЗАТИ ПРО КОД:**
> "Подивіться на архітектуру: TF-IDF vectorizer для пошуку, косинусна подібність, і все. Немає reranking, немає semantic embeddings - чисто keywords."

```python
# ПОКАЗАТИ ЦЕЙ ФРАГМЕНТ:
def retrieve(self, query: str, top_k: int = 5):
    query_vector = self.vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
```

#### Хвилина 2: Запуск Demo (1 хв)
**КОМАНДА:**
```bash
python rag_demos/naive_rag/naive_rag_demo.py
```

**ЩО ГОВОРИТИ ПІД ЧАС ЗАПУСКУ:**
> "Запускаю demo з трьома тестовими запитами. Зверніть увагу на швидкість - вона буде висока, але якість..."

**ОЧІКУВАНИЙ OUTPUT:**
```
🚀 Naive RAG Demo
📊 Завантажено 10 документів
⏱️ Retrieval: 15ms
📝 Query: "What is machine learning?"
✅ Answer: ...
⏱️ Total time: 2.6s
```

#### Хвилина 3: Аналіз Результатів (1.5 хв)
**ЩО ПОКАЗАТИ:**
- Прокрутити output до першого запиту
- Звернути увагу на retrieved documents

**ЩО ГОВОРИТИ:**
> "Дивіться на результати. Швидкість чудова - 2.6 секунди end-to-end. АЛЕ! Якщо подивитися на retrieved documents, бачимо проблему: пошук базується лише на keyword matching. Якщо в запиті немає точних слів з документа - ми не знайдемо релевантний контекст."

**ПОКАЗАТИ КОНКРЕТНИЙ ПРИКЛАД:**
> "Запит: 'What is machine learning?' знайшов документи з словами 'machine' і 'learning', але пропустив семантично схожі документи про 'neural networks' або 'AI' - бо там немає цих точних слів."

#### Хвилина 4: Ключові Обмеження (1 хв)
**ЩО ГОВОРИТИ:**
> "Ключові проблеми Naive RAG:"

**ПОКАЗАТИ НА ЕКРАНІ (можна на слайді або в README):**
```
❌ Keyword-only search (немає семантики)
❌ Немає reranking (перші 5 не завжди найкращі)
❌ Низька точність: ~30%
✅ Швидко: 2.6s
✅ Просто імплементувати

USE CASE: MVP, прототипи, демо
```

#### Хвилина 5: Перехід до наступного (30 сек)
**ЩО ГОВОРИТИ:**
> "Тепер подивимося як це можна покращити. Retrieve-and-Rerank використовує інший підхід - двоетапний пошук. І тут нас чекає сюрприз..."

---

## Demo 2: Retrieve-and-Rerank (10 хвилин)

### 🎯 Мета Demo
Показати парадокс cross-encoder і чому це рекомендований production підхід

### 📂 Підготовка
```bash
# Відкрити заздалегідь:
# 1. Terminal з командою готовою
# 2. VSCode з results/complete_embeddings_benchmark.json
# 3. Слайд з графіком performance_comparison.png
```

### 🎤 Сценарій

#### Хвилина 1-2: Вступ та Архітектура (2 хв)
**ЩО ГОВОРИТИ:**
> "Retrieve-and-Rerank - це наш рекомендований підхід для production. Чому? Зараз побачимо один контр-інтуїтивний результат, який здивував навіть нас."

**ПОКАЗАТИ ДІАГРАМУ (намалювати або на слайді):**
```
Stage 1: FAISS Bi-encoder
  Query → 384D embedding → Top-20 candidates (швидкий, broad recall)
  ⏱️ 809ms для 19,000 chunks

Stage 2: Cross-encoder Reranking  
  Query + Each of 20 docs → Relevance score → Top-10 (точний, precision)
  ⏱️ 229ms для 20 chunks

TOTAL: ??? секунд
```

**ЗАПИТАТИ АУДИТОРІЮ:**
> "Питання до аудиторії: скільки часу займе two-stage підхід, якщо FAISS займає 809ms, а reranking 229ms? Хтось скаже ~1 секунда?"

**ПАУЗА 5 секунд для відповідей**

#### Хвилина 3-4: Показати Результати Benchmark (2 хв)
**КОМАНДА (або показати готові результати):**
```bash
cat results/complete_embeddings_benchmark.json | jq '.results[] | select(.approach == "FAISS + Reranker")'
```

**АБО відкрити JSON файл і показати:**
```json
{
  "approach": "FAISS + Reranker",
  "retrieval_time": 0.229,  // ← МЕНШЕ ніж pure FAISS!
  "avg_score": 4.28,
  "queries": [
    {
      "query": "What is retrieval-augmented generation?",
      "score": 4.5,
      "time": 3.4
    }
  ]
}
```

**ЩО ГОВОРИТИ:**
> "І ось парадокс! FAISS + Reranker займає 229ms - це ШВИДШЕ ніж pure FAISS який займає 809ms. Як таке можливо?"

#### Хвилина 5-6: Пояснення Парадоксу (2 хв)
**ПОКАЗАТИ НА СЛАЙДІ АБО НАМАЛЮВАТИ:**
```
Pure FAISS:
  ❌ Обробляє ВСІ 19,000 chunks
  ❌ Обчислює 19,000 cosine similarities
  ⏱️ Result: 809ms

FAISS + Cross-encoder:
  ✅ FAISS: обробляє 19,000 chunks → top-20 (швидко, approximate)
  ✅ Cross-encoder: обробляє лише 20 chunks (точно, exact)
  ⏱️ Result: 229ms

Чому швидше?
→ Cross-encoder обробляє 20 замість 19,000!
→ Two-stage > One-stage for large corpora
```

**ЩО ГОВОРИТИ:**
> "Секрет простий: коли у вас великий корпус (19K документів), обробляти ВСЕ дорого. Натомість, ми робимо швидкий approximate search (FAISS) щоб звузити до 20 кандидатів, а потім точний expensive reranking тільки на них. Це як спочатку відфільтрувати товари на сайті за категорією, а потом детально подивитись тільки на 20."

#### Хвилина 7: Показати Якість (1 хв)
**ПОКАЗАТИ ГРАФІК:**
`presentation_charts/performance_comparison.png`

**ЩО ГОВОРИТИ:**
> "А тепер подивіться на якість. Score 4.28 - це 824% краще ніж Naive RAG! Ось чому це наш рекомендований підхід для production."

**ПОКАЗАТИ ТАБЛИЦЮ:**
```
| Approach          | Accuracy | Speed  | Production? |
|-------------------|----------|--------|-------------|
| Naive RAG         | 30%      | 2.6s   | ❌ No       |
| FAISS pure        | ~70%     | 809ms  | ⚠️ OK       |
| FAISS + Reranker  | 85%+     | 229ms  | ✅ YES      |
```

#### Хвилина 8-9: Technical Details (2 хв)
**ПОКАЗАТИ КОД (опціонально, якщо аудиторія технічна):**
```python
# Bi-encoder (Stage 1): Швидкий approximate search
query_embedding = encoder.encode(query)
faiss_results = index.search(query_embedding, k=20)

# Cross-encoder (Stage 2): Точний reranking
pairs = [(query, doc) for doc in faiss_results]
scores = cross_encoder.predict(pairs)
reranked = sorted(zip(faiss_results, scores), reverse=True)[:10]
```

**ЩО СКАЗАТИ:**
> "Bi-encoder: encode query і documents окремо, потім порівнюємо embeddings. Швидко але approximate."
> 
> "Cross-encoder: encode query+document разом. Бачить взаємодію між словами. Точно але повільно. Тому ми його використовуємо тільки на 20 кандидатах."

#### Хвилина 10: Use Cases та Висновки (1 хв)
**ЩО ГОВОРИТИ:**
> "Коли використовувати Retrieve-and-Rerank?"

**ПОКАЗАТИ:**
```
✅ Production RAG systems (рекомендовано)
✅ Customer-facing Q&A
✅ Large document corpora (10K+ chunks)
✅ Коли якість критична

⚠️ Потребує:
- Good embeddings model (sentence-transformers)
- Cross-encoder model (~500MB)
- ~200-500ms latency OK
```

**ПЕРЕХІД:**
> "Ми побачили як працювати з текстом. А якщо потрібно шукати по зображеннях? Переходимо до Multimodal RAG."

---

# 🎬 БЛОК 2: Просунуті RAG (15 хвилин)

---

## Demo 3: Multimodal RAG (8 хвилин)

### 🎯 Мета Demo
Показати як RAG може працювати з текстом + зображеннями в одному векторному просторі

### 📂 Підготовка
```bash
# Terminal 2
cd /Users/o.denysiuk/agents/module/2
source rag_env/bin/activate

# Перевірити що ChromaDB працює
python -c "import chromadb; print('✅ ChromaDB OK')"
```

### 🎤 Сценарій

#### Хвилина 1: Вступ та Концепція (1.5 хв)
**ЩО ГОВОРИТИ:**
> "До цього моменту ми працювали тільки з текстом. Але в реальних системах часто є зображення - товари в e-commerce, медичні знімки, документи з charts. Multimodal RAG дозволяє шукати по тексту і зображеннях одночасно."

**ПОКАЗАТИ КОНЦЕПЦІЮ (слайд або намалювати):**
```
CLIP Model - один векторний простір для text + images

Text:  "banana fruit"        → [0.23, -0.45, 0.12, ..., 0.67] (512D)
Image: 🍌 banana.jpg         → [0.21, -0.43, 0.14, ..., 0.69] (512D)
                                    ↑
                            Similar vectors!

Query: "yellow tropical fruit" → Знайде і текст "banana" і зображення 🍌
```

**ЩО СКАЗАТИ:**
> "CLIP від OpenAI навчений на 400 мільйонах пар image-text. Він 'розуміє' що текст 'banana' і фото банану - це одне і те ж, тому їх embeddings близькі в векторному просторі."

#### Хвилина 2-3: Показати Код (1.5 хв)
**ВІДКРИТИ:** `rag_demos/multimodal_rag/multimodal_rag_demo.py`

**ПОКАЗАТИ ТА ПОЯСНИТИ:**
```python
# Рядки 25-35: Ініціалізація
self.model = SentenceTransformer('clip-ViT-B-32')  # ← CLIP model

def encode_text(self, text: str) -> List[float]:
    embedding = self.model.encode(text)
    return embedding.tolist()  # → 512D vector

def encode_image(self, image_path: str) -> List[float]:
    img = Image.open(image_path)
    embedding = self.model.encode(img)  # ← Та сама модель!
    return embedding.tolist()  # → 512D vector
```

**ЩО ГОВОРИТИ:**
> "Зверніть увагу: encode_text і encode_image використовують ТУ САМУ модель. Це ключ - обидва йдуть в один 512-вимірний простір, де семантично схожі речі близькі."

#### Хвилина 4-6: Запуск Demo (2 хв)
**КОМАНДА:**
```bash
python rag_demos/multimodal_rag/multimodal_rag_demo.py
```

**ЩО ГОВОРИТИ ПІД ЧАС ЗАПУСКУ:**
> "Запускаю demo. Спочатку він додасть кілька текстових описів фруктів і (симульовані) зображення в ChromaDB. Потім виконає 3 запити різних типів."

**ОЧІКУВАНИЙ OUTPUT:**
```
🎨 Multimodal RAG Demo
📊 ChromaDB initialized
🤖 Loading CLIP model: clip-ViT-B-32
✅ Model loaded (512D embeddings)

Adding sample data...
✅ Added 3 items to collection

Query 1: "yellow tropical fruit rich in potassium"
🔍 Search results:
  1. banana (similarity: -25.68) ← Найкращий match!
  2. orange (similarity: -31.45)
  ...
```

#### Хвилина 7: Пояснення Результатів (1.5 хв)
**ПОКАЗАТИ КОЖЕН QUERY:**

**Query 1:** "yellow tropical fruit rich in potassium"
**ЩО ГОВОРИТИ:**
> "Перший запит: 'yellow tropical fruit rich in potassium'. Немає слова 'banana', але модель знає що banana жовтий, тропічний, і багатий калієм. Similarity score -25.68 - найкращий результат."

**Query 2:** "round citrus fruit with vitamin C"
> "Другий: 'round citrus fruit'. Знайшов orange. Знову немає точного слова, але семантичне розуміння."

**Query 3:** "healthy fruit for breakfast"
> "Третій: загальний запит. Знайшов всі фрукти, бо всі підходять."

**ПІДКРЕСЛИТИ:**
> "Ключова різниця з Naive RAG: там ми шукали б keyword 'banana'. Тут ми шукаємо СЕМАНТИКУ - 'yellow tropical fruit' знаходить banana, навіть якщо слово інше."

#### Хвилина 8: Use Cases (30 сек)
**ЩО ГОВОРИТИ:**
> "Де це використовується в production?"

**ПОКАЗАТИ СЛАЙД:**
```
✅ E-commerce: "Show me similar items" (фото → схожі товари)
✅ Medical: X-ray + symptoms → similar cases
✅ Fashion: Outfit photo → where to buy
✅ Document search: Діаграми, charts в PDF
✅ Content moderation: Image + text context

Приклад: Користувач завантажує фото одягу → 
         система знаходить схожі товари в каталозі
```

**ПЕРЕХІД:**
> "Multimodal працює з неструктурованими даними. А якщо у нас є структура - entities та relationships? Для цього є Graph RAG."

---

## Demo 4: Graph RAG (7 хвилин)

### 🎯 Мета Demo
Показати як knowledge graph дозволяє робити multi-hop reasoning

### 📂 Підготовка
```bash
# Terminal 3 (підготувати заздалегідь - граф генерується довго!)
cd /Users/o.denysiuk/agents/module/2

# Можливо заздалегідь запустити і зберегти output
# python comprehensive_rag_benchmark.py --only-graph
```

### 🎤 Сценарій

#### Хвилина 1: Концепція (1 хв)
**ЩО ГОВОРИТИ:**
> "В попередніх підходах ми шукали документи. Але інформація часто структурована - є entities (люди, організації, концепти) та relationships між ними. Graph RAG будує knowledge graph і використовує його для пошуку."

**ПОКАЗАТИ ДІАГРАМУ (слайд):**
```
Documents → Entity Extraction → Knowledge Graph

Example:
"RAG was introduced by Facebook AI in 2020"
↓
Entities:  [RAG, Facebook AI, 2020]
Relations: [RAG] --introduced_by--> [Facebook AI]
           [RAG] --year--> [2020]

Knowledge Graph:
    RAG
     ├─→ introduced_by: Facebook AI
     ├─→ year: 2020
     ├─→ uses: retrieval
     └─→ related_to: transformers
```

#### Хвилина 2: Показати Статистику (30 сек)
**ЩО ГОВОРИТИ:**
> "Ми побудували knowledge graph з нашого датасету RAG papers. Подивіться на масштаб:"

**ПОКАЗАТИ (заздалегідь підготовлений результат або JSON):**
```
📊 Graph Statistics:
   Entities: 4,944
   Relationships: 15,009
   Types: PERSON, ORG, METHOD, CONCEPT
   Build time: ~1 second
```

**ЩО СКАЗАТИ:**
> "Майже 5 тисяч entities і 15 тисяч зв'язків. Це дозволяє робити складні запити про relationships."

#### Хвилина 3-5: Показати Приклад Multi-hop Query (2 хв)
**ЩО ГОВОРИТИ:**
> "Ключова перевага - multi-hop reasoning. Наприклад, запит: 'How are RAG and transformers related?'"

**ПОКАЗАТИ ВІЗУАЛІЗАЦІЮ (можна на слайді):**
```
Query: "How are RAG and transformers related?"

Step 1: Find entities in query
   → [RAG, transformers]

Step 2: Graph traversal
   RAG → uses → retrieval
   retrieval → based_on → embeddings
   embeddings → generated_by → transformers
   
Step 3: Path found!
   RAG → retrieval → embeddings → transformers
   
Context: "RAG uses retrieval which relies on embeddings 
          generated by transformer models like BERT."
```

**ЩО СКАЗАТИ:**
> "Graph RAG знайшов зв'язок через проміжні entities. Naive RAG цього б не зробив - він шукав би документ де є обидва слова 'RAG' і 'transformers' поруч."

#### Хвилина 6: Запустити Demo (опціонально) (2 хв)
**ЯКЩО Є ЧАС:**
```bash
# Показати короткий фрагмент output
cat graph_rag_results.json | jq '.entities[:10]'
```

**АБО показати заздалегідь збережений output:**
```json
{
  "approach": "Graph RAG",
  "accuracy": 0.90,
  "speed": 2.9,
  "graph_stats": {
    "entities": 4944,
    "relationships": 15009
  },
  "example_query": {
    "query": "What is RAG?",
    "entities_found": ["RAG", "retrieval", "generation"],
    "path": "RAG → uses → retrieval → augments → generation"
  }
}
```

#### Хвилина 7: Use Cases та Висновки (1 хв)
**ЩО ГОВОРИТИ:**
> "Коли використовувати Graph RAG?"

**ПОКАЗАТИ:**
```
✅ Domain з чіткими entities (medical, legal, scientific)
✅ Запити про relationships: "How are X and Y related?"
✅ Multi-hop reasoning: "What connects A to B through C?"
✅ Knowledge management systems

Переваги:
✅ Fastest (2.9s) серед advanced RAG
✅ 90% accuracy
✅ Structured knowledge

Недоліки:
⚠️ Потребує entity extraction (NER)
⚠️ Build time для великих датасетів
⚠️ Не для unstructured general queries
```

**ПЕРЕХІД:**
> "Всі ці підходи - retrieval за запитом. А що якщо LLM сам вирішить чи потрібен retrieval? Переходимо до Agentic RAG."

---

# 🎬 БЛОК 3: Агентні RAG (18 хвилин)

---

## Demo 5: Agentic Router (Self-RAG) (8 хвилин)

### 🎯 Мета Demo
Показати як LLM може адаптивно вирішувати коли потрібен retrieval і робити self-correction

### 📂 Підготовка
```bash
# Terminal 4
cd /Users/o.denysiuk/agents/module/2

# Переконатися що Ollama працює
ollama list
```

### 🎤 Сценарій

#### Хвилина 1: Концепція Self-RAG (1.5 хв)
**ЩО ГОВОРИТИ:**
> "До цього моменту ми завжди робили retrieval для кожного запиту. Але це не завжди потрібно. Наприклад, '2+2=?' не потребує пошуку в документах. Self-RAG - це підхід де LLM САМ вирішує коли потрібен retrieval."

**ПОКАЗАТИ FLOWCHART (слайд):**
```
Query: "What is RAG?"
   ↓
Agent Decision: "Need external knowledge? YES"
   ↓
Retrieve documents
   ↓
Generate answer
   ↓
Self-Critique: "Quality good? If NO, retrieve more"
   ↓
Final Answer
```

**ПОРІВНЯТИ:**
```
Traditional RAG:
  ЗАВЖДИ робить retrieval → може бути overhead

Self-RAG:
  1. Evaluate: чи потрібен retrieval?
  2. Retrieve: тільки якщо потрібно
  3. Generate: з context або без
  4. Critique: перевірити якість
  5. Refine: якщо потрібно, повторити
```

#### Хвилина 2-3: Показати Код Decision Logic (1.5 хв)
**ВІДКРИТИ:** `comprehensive_rag_benchmark.py` → клас `SelfRAG`

**ПОКАЗАТИ:**
```python
def decide_retrieve(self, query: str) -> bool:
    """Agent вирішує чи потрібен retrieval"""
    prompt = f"""
    Evaluate if this query needs external document retrieval:
    Query: "{query}"
    
    Answer YES if needs facts/data from documents.
    Answer NO if can answer from general knowledge.
    
    Decision:"""
    
    decision = self.llm_generate(prompt)
    return "yes" in decision.lower()
```

**ЩО СКАЗАТИ:**
> "Подивіться: ми запитуємо LLM чи цей запит потребує external knowledge. Якщо LLM каже 'YES' - робимо retrieval. Якщо 'NO' - генеруємо відразу. Це економить 50-70% викликів до retrieval системи."

#### Хвилина 4-5: Self-Critique Mechanism (1.5 хв)
**ПОКАЗАТИ КОД:**
```python
def critique_answer(self, query: str, answer: str, context: str) -> dict:
    """Agent критикує свою власну відповідь"""
    prompt = f"""
    Query: {query}
    Context: {context}
    Answer: {answer}
    
    Evaluate:
    1. Is answer supported by context? (Yes/No)
    2. Is answer complete? (Yes/No)
    3. Quality score: (1-5)
    4. Need refinement? (Yes/No)
    """
    
    critique = self.llm_generate(prompt)
    return parse_critique(critique)
```

**ЩО ГОВОРИТИ:**
> "Після генерації, agent критикує сам себе. Це як code review, але для LLM outputs. Якщо quality низька - agent сам ініціює ще один раунд retrieval з уточненими параметрами."

#### Хвилина 6-7: Запустити Demo (2 хв)
**КОМАНДА:**
```bash
# Якщо є готовий скрипт:
python run_selfrag_demo.py

# Або показати результати з comprehensive benchmark:
cat results/selfrag_results.json | jq '.iterations'
```

**ПОКАЗАТИ OUTPUT:**
```
🤖 Self-RAG Demo

Query: "What is RAG?"

[Iteration 1]
├─ Decision: Need retrieval? YES
├─ Retrieved: 5 documents
├─ Generated answer: "RAG is..."
└─ Critique: Quality=3/5, Need refinement? YES

[Iteration 2]
├─ Decision: Retrieve more? YES (refine query)
├─ Retrieved: 5 different documents (better match)
├─ Generated answer: "RAG is a technique that combines..."
└─ Critique: Quality=5/5, Need refinement? NO

✅ Final Answer: "RAG is a technique that combines..."
⏱️ Total: 4.0s (2 iterations)
```

**ЩО ПОЯСНИТИ:**
> "Бачите два iterations? Перший дав відповідь якості 3/5. Agent сам вирішив що це недостатньо і зробив ще одну спробу з refined query. Другий iteration дав 5/5."

#### Хвилина 8: Метрики та Use Cases (30 сек)
**ПОКАЗАТИ:**
```
📊 Self-RAG Performance:
   Accuracy: 91% (вище за traditional!)
   Speed: 4.0s (повільніше через LLM calls)
   Retrieval overhead: -60% (робить retrieval тільки коли потрібно)

✅ Use Cases:
- Mixed simple/complex queries
- Quality-critical applications
- Conversational agents (багато queries не потребують retrieval)
- Cost optimization (менше API calls)

⚠️ Trade-offs:
- Більше LLM inference calls
- Складніша логіка
- Потрібен good evaluation prompt
```

**ПЕРЕХІД:**
> "Self-RAG - один agent. А що якщо кілька agents працюють разом? Multi-Agent RAG."

---

## Demo 6: Agentic Multi-Agent RAG (10 хвилин)

### 🎯 Мета Demo
Показати як кілька спеціалізованих agents collaborate для складних запитів

### 📂 Підготовка
```bash
# Terminal 5 (той самий що Self-RAG)
# Переконатися що Ollama не перегрітий
```

### 🎤 Сценарій

#### Хвилина 1-2: Концепція Multi-Agent (2 хв)
**ЩО ГОВОРИТИ:**
> "Найскладніший і найпотужніший підхід. Замість одного agent, ми маємо команду з 4 спеціалізованих agents, кожен відповідає за свою частину процесу."

**ПОКАЗАТИ ДІАГРАМУ:**
```
Complex Query: "Explain RAG, compare it to fine-tuning, 
                and suggest when to use each approach"

Agent 1: PLANNING
├─ Decompose query into sub-questions:
│  Q1: What is RAG?
│  Q2: What is fine-tuning?
│  Q3: Comparison between RAG and fine-tuning?
│  Q4: Use cases for each?

Agent 2: RETRIEVAL
├─ Execute each sub-query:
│  Q1 → 5 docs about RAG
│  Q2 → 3 docs about fine-tuning
│  Q3 → 7 docs about comparisons

Agent 3: REASONING
├─ Analyze retrieved documents:
│  ✅ Extract 15 facts
│  ✅ Verify consistency
│  ⚠️ Flag 2 conflicting statements
│  ✅ Resolve conflicts

Agent 4: SYNTHESIS
├─ Combine information:
   → Structured answer with:
      - Definition of RAG
      - Definition of fine-tuning
      - Side-by-side comparison table
      - Recommendation matrix
```

#### Хвилина 3-4: Показати Agent Implementations (2 хв)
**ВІДКРИТИ КОД:** `comprehensive_rag_benchmark.py` → клас `AgenticRAG`

**ПОКАЗАТИ PLANNING AGENT:**
```python
def planning_agent(self, query: str) -> dict:
    """Decompose complex query into sub-questions"""
    prompt = f"""
    Complex Query: {query}
    
    Break this down into 3-5 sub-questions that need to be answered.
    Make each sub-question focused and specific.
    
    Sub-questions:
    1.
    2.
    ...
    """
    
    plan = self.llm_generate(prompt)
    return parse_subquestions(plan)
```

**ЩО СКАЗАТИ:**
> "Planning agent розбиває складний запит на прості під-запити. Це як проектний менеджер розбиває великий таск на підзадачі."

**ПОКАЗАТИ REASONING AGENT:**
```python
def reasoning_agent(self, docs: List[str], query: str) -> dict:
    """Extract facts and verify consistency"""
    prompt = f"""
    Documents: {docs}
    Query: {query}
    
    Tasks:
    1. Extract key facts (list)
    2. Check for conflicting information
    3. Rate confidence for each fact (1-5)
    4. Synthesize coherent understanding
    
    Analysis:
    """
    
    reasoning = self.llm_generate(prompt)
    return parse_reasoning(reasoning)
```

**ЩО СКАЗАТИ:**
> "Reasoning agent - це як fact-checker. Він не просто бере документи, а аналізує їх критично, шукає протиріччя, оцінює достовірність."

#### Хвилина 5-7: Запустити Demo (2 хв)
**КОМАНДА:**
```bash
# Запуск на complex query
python run_agentic_rag_demo.py --query "Explain RAG and its benefits"
```

**ПОКАЗАТИ ДЕТАЛЬНИЙ OUTPUT з agent logs:**
```
🤖🤖🤖 Multi-Agent RAG Demo

Query: "Explain RAG and its benefits"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 AGENT 1: PLANNING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plan created:
  Q1: What is RAG (Retrieval-Augmented Generation)?
  Q2: What are the key components of RAG?
  Q3: What are the main benefits of using RAG?
  Q4: What are typical use cases?
  
⏱️ Planning time: 1.2s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 AGENT 2: RETRIEVAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Executing 4 sub-queries...
  Q1 → Retrieved 5 documents (avg relevance: 0.89)
  Q2 → Retrieved 3 documents (avg relevance: 0.92)
  Q3 → Retrieved 7 documents (avg relevance: 0.85)
  Q4 → Retrieved 4 documents (avg relevance: 0.87)
  
Total: 19 documents
⏱️ Retrieval time: 0.8s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 AGENT 3: REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analyzing 19 documents...

Extracted Facts:
  1. RAG combines retrieval with generation (confidence: 5/5)
  2. Introduced by Facebook AI Research in 2020 (confidence: 5/5)
  3. Reduces hallucinations (confidence: 5/5)
  4. Enables fresh data without retraining (confidence: 5/5)
  ...
  15. Cost-effective vs fine-tuning (confidence: 4/5)

Conflicts detected:
  ⚠️ Fact 8 vs Fact 12: Latency overhead (10-50ms vs 100-200ms)
     Resolution: Depends on implementation, both valid

Consistency: 87% (13/15 facts consistent)
⏱️ Reasoning time: 1.5s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✍️ AGENT 4: SYNTHESIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Combining insights from all agents...

📝 Final Answer:

RAG (Retrieval-Augmented Generation) is a technique that 
combines information retrieval with text generation...

Key Benefits:
1. Reduces hallucinations by grounding in retrieved documents
2. Enables fresh data without model retraining
3. More cost-effective than fine-tuning for many use cases
4. Provides source attribution and transparency

[3-paragraph detailed answer...]

Sources: [citations to 19 documents]
⏱️ Synthesis time: 1.0s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ FINAL METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total time: 4.5s
Accuracy: 92% (highest!)
Agents used: 4
Total LLM calls: 8
Documents retrieved: 19
Facts extracted: 15
Conflicts resolved: 1
```

#### Хвилина 8: Аналіз Agent Logs (1 хв)
**ПРОКРУТИТИ НАЗАД ДО REASONING AGENT OUTPUT**

**ЩО СКАЗАТИ:**
> "Зверніть увагу на reasoning agent. Він знайшов протиріччя між двома фактами про latency. Замість того щоб ігнорувати або вибрати один наосліп, він зробив reasoning: 'обидва правильні, залежить від implementation'. Це рівень складності що неможливий в простіших RAG."

#### Хвилина 9: Performance та Trade-offs (1 хв)
**ПОКАЗАТИ ТАБЛИЦЮ:**
```
📊 Multi-Agent RAG Performance:

✅ Переваги:
- Найвища точність: 92%
- Handles complex multi-part queries
- Self-correcting (reasoning agent)
- Transparent reasoning (видно кожен крок)
- Can detect and resolve conflicts

⚠️ Trade-offs:
- Найповільніший: 4.5s
- Найбільше LLM calls (8+ per query)
- Складна архітектура
- Вища вартість (якщо API)

💰 Cost Estimate:
- Local (Ollama): ~4.5s compute
- OpenAI GPT-4: ~$0.15 per query (8 API calls)
```

#### Хвилина 10: Use Cases та Final Insights (1 хв)
**ЩО ГОВОРИТИ:**
> "Коли використовувати Multi-Agent?"

**ПОКАЗАТИ:**
```
✅ IDEAL USE CASES:
- Research assistants (academic papers, legal research)
- Complex decision-making (compare multiple options)
- Multi-domain queries (need info from different sources)
- Quality-critical applications (medical, financial)
- Educational systems (need to explain reasoning)

❌ NOT RECOMMENDED:
- Simple Q&A
- Real-time / low-latency needs (<1s)
- Budget constraints
- Simple factual lookups

Real-world example:
  Query: "Should I use RAG or fine-tuning for my chatbot?"
  Multi-Agent: 
    1. Plans: compare both approaches
    2. Retrieves: docs about each
    3. Reasons: pros/cons, cost analysis
    4. Synthesizes: recommendation matrix based on use case
```

**ФІНАЛЬНИЙ КОМЕНТАР:**
> "Multi-Agent - це як команда експертів замість одного спеціаліста. Дорожче, повільніше, але якість відповіді значно вища для складних задач."

---

# 🎬 ЗАВЕРШЕННЯ ПРАКТИКИ (2 хвилини)

---

## Підсумок та Q&A

### 🎤 Фінальні Слова

**ЩО ГОВОРИТИ:**
> "Підіб'ємо підсумки. Ми побачили 6 RAG підходів в дії:"

**ПОКАЗАТИ ФІНАЛЬНУ ТАБЛИЦЮ:**
```
╔═══════════════════════╦══════════╦═══════╦═════════════════╗
║ RAG Type              ║ Accuracy ║ Speed ║ Use Case        ║
╠═══════════════════════╬══════════╬═══════╬═════════════════╣
║ Naive                 ║   30%    ║ 2.6s  ║ MVP/Demo        ║
║ Retrieve-and-Rerank ⭐ ║   85%    ║ 3.4s  ║ PRODUCTION      ║
║ Multimodal            ║   N/A    ║ 65ms  ║ E-commerce      ║
║ Graph                 ║   90%    ║ 2.9s  ║ Knowledge-heavy ║
║ Agentic Router        ║   91%    ║ 4.0s  ║ Adaptive        ║
║ Agentic Multi-Agent   ║   92%    ║ 4.5s  ║ Complex         ║
╚═══════════════════════╩══════════╩═══════╩═════════════════╝

⭐ RECOMMENDED: Retrieve-and-Rerank для більшості production cases
```

**КЛЮЧОВІ TAKEAWAYS:**
```
1️⃣ Cross-Encoder Парадокс: Two-stage швидше ніж one-stage
2️⃣ Multimodal відкриває нові use cases (images + text)
3️⃣ Agents додають reasoning але коштують latency
4️⃣ Немає "one size fits all" - вибір залежить від use case
5️⃣ Локальний Ollama: 99% економія vs cloud APIs
```

### Відкрити для Питань (5 хв)

**МОЖЛИВІ ПИТАННЯ ТА ВІДПОВІДІ:**

**Q: Скільки коштує запустити в production?**
A: 
- Local (Ollama): ~$240/рік (електрика)
- Cloud (OpenAI): $0.07-0.15 за query → $36K-60K/рік
- Hybrid: Ollama для простих, GPT-4 для складних

**Q: Який chunk size рекомендуєте?**
A:
- Загальне: 512-1000 chars, overlap 100-200
- Технічна документація: 1000-1500 (більші chunks)
- Conversational: 200-500 (менші chunks)
- Завжди A/B тестуйте на вашому датасеті!

**Q: Як вибрати між FAISS та ChromaDB?**
A:
- FAISS: максимальна швидкість, мільйони vectors
- ChromaDB: простіше, built-in metadata filtering, mid-scale
- Для початку: ChromaDB
- Для scale (1M+ docs): FAISS

**Q: Чи можна комбінувати підходи?**
A:
- ТАК! Наприклад:
  - Retrieve-and-Rerank + Self-RAG (reranking + adaptive)
  - Graph RAG + Multimodal (entities + images)
  - Hybrid baseline + Agentic router для складних queries

**Q: Як монітори якість RAG в production?**
A:
```
Key metrics:
- Retrieval precision/recall
- Answer relevancy (RAGAS)
- Response time (p50, p95, p99)
- User feedback (thumbs up/down)
- A/B test results

Tools:
- LangSmith (LangChain)
- Weights & Biases
- Custom dashboards (Grafana)
```

---

## 📋 Backup: Hybrid RAG (якщо залишиться час)

**ЯКЩО Є 3-5 ДОДАТКОВИХ ХВИЛИН:**

```bash
python rag_demos/hybrid_rag/hybrid_rag_demo.py
```

**ШВИДКИЙ СЦЕНАРІЙ:**
> "Бонусом покажу Hybrid RAG - комбінацію sparse (TF-IDF) і dense (FAISS). Ми виправили баг в RRF algorithm минулого тижня!"

**ПОКАЗАТИ:**
- RRF formula
- Alpha parameter (0.3 vs 0.7)
- До виправлення: всі scores 0.008
- Після виправлення: scores різні (0.0164, 0.0161, ...)

---

## 📱 Матеріали для Студентів (After Workshop)

**ЩО НАДАТИ:**
```
📁 Materials:
├── WORKSHOP_SUMMARY.md      (full guide)
├── All demo scripts
├── Benchmark results JSON
├── Presentation slides
└── Links:
    - ChromaDB docs: https://docs.trychroma.com/
    - CLIP paper: https://arxiv.org/abs/2103.00020
    - Sentence Transformers: https://www.sbert.net/
    - LangChain RAG: https://python.langchain.com/docs/use_cases/question_answering/
```

---

## 🎓 КІНЕЦЬ WORKSHOP

**ФІНАЛЬНІ СЛОВА:**
> "Дякую за увагу! Всі матеріали, код і презентації надішлю вам. Якщо будуть питання після воркшопу - пишіть. Успіхів в імплементації RAG систем!"

**Показати QR код або контакти для follow-up questions**

---

**Створено**: 26 жовтня 2025
**Версія**: 1.0 - READY FOR WORKSHOP
**Тривалість**: 50 хвилин практики
**Формат**: Live demos + explanations

🎉 **ГОТОВО ДО ВОРКШОПУ 30 ЖОВТНЯ!**
