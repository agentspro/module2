"""
ChromaDB RAG з Advanced техніками
====================================

Advanced RAG система з використанням ChromaDB для векторного зберігання.

Особливості:
- Persistent storage (ChromaDB на диску)
- Sentence-Transformers embeddings (all-MiniLM-L6-v2)
- Hybrid search (ChromaDB vector + BM25 keyword)
- Query rewriting, re-ranking, context enrichment
- Сумісність з тестовою інфраструктурою

Час виконання: ~3-4 секунди на запит
Точність: ~65-70% (очікується)
"""

import sys
from pathlib import Path
import time
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import fitz  # PyMuPDF
import requests
import math

# ChromaDB imports
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    print("ПОМИЛКА: ChromaDB не встановлено!")
    print("Встановіть: pip install chromadb")
    sys.exit(1)

# Додаємо шлях до загальних утиліт
sys.path.append(str(Path(__file__).parent.parent))
from utils import DocumentLoader, save_results


# ==================== LLM Integration ====================

def detect_llm_provider() -> str:
    """Визначає який LLM доступний"""
    # Перевірка OpenAI API
    import os
    if os.getenv("OPENAI_API_KEY"):
        return "OpenAI (gpt-4o-mini)"

    # Перевірка Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return "Ollama (llama3.2)"
    except:
        pass

    return "Ollama (необхідно запустити)"


def generate_answer_with_llm(question: str, contexts: List[str], max_tokens: int = 256) -> str:
    """Генерує відповідь через доступний LLM"""
    import os

    # Спробуємо OpenAI спочатку
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()

            context_text = "\n\n".join(contexts[:3])  # Топ-3 контексти

            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Answer concisely and factually."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer based on the context above:"}
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI помилка: {e}, fallback на Ollama")

    # Fallback на Ollama
    try:
        context_text = "\n\n".join(contexts[:3])

        prompt = f"""Context:
{context_text}

Question: {question}

Answer based on the context above:"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.7}
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"Помилка Ollama: {response.status_code}"
    except Exception as e:
        return f"LLM недоступний: {str(e)}"


# ==================== BM25 для Hybrid Search ====================

class BM25Retriever:
    """BM25 алгоритм для keyword-based пошуку"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0

    def fit(self, corpus: List[str]):
        """Індексує корпус"""
        self.corpus = corpus
        self.doc_len = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

        # Обчислюємо IDF
        df = {}
        for doc in corpus:
            words = set(doc.lower().split())
            for word in words:
                df[word] = df.get(word, 0) + 1

        num_docs = len(corpus)
        self.idf = {word: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
                    for word, freq in df.items()}

    def get_scores(self, query: str) -> np.ndarray:
        """Обчислює BM25 scores для всіх документів"""
        query_words = query.lower().split()
        scores = np.zeros(len(self.corpus))

        for word in query_words:
            if word not in self.idf:
                continue

            idf_score = self.idf[word]

            for idx, doc in enumerate(self.corpus):
                doc_words = doc.lower().split()
                word_freq = doc_words.count(word)

                if word_freq == 0:
                    continue

                # BM25 formula
                numerator = word_freq * (self.k1 + 1)
                denominator = word_freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / self.avgdl)
                scores[idx] += idf_score * (numerator / denominator)

        return scores


# ==================== ChromaDB RAG System ====================

class ChromaDBRAG:
    """
    Advanced RAG система з ChromaDB векторним сховищем

    Використовує:
    - ChromaDB для persistent векторного зберігання
    - Sentence-Transformers (all-MiniLM-L6-v2) для embeddings
    - BM25 для keyword search
    - Hybrid search (vector + BM25)
    - Query rewriting, re-ranking, context enrichment
    """

    def __init__(
        self,
        documents_path: str = "data/pdfs",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        persist_directory: str = "chromadb_storage"
    ):
        self.documents_path = documents_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory

        # Дані
        self.chunks = []
        self.documents = []

        # BM25 для hybrid search
        self.bm25 = BM25Retriever()

        # ChromaDB клієнт
        self.chroma_client = None
        self.collection = None
        self._init_chromadb()

    def _init_chromadb(self):
        """Ініціалізує ChromaDB клієнт та collection"""
        if not HAS_CHROMADB:
            raise ImportError("ChromaDB не встановлено!")

        # Створюємо persistent клієнт
        self.chroma_client = chromadb.PersistentClient(
            path=str(Path(__file__).parent / self.persist_directory)
        )

        # Sentence-Transformers embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Створюємо або завантажуємо collection
        try:
            self.collection = self.chroma_client.get_collection(
                name="rag_documents",
                embedding_function=self.embedding_function
            )
            print(f"Завантажено існуючу ChromaDB collection: {self.collection.count()} документів")
        except:
            self.collection = self.chroma_client.create_collection(
                name="rag_documents",
                embedding_function=self.embedding_function,
                metadata={"description": "RAG document chunks"}
            )
            print("Створено нову ChromaDB collection")

    def load_and_process_documents(self, max_documents: Optional[int] = None) -> List[Dict]:
        """Завантажує та чанкує PDF документи"""
        loader = DocumentLoader()
        self.documents = loader.load_pdfs(
            directory=self.documents_path,
            max_documents=max_documents
        )

        # Створюємо чанки
        self.chunks = []
        for doc in self.documents:
            doc_chunks = self._chunk_text(
                doc["content"],
                self.chunk_size,
                self.chunk_overlap
            )

            for idx, chunk_text in enumerate(doc_chunks):
                self.chunks.append({
                    "id": len(self.chunks),
                    "content": chunk_text,
                    "source": doc["filename"],
                    "chunk_index": idx
                })

        return self.documents

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Розбиває текст на чанки з перекриттям"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]

            if chunk.strip():
                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    def create_embeddings(self):
        """Створює embeddings та зберігає у ChromaDB"""
        if not self.chunks:
            raise ValueError("Спочатку завантажте документи!")

        # Перевіряємо чи вже є дані
        if self.collection.count() > 0:
            print(f"ChromaDB вже містить {self.collection.count()} документів. Очищаємо...")
            # Видаляємо стару collection та створюємо нову
            self.chroma_client.delete_collection("rag_documents")
            self.collection = self.chroma_client.create_collection(
                name="rag_documents",
                embedding_function=self.embedding_function
            )

        # Додаємо чанки до ChromaDB
        ids = [f"chunk_{chunk['id']}" for chunk in self.chunks]
        documents = [chunk["content"] for chunk in self.chunks]
        metadatas = [{
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "chunk_id": chunk["id"]
        } for chunk in self.chunks]

        # Додаємо батчами (ChromaDB має ліміт)
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_meta = metadatas[i:i+batch_size]

            self.collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta
            )

        # Створюємо BM25 індекс для hybrid search
        self.bm25.fit([chunk["content"] for chunk in self.chunks])

        print(f"ChromaDB: додано {len(self.chunks)} чанків")

    def query_rewriting(self, query: str, num_variants: int = 3) -> List[str]:
        """Генерує альтернативні формулювання запиту"""
        # Базова реалізація без LLM
        variants = [query]

        # Додаємо питальні форми
        if not query.endswith("?"):
            variants.append(query + "?")

        # Додаємо версію без питального слова
        question_words = ["what", "how", "why", "when", "where", "who", "which"]
        for word in question_words:
            if query.lower().startswith(word):
                variants.append(query.split(None, 1)[1] if " " in query else query)
                break

        return variants[:num_variants]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Гібридний пошук: ChromaDB vector + BM25 keyword

        Args:
            query: Пошуковий запит
            top_k: Кількість результатів
            alpha: Баланс (0=тільки BM25, 1=тільки vector)

        Returns:
            List of (chunk_id, combined_score)
        """
        # 1. Vector search через ChromaDB
        chroma_results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k * 2, len(self.chunks))
        )

        # Витягуємо chunk_ids та distances
        vector_scores = {}
        if chroma_results['ids'] and chroma_results['ids'][0]:
            for idx, chunk_str_id in enumerate(chroma_results['ids'][0]):
                chunk_id = int(chunk_str_id.split('_')[1])
                # ChromaDB повертає distances, конвертуємо в similarity
                distance = chroma_results['distances'][0][idx]
                similarity = 1.0 / (1.0 + distance)  # Inverse distance
                vector_scores[chunk_id] = similarity

        # 2. BM25 keyword search
        bm25_scores = self.bm25.get_scores(query)

        # Нормалізація BM25 scores
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # 3. Комбінуємо scores
        combined_scores = {}
        for idx in range(len(self.chunks)):
            vec_score = vector_scores.get(idx, 0.0)
            bm25_score = bm25_scores[idx]

            combined_scores[idx] = alpha * vec_score + (1 - alpha) * bm25_score

        # Сортуємо та повертаємо топ-k
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return sorted_results

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """Re-ranking кандидатів з бонусами за релевантність"""
        reranked = []
        query_words = set(query.lower().split())

        for chunk_id, score in candidates:
            chunk = self.chunks[chunk_id]
            chunk_words = set(chunk["content"].lower().split())

            # Бонус за exact match слів з запиту
            overlap = len(query_words & chunk_words)
            bonus = overlap * 0.1

            # Бонус за довжину (довші чанки краще)
            length_bonus = min(len(chunk["content"]) / 1000, 0.2)

            final_score = score + bonus + length_bonus
            reranked.append((chunk_id, final_score))

        return sorted(reranked, key=lambda x: x[1], reverse=True)

    def context_enrichment(self, chunk_ids: List[int]) -> List[Dict]:
        """Збагачує контекст додаючи сусідні чанки"""
        enriched = []

        for chunk_id in chunk_ids:
            chunk = self.chunks[chunk_id].copy()

            # Додаємо попередній чанк
            if chunk_id > 0:
                prev_chunk = self.chunks[chunk_id - 1]
                if prev_chunk["source"] == chunk["source"]:
                    chunk["prev_context"] = prev_chunk["content"][:200]

            # Додаємо наступний чанк
            if chunk_id < len(self.chunks) - 1:
                next_chunk = self.chunks[chunk_id + 1]
                if next_chunk["source"] == chunk["source"]:
                    chunk["next_context"] = next_chunk["content"][:200]

            enriched.append(chunk)

        return enriched

    def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Виконує повний RAG pipeline з Advanced техніками

        Pipeline:
        1. Query Rewriting - альтернативні формулювання
        2. Hybrid Search - vector + BM25 для всіх варіантів
        3. Re-ranking - бонуси за релевантність
        4. Context Enrichment - додає сусідні чанки
        5. LLM Generation - генерує відповідь
        """
        start_time = time.time()

        # 1. Query Rewriting
        query_variants = self.query_rewriting(question, num_variants=2)

        # 2. Hybrid Search для всіх варіантів
        all_results = {}
        for variant in query_variants:
            results = self.hybrid_search(variant, top_k=10, alpha=0.5)
            for chunk_id, score in results:
                if chunk_id in all_results:
                    all_results[chunk_id] = max(all_results[chunk_id], score)
                else:
                    all_results[chunk_id] = score

        # Топ-кандидати
        candidates = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:10]

        # 3. Re-ranking
        reranked = self.rerank(question, candidates)
        top_chunks = reranked[:top_k]

        # 4. Context Enrichment
        chunk_ids = [chunk_id for chunk_id, _ in top_chunks]
        enriched_chunks = self.context_enrichment(chunk_ids)

        # 5. Генерація відповіді
        answer = self.generate_answer(question, enriched_chunks)

        execution_time = time.time() - start_time

        result = {
            "question": question,
            "answer": answer,
            "techniques_used": ["Query Rewriting", "Hybrid Search (ChromaDB+BM25)", "Re-ranking", "Context Enrichment"],
            "relevant_chunks": len(enriched_chunks),
            "sources": list(set([c["source"] for c in enriched_chunks])),
            "scores": [score for _, score in top_chunks],
            "execution_time": execution_time,
            "storage": "ChromaDB (persistent)"
        }

        return result

    def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Генерує відповідь через LLM зі збагаченим контекстом"""
        if not chunks:
            return "Не знайдено релевантної інформації."

        # Витягуємо контексти включаючи збагачений контекст
        contexts = []
        for chunk in chunks:
            context_parts = [chunk["content"]]

            if "prev_context" in chunk:
                context_parts.insert(0, f"[Попередній контекст] {chunk['prev_context']}")

            if "next_context" in chunk:
                context_parts.append(f"[Наступний контекст] {chunk['next_context']}")

            contexts.append(" ".join(context_parts))

        answer = generate_answer_with_llm(
            question=query,
            contexts=contexts,
            max_tokens=256
        )

        return answer


# ==================== Demo Runner ====================

def run_chromadb_rag_demo():
    """Запускає демонстрацію ChromaDB RAG"""
    print("="*70)
    print("CHROMADB RAG ДЕМОНСТРАЦІЯ")
    print("="*70)

    # Ініціалізація
    chunk_size = 500
    chunk_overlap = 100
    rag = ChromaDBRAG(
        documents_path="data/pdfs",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        persist_directory="chromadb_storage"
    )

    # Виводимо конфігурацію
    print(f"\nКонфігурація:")
    llm_model = detect_llm_provider()
    print(f"  Модель LLM: {llm_model}")
    print(f"  Розмір чанку: {chunk_size} символів")
    print(f"  Перекриття чанків: {chunk_overlap} символів")
    print(f"  Vector DB: ChromaDB (persistent)")
    print(f"  Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)")
    print(f"  Техніки: Query Rewriting, Hybrid Search, Re-ranking, Context Enrichment")

    # Завантаження
    print(f"\nЗавантаження документів...")
    documents = rag.load_and_process_documents(max_documents=50)
    print(f"Завантажено: {len(documents)} документів, {len(rag.chunks)} чанків")

    # Створення індексів
    print(f"Створення індексів (ChromaDB + BM25)...")
    rag.create_embeddings()
    print(f"Створено: ChromaDB collection + BM25 індекс")

    # Завантажуємо УНІФІКОВАНИЙ тестовий датасет
    loader = DocumentLoader()
    unified_queries = loader.load_unified_queries(max_queries=50)
    print(f"Тестових запитів: {len(unified_queries)}")

    print("\n" + "="*70)
    print("ВИКОНАННЯ ТЕСТІВ")
    print("="*70)

    all_results = {
        "system_name": "ChromaDB RAG",
        "total_documents": len(documents),
        "total_chunks": len(rag.chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "llm_model": detect_llm_provider(),
        "vector_db": "ChromaDB (persistent)",
        "embeddings": "all-MiniLM-L6-v2",
        "queries": []
    }

    # Групуємо по категоріях
    queries_by_category = defaultdict(list)
    for query in unified_queries:
        queries_by_category[query.get("category", "general")].append(query)

    # Тестуємо запити по категоріях
    for category, queries in queries_by_category.items():
        print(f"\nКатегорія: {category}")

        for query_data in queries:
            question = query_data.get("question", "")

            # Виконуємо запит
            result = rag.query(question, top_k=5)
            result["category"] = category
            result["query_id"] = query_data.get("id")
            result["difficulty"] = query_data.get("difficulty")
            all_results["queries"].append(result)

            # Виводимо короткий результат
            print(f"  ID {query_data.get('id')}: {question[:70]}...")
            print(f"  Час: {result['execution_time']:.2f}с | Оцінка: {result['scores'][0]:.3f}")

    # Статистика
    avg_time = np.mean([q["execution_time"] for q in all_results["queries"]])
    avg_score = np.mean([q["scores"][0] for q in all_results["queries"]])

    all_results["metrics"] = {
        "average_execution_time": avg_time,
        "average_top_score": avg_score,
        "total_queries": len(all_results["queries"]),
        "storage_type": "persistent (disk-based)"
    }

    save_results(all_results, "results/chromadb_rag_results.json")

    print("\n" + "="*70)
    print("ПІДСУМОК")
    print("="*70)
    print(f"Всього запитів: {len(all_results['queries'])}")
    print(f"Середній час: {avg_time:.2f}с")
    print(f"Середня оцінка: {avg_score:.3f}")
    print(f"Vector DB: ChromaDB (persistent storage)")
    print(f"\nРезультати збережено: results/chromadb_rag_results.json")
    print("="*70)


if __name__ == "__main__":
    run_chromadb_rag_demo()
