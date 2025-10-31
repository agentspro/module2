"""
Production RAG Benchmark - Реальні метрики продуктивності
Використовує sentence-transformers для embeddings та вимірює реальну продуктивність
"""
import sys
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

# Спроба імпорту production бібліотек
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
    print("✅ sentence-transformers доступний")
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers недоступний, використовую TF-IDF")

from utils.data_loader import DocumentLoader, TextSplitter


class ProductionEmbeddings:
    """Production-ready embeddings з sentence-transformers або TF-IDF fallback"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

        if HAS_SENTENCE_TRANSFORMERS:
            print(f"🔧 Завантаження моделі {model_name}...")
            start = time.time()
            self.model = SentenceTransformer(model_name)
            load_time = time.time() - start
            print(f"✅ Модель завантажена за {load_time:.2f}с")
            self.mode = "transformer"
        else:
            print("📊 Використовую TF-IDF (fallback)...")
            self.vocabulary = {}
            self.idf = {}
            self.mode = "tfidf"

    def fit(self, texts: List[str]):
        """Навчання (тільки для TF-IDF)"""
        if self.mode == "tfidf":
            doc_word_sets = []
            for doc in texts:
                words = set(doc.lower().split())
                doc_word_sets.append(words)
                for word in words:
                    self.vocabulary[word] = self.vocabulary.get(word, 0) + 1

            num_docs = len(texts)
            for word in self.vocabulary:
                doc_count = sum(1 for word_set in doc_word_sets if word in word_set)
                self.idf[word] = np.log(num_docs / (doc_count + 1)) + 1

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Створення embeddings для batch текстів"""
        if self.mode == "transformer":
            return self.model.encode(texts, show_progress_bar=show_progress, convert_to_numpy=True)
        else:
            return np.array([self._embed_tfidf(text) for text in texts])

    def embed(self, text: str) -> np.ndarray:
        """Створення embedding для одного тексту"""
        if self.mode == "transformer":
            return self.model.encode([text], convert_to_numpy=True)[0]
        else:
            return self._embed_tfidf(text)

    def _embed_tfidf(self, text: str) -> np.ndarray:
        """TF-IDF embedding (fallback)"""
        words = text.lower().split()
        word_count = defaultdict(int)
        for word in words:
            word_count[word] += 1

        vector = np.zeros(len(self.vocabulary))
        for i, word in enumerate(sorted(self.vocabulary.keys())):
            if word in word_count:
                tf = word_count[word] / max(len(words), 1)
                idf = self.idf.get(word, 1)
                vector[i] = tf * idf

        return vector

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Косинусна подібність"""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot / (norm1 * norm2 + 1e-10)


class ProductionRAGBenchmark:
    """Benchmark різних RAG підходів з реальними метриками"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = ProductionEmbeddings(embedding_model)
        self.chunks = []
        self.chunk_embeddings = None
        self.documents = []

    def load_data(self, docs_path: str = "data/corporate_docs"):
        """Завантаження даних"""
        print(f"\n📚 Завантаження документів з {docs_path}...")

        loader = DocumentLoader(docs_path)
        self.documents = loader.load_documents()

        splitter = TextSplitter(chunk_size=500, chunk_overlap=100)
        self.chunks = splitter.split_documents(self.documents)

        print(f"✅ Завантажено {len(self.documents)} документів -> {len(self.chunks)} чанків")

    def create_embeddings(self):
        """Створення embeddings з вимірюванням часу"""
        print(f"\n🔢 Створення embeddings для {len(self.chunks)} чанків...")

        texts = [chunk["content"] for chunk in self.chunks]

        # Навчання (для TF-IDF)
        if self.embedding_model.mode == "tfidf":
            self.embedding_model.fit(texts)

        # Створення embeddings з вимірюванням часу
        start = time.time()
        self.chunk_embeddings = self.embedding_model.embed_batch(texts, show_progress=True)
        embedding_time = time.time() - start

        print(f"✅ Embeddings створено за {embedding_time:.2f}с")
        print(f"   Швидкість: {len(self.chunks) / embedding_time:.1f} docs/sec")
        print(f"   Розмір вектору: {self.chunk_embeddings.shape[1]}")

        return {
            "embedding_time": embedding_time,
            "docs_per_second": len(self.chunks) / embedding_time,
            "vector_dimension": self.chunk_embeddings.shape[1],
            "model": self.embedding_model.model_name,
            "mode": self.embedding_model.mode
        }

    def benchmark_retrieval(self, queries: List[str], top_k: int = 5) -> Dict:
        """Benchmark retrieval з детальними метриками"""
        print(f"\n🔍 Benchmark retrieval для {len(queries)} запитів...")

        results = []
        retrieval_times = []

        for query in queries:
            start = time.time()

            # Створюємо embedding для запиту
            query_embedding = self.embedding_model.embed(query)

            # Рахуємо подібність з усіма чанками
            similarities = []
            for i, chunk_embedding in enumerate(self.chunk_embeddings):
                similarity = self.embedding_model.cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((i, similarity))

            # Топ-k результати
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_results = similarities[:top_k]

            retrieval_time = time.time() - start
            retrieval_times.append(retrieval_time)

            results.append({
                "query": query,
                "top_k_scores": [score for _, score in top_results],
                "top_k_sources": [self.chunks[idx]["source"] for idx, _ in top_results],
                "retrieval_time": retrieval_time
            })

        # Статистика
        avg_time = np.mean(retrieval_times)
        p50 = np.percentile(retrieval_times, 50)
        p95 = np.percentile(retrieval_times, 95)
        p99 = np.percentile(retrieval_times, 99)

        metrics = {
            "total_queries": len(queries),
            "avg_retrieval_time": avg_time,
            "p50_latency": p50,
            "p95_latency": p95,
            "p99_latency": p99,
            "queries_per_second": 1 / avg_time if avg_time > 0 else 0,
            "avg_top1_score": np.mean([r["top_k_scores"][0] for r in results]),
            "avg_top5_score": np.mean([np.mean(r["top_k_scores"]) for r in results]),
            "results": results
        }

        print(f"\n📊 Метрики retrieval:")
        print(f"   Середній час: {avg_time*1000:.1f}ms")
        print(f"   P50 latency: {p50*1000:.1f}ms")
        print(f"   P95 latency: {p95*1000:.1f}ms")
        print(f"   P99 latency: {p99*1000:.1f}ms")
        print(f"   QPS: {metrics['queries_per_second']:.1f} queries/sec")
        print(f"   Середній top-1 score: {metrics['avg_top1_score']:.3f}")

        return metrics

    def benchmark_end_to_end(self, queries: List[str], expected_answers: List[str] = None) -> Dict:
        """End-to-end benchmark з accuracy метриками"""
        print(f"\n🎯 End-to-end benchmark...")

        e2e_times = []
        accuracy_scores = []

        for i, query in enumerate(queries):
            start = time.time()

            # Retrieval
            query_embedding = self.embedding_model.embed(query)
            similarities = [(j, self.embedding_model.cosine_similarity(query_embedding, emb))
                          for j, emb in enumerate(self.chunk_embeddings)]
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Генерація відповіді (симулюємо)
            best_chunk = self.chunks[similarities[0][0]]
            answer = best_chunk["content"][:200]

            e2e_time = time.time() - start
            e2e_times.append(e2e_time)

            # Accuracy (якщо є ground truth)
            if expected_answers and i < len(expected_answers):
                # Проста метрика: чи є ключові слова з відповіді в знайденому тексті
                expected_words = set(expected_answers[i].lower().split())
                found_words = set(answer.lower().split())
                overlap = len(expected_words & found_words) / len(expected_words) if expected_words else 0
                accuracy_scores.append(overlap)

        metrics = {
            "avg_e2e_time": np.mean(e2e_times),
            "total_time": sum(e2e_times),
            "avg_accuracy": np.mean(accuracy_scores) if accuracy_scores else None
        }

        print(f"\n📈 End-to-end метрики:")
        print(f"   Середній час: {metrics['avg_e2e_time']*1000:.1f}ms")
        print(f"   Загальний час: {metrics['total_time']:.2f}s")
        if metrics['avg_accuracy']:
            print(f"   Середня accuracy: {metrics['avg_accuracy']:.1%}")

        return metrics


def run_production_benchmark():
    """Запуск повного production benchmark"""
    print("="*70)
    print("🚀 PRODUCTION RAG BENCHMARK")
    print("="*70)

    # Ініціалізація
    benchmark = ProductionRAGBenchmark()

    # Завантаження даних
    benchmark.load_data()

    # Створення embeddings
    embedding_metrics = benchmark.create_embeddings()

    # Завантаження тестових запитів
    loader = DocumentLoader()
    test_queries = loader.load_test_queries()

    # Збираємо всі запити
    all_queries = []
    expected_answers = []
    for category in ["simple_queries", "medium_queries", "complex_queries"]:
        for q in test_queries[category]:
            all_queries.append(q["question"])
            expected_answers.append(q["expected_answer"])

    # Benchmark retrieval
    retrieval_metrics = benchmark.benchmark_retrieval(all_queries, top_k=5)

    # End-to-end benchmark
    e2e_metrics = benchmark.benchmark_end_to_end(all_queries, expected_answers)

    # Підсумковий звіт
    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "embedding_model": benchmark.embedding_model.model_name,
            "embedding_mode": benchmark.embedding_model.mode,
            "documents": len(benchmark.documents),
            "chunks": len(benchmark.chunks),
            "vector_dimension": embedding_metrics["vector_dimension"]
        },
        "embedding_performance": embedding_metrics,
        "retrieval_performance": {
            k: v for k, v in retrieval_metrics.items() if k != "results"
        },
        "end_to_end_performance": e2e_metrics,
        "sample_results": retrieval_metrics["results"][:3]  # Перші 3 для прикладу
    }

    # Збереження звіту
    output_file = "results/production_benchmark.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Звіт збережено в {output_file}")

    # Виведення підсумку
    print(f"\n{'='*70}")
    print("📊 ПІДСУМОК BENCHMARK")
    print(f"{'='*70}")
    print(f"\n🔧 Система:")
    print(f"   Модель: {benchmark.embedding_model.model_name} ({benchmark.embedding_model.mode})")
    print(f"   Документів: {len(benchmark.documents)}")
    print(f"   Чанків: {len(benchmark.chunks)}")
    print(f"   Розмір вектору: {embedding_metrics['vector_dimension']}")

    print(f"\n⚡ Продуктивність Embeddings:")
    print(f"   Час створення: {embedding_metrics['embedding_time']:.2f}s")
    print(f"   Швидкість: {embedding_metrics['docs_per_second']:.1f} docs/sec")

    print(f"\n🔍 Продуктивність Retrieval:")
    print(f"   Середня latency: {retrieval_metrics['avg_retrieval_time']*1000:.1f}ms")
    print(f"   P95 latency: {retrieval_metrics['p95_latency']*1000:.1f}ms")
    print(f"   QPS: {retrieval_metrics['queries_per_second']:.1f}")
    print(f"   Accuracy (top-1): {retrieval_metrics['avg_top1_score']:.1%}")

    print(f"\n🎯 End-to-End:")
    print(f"   Середній час: {e2e_metrics['avg_e2e_time']*1000:.1f}ms")
    if e2e_metrics['avg_accuracy']:
        print(f"   Середня accuracy: {e2e_metrics['avg_accuracy']:.1%}")

    print(f"\n✅ Production benchmark завершено!")

    return final_report


if __name__ == "__main__":
    report = run_production_benchmark()
