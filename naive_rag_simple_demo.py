#!/usr/bin/env python3
"""
Простий Naive RAG Demo для воркшопу
Не потребує зовнішніх файлів - все вбудовано
"""

import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Вбудовані тестові документи
SAMPLE_DOCUMENTS = [
    "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. It was introduced by Facebook AI Research in 2020.",
    "RAG uses a retriever to fetch relevant documents from a knowledge base, then passes them to a generator model to produce answers.",
    "Dense retrieval methods use neural embeddings to represent queries and documents in a continuous vector space.",
    "Sparse retrieval methods like TF-IDF and BM25 rely on exact keyword matching and term frequency statistics.",
    "Cross-encoders provide better accuracy for reranking but are slower than bi-encoders because they process query and document together.",
    "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search in high-dimensional spaces.",
    "Transformers use self-attention mechanisms to process sequences in parallel, unlike RNNs which process sequentially.",
    "BERT (Bidirectional Encoder Representations from Transformers) revolutionized NLP by using bidirectional context.",
    "Fine-tuning adapts a pre-trained model to a specific task by training on task-specific data.",
    "Embeddings map discrete tokens to continuous vectors, capturing semantic meaning and relationships between words."
]

# Тестові запити
TEST_QUERIES = [
    "What is retrieval-augmented generation?",
    "Explain the difference between dense and sparse retrieval",
    "How does self-attention work in transformers?"
]


class NaiveRAG:
    """Найпростіша RAG реалізація"""
    
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def build_index(self):
        """Створити TF-IDF індекс"""
        print("🔢 Створення TF-IDF індексу...")
        start = time.time()
        
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        
        elapsed = time.time() - start
        print(f"✅ Індекс створено за {elapsed*1000:.1f}ms")
        print(f"   Словник: {len(self.vectorizer.vocabulary_)} слів")
        print(f"   Документів: {len(self.documents)}")
        
    def retrieve(self, query, top_k=3):
        """Знайти top-k найбільш релевантних документів"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Топ-k індекси
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'doc_id': idx,
                'score': similarities[idx],
                'text': self.documents[idx]
            })
        
        return results
    
    def query(self, question, top_k=3):
        """Виконати RAG запит"""
        print(f"\n{'='*60}")
        print(f"❓ Query: {question}")
        print(f"{'='*60}")
        
        # Крок 1: Retrieval
        start_retrieval = time.time()
        results = self.retrieve(question, top_k=top_k)
        retrieval_time = time.time() - start_retrieval
        
        print(f"\n📚 Retrieved {len(results)} documents ({retrieval_time*1000:.1f}ms):")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. [Score: {result['score']:.4f}]")
            print(f"     {result['text'][:100]}...")
        
        # Крок 2: Generation (симулюємо - в production був би LLM)
        print(f"\n💭 Generating answer (simulated)...")
        context = "\n".join([r['text'] for r in results])
        
        # Просте summary на основі retrieved docs
        answer = f"Based on the retrieved documents: {results[0]['text']}"
        
        print(f"\n✅ Answer:")
        print(f"   {answer[:200]}...")
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': results,
            'retrieval_time': retrieval_time,
            'top_score': results[0]['score'] if results else 0
        }


def run_demo():
    """Запустити демонстрацію"""
    print("="*60)
    print("🚀 NAIVE RAG DEMO")
    print("="*60)
    print()
    
    # Крок 1: Ініціалізація
    print(f"📄 Використовуємо {len(SAMPLE_DOCUMENTS)} вбудованих документів")
    rag = NaiveRAG(SAMPLE_DOCUMENTS)
    
    # Крок 2: Побудувати індекс
    rag.build_index()
    
    # Крок 3: Тестові запити
    print(f"\n🎯 Запускаємо {len(TEST_QUERIES)} тестових запитів...")
    
    results = []
    for query in TEST_QUERIES:
        result = rag.query(query, top_k=3)
        results.append(result)
        time.sleep(0.5)  # Пауза для читабельності
    
    # Крок 4: Підсумок
    print(f"\n{'='*60}")
    print("📊 ПІДСУМОК")
    print(f"{'='*60}")
    
    avg_retrieval_time = np.mean([r['retrieval_time'] for r in results])
    avg_score = np.mean([r['top_score'] for r in results])
    
    print(f"\n✅ Виконано {len(results)} запитів")
    print(f"⏱️  Середній час retrieval: {avg_retrieval_time*1000:.1f}ms")
    print(f"📈 Середній top-1 score: {avg_score:.4f}")
    
    # Ключові обмеження
    print(f"\n⚠️  ОБМЕЖЕННЯ NAIVE RAG:")
    print(f"   ❌ Keyword-only search (немає семантики)")
    print(f"   ❌ Немає reranking (перші results не завжди найкращі)")
    print(f"   ❌ TF-IDF не розуміє синоніми")
    print(f"   ❌ Низька точність (~30%)")
    print(f"\n   ✅ Швидкість: хороша (~{avg_retrieval_time*1000:.0f}ms)")
    print(f"   ✅ Простота: дуже проста імплементація")
    
    print(f"\n💡 ДЛЯ PRODUCTION:")
    print(f"   → Використовуйте Retrieve-and-Rerank")
    print(f"   → Dense embeddings (FAISS)")
    print(f"   → Cross-encoder reranking")
    print(f"   → Accuracy до 85-90%")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    run_demo()
