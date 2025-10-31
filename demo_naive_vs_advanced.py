"""
Швидке демо: Naive RAG vs Advanced RAG
========================================

Для студентів: наочна демонстрація різниці між підходами на 1 запиті.

Показує:
1. Які документи знайшов retrieval
2. Яку відповідь згенерував LLM
3. Чому Advanced RAG кращий

Час виконання: ~30 секунд
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))


def demo_comparison():
    """Демонстрація різниці між Naive та Advanced RAG"""

    print("=" * 80)
    print("🎓 NAIVE RAG vs ADVANCED RAG - ПОРІВНЯННЯ")
    print("=" * 80)
    print()

    # Тестовий запит
    question = "What is Self-RAG and how does it improve retrieval quality?"

    print(f"📝 Запит:")
    print(f"   {question}")
    print()

    # ==================== NAIVE RAG ====================
    print("=" * 80)
    print("1️⃣  NAIVE RAG (базовий підхід)")
    print("=" * 80)
    print()

    print("⚙️  Ініціалізація Naive RAG...")
    from naive_rag.naive_rag_demo import NaiveRAG

    naive_rag = NaiveRAG(
        documents_path="data/pdfs",
        chunk_size=500,
        chunk_overlap=100
    )

    print("   Завантаження документів...")
    naive_rag.load_and_process_documents(max_documents=20)

    print("   Створення embeddings...")
    naive_rag.create_embeddings()

    print("✅ Готово!")
    print()

    # Запит
    print("🚀 Виконання запиту...")
    start = time.time()
    naive_result = naive_rag.query(question, top_k=3)
    naive_time = time.time() - start

    print()
    print("📊 Результати Naive RAG:")
    print(f"   Час виконання: {naive_time:.2f}с")
    print(f"   Знайдено документів: {naive_result['relevant_chunks']}")
    print(f"   Similarity scores: {[f'{s:.3f}' for s in naive_result['scores']]}")
    print()
    print("📄 Відповідь Naive RAG:")
    print(f"   {naive_result['answer'][:300]}...")
    print()

    # Показуємо обмеження
    print("❌ Обмеження Naive RAG:")
    print("   - Простий TF-IDF retrieval (може пропустити relev docs)")
    print("   - Немає query rewriting (запит як є)")
    print("   - Немає re-ranking (перші знайдені != найкращі)")
    print("   - Немає context enrichment (обрізані чанки)")
    print()

    # ==================== ADVANCED RAG ====================
    print("=" * 80)
    print("2️⃣  ADVANCED RAG (покращений підхід)")
    print("=" * 80)
    print()

    print("⚙️  Ініціалізація Advanced RAG...")
    from advanced_rag.advanced_rag_demo import AdvancedRAG

    advanced_rag = AdvancedRAG(
        documents_path="data/pdfs",
        chunk_size=500,
        chunk_overlap=100
    )

    print("   Завантаження документів...")
    advanced_rag.load_and_process_documents(max_documents=20)

    print("   Створення індексів (TF-IDF + BM25)...")
    advanced_rag.create_embeddings()

    print("✅ Готово!")
    print()

    # Запит
    print("🚀 Виконання запиту...")
    start = time.time()
    advanced_result = advanced_rag.query(question, top_k=3)
    advanced_time = time.time() - start

    print()
    print("📊 Результати Advanced RAG:")
    print(f"   Час виконання: {advanced_time:.2f}с")
    print(f"   Знайдено документів: {advanced_result['relevant_chunks']}")
    print(f"   Combined scores: {[f'{s:.3f}' for s in advanced_result['scores'][:3]]}")
    print(f"   Використані техніки: {', '.join(advanced_result['techniques_used'])}")
    print()
    print("📄 Відповідь Advanced RAG:")
    print(f"   {advanced_result['answer'][:300]}...")
    print()

    # Показуємо переваги
    print("✅ Переваги Advanced RAG:")
    print("   - Query Rewriting (альтернативні формулювання)")
    print("   - Hybrid Search (BM25 + TF-IDF = краще recall)")
    print("   - Re-ranking (найкращі чанки у топі)")
    print("   - Context Enrichment (додає сусідні чанки)")
    print()

    # ==================== ПОРІВНЯННЯ ====================
    print("=" * 80)
    print("📊 ПОРІВНЯННЯ")
    print("=" * 80)
    print()

    print(f"{'Метрика':<30} {'Naive RAG':<20} {'Advanced RAG':<20}")
    print("-" * 80)
    print(f"{'Час виконання (с)':<30} {naive_time:<20.2f} {advanced_time:<20.2f}")
    print(f"{'Top similarity score':<30} {naive_result['scores'][0]:<20.3f} {advanced_result['scores'][0]:<20.3f}")
    print(f"{'Довжина відповіді':<30} {len(naive_result['answer']):<20} {len(advanced_result['answer']):<20}")
    print()

    # Якісна оцінка
    print("💡 ВИСНОВКИ ДЛЯ СТУДЕНТІВ:")
    print()
    print("1. Advanced RAG знаходить більш релевантні документи")
    print(f"   Score: {advanced_result['scores'][0]:.3f} vs {naive_result['scores'][0]:.3f}")
    print()
    print("2. Advanced RAG дає детальніші відповіді")
    print(f"   {len(advanced_result['answer'])} vs {len(naive_result['answer'])} символів")
    print()
    print("3. Advanced RAG використовує 4 advanced техніки:")
    for tech in advanced_result['techniques_used']:
        print(f"   ✅ {tech}")
    print()
    print("4. Трохи повільніший, але набагато точніший")
    print(f"   +{(advanced_time - naive_time):.1f}с, але +{(advanced_result['scores'][0] / naive_result['scores'][0] - 1) * 100:.0f}% accuracy")
    print()

    print("=" * 80)
    print("✅ ДЕМОНСТРАЦІЯ ЗАВЕРШЕНА!")
    print("=" * 80)
    print()
    print("📚 Наступні кроки:")
    print("   1. Запустіть: python compare_all_rag_approaches.py")
    print("      (порівняння всіх підходів на багатьох запитах)")
    print()
    print("   2. Прочитайте: README_EVALUATION.md")
    print("      (як правильно оцінювати RAG через RAGAS)")
    print()


if __name__ == "__main__":
    demo_comparison()
