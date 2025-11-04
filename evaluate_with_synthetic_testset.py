"""
Професійний RAG Evaluation з синтетичним датасетом
===================================================

Оцінює RAG систему на 50-200+ реальних тестових кейсах з усіма RAGAS метриками:

RAGAS Metrics (4 основні):
1. Faithfulness (0-1)         - чи базується відповідь на контексті (галюцинації?)
2. Answer Relevancy (0-1)     - чи релевантна відповідь запиту
3. Context Precision (0-1)    - чи точний retrieved контекст
4. Context Recall (0-1)       - чи повний retrieved контекст

Production targets:
- Faithfulness: > 0.90 (критично! галюцинації = провал)
- Answer Relevancy: > 0.85
- Context Precision: > 0.80
- Context Recall: > 0.80
"""

import sys
import os
from pathlib import Path
import json
import time
import numpy as np
from typing import Dict, List
from dotenv import load_dotenv

# Завантаження змінних середовища з .env (шукаємо в поточній та батьківській директорії)
load_dotenv()  # Спочатку поточна директорія
if not os.getenv('OPENAI_API_KEY'):
    # Якщо не знайшли, шукаємо в батьківській директорії
    load_dotenv(Path(__file__).parent.parent / '.env')

# RAGAS imports
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    print("❌ RAGAS не встановлено")
    print("   Встановіть: pip install ragas datasets langchain-openai")
    sys.exit(1)


def load_synthetic_testset(file_path: str = "data/synthetic_testset.json") -> Dict:
    """
    Завантажити синтетичний тестовий датасет

    Args:
        file_path: Шлях до JSON файлу з датасетом

    Returns:
        Dict з тестовими кейсами
    """
    print(f"📂 Завантаження тестового датасету: {file_path}")

    path = Path(file_path)
    if not path.exists():
        print(f"❌ Файл не знайдено: {file_path}")
        print(f"   Спочатку згенеруйте датасет:")
        print(f"   python generate_synthetic_testset.py")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    num_tests = len(data.get("testcases", []))
    print(f"✅ Завантажено: {num_tests} тестових кейсів")

    return data


def run_rag_on_testset(rag_system, testset: Dict) -> List[Dict]:
    """
    Запустити RAG систему на всіх тестових кейсах

    Args:
        rag_system: Ініціалізована RAG система (з методом query())
        testset: Тестовий датасет

    Returns:
        Список результатів з відповідями
    """
    print(f"\n🚀 Запуск RAG системи на тестовому датасеті...")

    results = []
    testcases = testset.get("testcases", [])

    print(f"   Обробка {len(testcases)} запитів...")
    print(f"   Це займе ~{len(testcases) * 3 / 60:.1f} хвилин")
    print()

    start_time = time.time()

    for idx, testcase in enumerate(testcases, 1):
        question = testcase["question"]

        # Показуємо прогрес кожні 10 запитів
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / idx
            remaining = (len(testcases) - idx) * avg_time
            print(f"   Прогрес: {idx}/{len(testcases)} ({idx/len(testcases)*100:.1f}%) - "
                  f"Залишилось ~{remaining/60:.1f} хв")

        # Запускаємо RAG
        try:
            result = rag_system.query(question)

            results.append({
                "question": question,
                "answer": result.get("answer", ""),
                "contexts": result.get("contexts", []),
                "ground_truth": testcase.get("ground_truth", ""),
                "evolution_type": testcase.get("evolution_type", "unknown")
            })

        except Exception as e:
            print(f"   ⚠️  Помилка на запиті {idx}: {e}")
            # Додаємо порожню відповідь
            results.append({
                "question": question,
                "answer": "",
                "contexts": [],
                "ground_truth": testcase.get("ground_truth", ""),
                "evolution_type": testcase.get("evolution_type", "unknown")
            })

    elapsed = time.time() - start_time
    print(f"\n✅ Оброблено: {len(results)} запитів за {elapsed/60:.1f} хв")
    print(f"   Середній час на запит: {elapsed/len(results):.2f}с")

    return results


def evaluate_with_ragas(results: List[Dict]) -> Dict:
    """
    Оцінити результати RAG системи через RAGAS

    Args:
        results: Список результатів з відповідями RAG

    Returns:
        Dict з RAGAS метриками
    """
    print(f"\n📊 RAGAS Evaluation...")
    print(f"   Метрики: Faithfulness, Answer Relevancy, Context Precision, Context Recall")
    print(f"   Це займе 2-5 хвилин для 50 запитів")
    print()

    # Підготовка даних для RAGAS
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for result in results:
        # Пропускаємо порожні відповіді
        if not result["answer"] or not result["contexts"]:
            continue

        data["question"].append(result["question"])
        data["answer"].append(result["answer"])
        data["contexts"].append(result["contexts"])
        data["ground_truth"].append(result["ground_truth"])

    dataset = Dataset.from_dict(data)

    print(f"✅ Підготовлено {len(dataset)} запитів для evaluation")

    # Ініціалізація LLM для RAGAS
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Всі 4 основні метрики
    metrics = [
        faithfulness,        # Чи відповідь базується на контексті?
        answer_relevancy,    # Чи відповідь релевантна запиту?
        context_precision,   # Чи точний retrieved контекст?
        context_recall       # Чи повний retrieved контекст?
    ]

    # Запуск evaluation
    start_time = time.time()

    evaluation_result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )

    elapsed = time.time() - start_time

    # Конвертація результатів
    def _convert_metric(val):
        if isinstance(val, list):
            return float(np.mean(val))
        return float(val)

    scores = {
        "faithfulness": _convert_metric(evaluation_result["faithfulness"]),
        "answer_relevancy": _convert_metric(evaluation_result["answer_relevancy"]),
        "context_precision": _convert_metric(evaluation_result["context_precision"]),
        "context_recall": _convert_metric(evaluation_result["context_recall"]),
        "queries_evaluated": len(dataset),
        "evaluation_time": elapsed
    }

    # Середній score
    scores["average_score"] = np.mean([
        scores["faithfulness"],
        scores["answer_relevancy"],
        scores["context_precision"],
        scores["context_recall"]
    ])

    print(f"\n✅ Evaluation завершено за {elapsed/60:.1f} хв")
    print()
    print(f"{'='*70}")
    print(f"📊 RAGAS МЕТРИКИ")
    print(f"{'='*70}")
    print(f"Faithfulness:        {scores['faithfulness']:.3f}  {'✅' if scores['faithfulness'] > 0.90 else '❌'} (target > 0.90)")
    print(f"Answer Relevancy:    {scores['answer_relevancy']:.3f}  {'✅' if scores['answer_relevancy'] > 0.85 else '⚠️ '} (target > 0.85)")
    print(f"Context Precision:   {scores['context_precision']:.3f}  {'✅' if scores['context_precision'] > 0.80 else '⚠️ '} (target > 0.80)")
    print(f"Context Recall:      {scores['context_recall']:.3f}  {'✅' if scores['context_recall'] > 0.80 else '⚠️ '} (target > 0.80)")
    print(f"{'-'*70}")
    print(f"Average Score:       {scores['average_score']:.3f}")
    print(f"{'='*70}")

    return scores


def analyze_by_query_type(results: List[Dict]) -> Dict:
    """
    Аналіз результатів по типах запитів

    Args:
        results: Результати RAG

    Returns:
        Статистика по типах
    """
    print(f"\n📈 Аналіз по типах запитів:")

    type_stats = {}

    for result in results:
        evo_type = result.get("evolution_type", "unknown")

        if evo_type not in type_stats:
            type_stats[evo_type] = {
                "count": 0,
                "total_time": 0
            }

        type_stats[evo_type]["count"] += 1

    for evo_type, stats in type_stats.items():
        print(f"   {evo_type}: {stats['count']} запитів ({stats['count']/len(results)*100:.1f}%)")

    return type_stats


def save_evaluation_results(
    scores: Dict,
    results: List[Dict],
    output_file: str = "results/ragas_evaluation_full.json"
):
    """
    Зберегти результати evaluation

    Args:
        scores: RAGAS метрики
        results: Результати RAG
        output_file: Шлях до вихідного файлу
    """
    print(f"\n💾 Збереження результатів: {output_file}")

    output = {
        "metadata": {
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(results),
            "ragas_version": "latest"
        },
        "ragas_scores": scores,
        "detailed_results": results[:10]  # Зберігаємо лише перші 10 для прикладу
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Збережено!")


def main():
    """Головна функція - evaluation RAG на синтетичному датасеті"""
    print("="*80)
    print("📊 RAG EVALUATION З СИНТЕТИЧНИМ ДАТАСЕТОМ (RAGAS)")
    print("="*80)
    print()

    if not HAS_RAGAS:
        return

    # Крок 1: Завантаження тестового датасету
    testset = load_synthetic_testset("data/synthetic_testset.json")

    # Крок 2: Ініціалізація RAG системи
    print(f"\n⚙️  Ініціалізація RAG системи...")
    print(f"   Використовуємо: Advanced RAG (найкращий баланс)")

    sys.path.append(str(Path(__file__).parent))
    from advanced_rag.advanced_rag_demo import AdvancedRAG

    rag = AdvancedRAG(
        documents_path="data/pdfs",
        chunk_size=500,
        chunk_overlap=100
    )

    print(f"   Завантаження документів...")
    rag.load_and_process_documents(max_documents=None)  # Всі документи!

    print(f"   Створення індексів...")
    rag.create_embeddings()

    print(f"✅ RAG система готова!")
    print(f"   Документів: {len(rag.chunks)} чанків")

    # Крок 3: Запуск RAG на всіх тестах
    results = run_rag_on_testset(rag, testset)

    # Крок 4: RAGAS Evaluation
    scores = evaluate_with_ragas(results)

    # Крок 5: Аналіз по типах
    type_stats = analyze_by_query_type(results)

    # Крок 6: Збереження
    save_evaluation_results(scores, results)

    # Висновки
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION ЗАВЕРШЕНО!")
    print(f"{'='*80}")
    print()

    if scores["average_score"] >= 0.85:
        print("🎉 ВІДМІННО! Система готова для production")
    elif scores["average_score"] >= 0.75:
        print("⚠️  ПРИЙНЯТНО, але потрібні покращення")
    else:
        print("❌ НИЗЬКА ЯКІСТЬ - потрібна серйозна оптимізація")

    print()
    print("Рекомендації:")
    if scores["faithfulness"] < 0.90:
        print("  - Faithfulness низький → додайте re-ranking або фільтрацію контексту")
    if scores["context_recall"] < 0.80:
        print("  - Context Recall низький → збільште top_k або покращіть retrieval")
    if scores["context_precision"] < 0.80:
        print("  - Context Precision низький → покращіть якість embeddings")
    if scores["answer_relevancy"] < 0.85:
        print("  - Answer Relevancy низький → оптимізуйте LLM prompt")

    print()


if __name__ == "__main__":
    main()
