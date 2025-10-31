"""
Головний скрипт для запуску всіх RAG демонстрацій
"""
import sys
import subprocess
from pathlib import Path
import time


def print_header(title: str):
    """Виводить заголовок секції"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_demo(script_path: str, demo_name: str) -> bool:
    """
    Запускає окрему демонстрацію

    Args:
        script_path: Шлях до скрипту
        demo_name: Назва демонстрації

    Returns:
        bool: True якщо успішно
    """
    print_header(f"🚀 Запуск: {demo_name}")

    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300  # 5 хвилин timeout
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(result.stdout)
            print(f"\n✅ {demo_name} завершено успішно за {execution_time:.2f}с")
            return True
        else:
            print(f"❌ Помилка в {demo_name}:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏱️  Timeout: {demo_name} перевищив ліміт часу (5 хв)")
        return False
    except Exception as e:
        print(f"❌ Виняток при запуску {demo_name}: {str(e)}")
        return False


def main():
    """Головна функція"""
    print_header("🎯 RAG ДЕМОНСТРАЦІЇ - ЗАПУСК ВСІХ ПРОГРАМ")

    # Список демонстрацій
    demos = [
        {
            "name": "Naive RAG",
            "script": "naive_rag/naive_rag_demo.py",
            "description": "Базова реалізація RAG (~25% точність)"
        },
        {
            "name": "Advanced RAG",
            "script": "advanced_rag/advanced_rag_demo.py",
            "description": "Покращена RAG з 5 техніками (~90% точність)"
        },
        {
            "name": "Hybrid RAG",
            "script": "hybrid_rag/hybrid_rag_demo.py",
            "description": "Гібридний пошук (Dense + Sparse)"
        },
        {
            "name": "Corrective RAG",
            "script": "corrective_rag/corrective_rag_demo.py",
            "description": "RAG з самоперевіркою та виправленням"
        }
    ]

    results = []
    total_start = time.time()

    # Запуск кожної демонстрації
    for i, demo in enumerate(demos, 1):
        print(f"\n{'='*70}")
        print(f"📍 Демонстрація {i}/{len(demos)}: {demo['name']}")
        print(f"📝 {demo['description']}")
        print(f"{'='*70}")

        success = run_demo(demo['script'], demo['name'])
        results.append({
            "name": demo['name'],
            "success": success
        })

        # Пауза між демонстраціями
        if i < len(demos):
            print("\n⏳ Пауза 3 секунди перед наступною демонстрацією...")
            time.sleep(3)

    total_time = time.time() - total_start

    # Підсумок
    print_header("📊 ПІДСУМОК ВИКОНАННЯ")

    print("Результати:")
    successful = 0
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"  {status} {result['name']}")
        if result["success"]:
            successful += 1

    print(f"\nУспішно: {successful}/{len(demos)}")
    print(f"Загальний час: {total_time:.2f}с")

    # Розташування результатів
    print("\n📁 Результати збережено в:")
    results_dir = Path("results")
    if results_dir.exists():
        for result_file in results_dir.glob("*.json"):
            print(f"  - {result_file}")

    if successful == len(demos):
        print("\n🎉 Всі демонстрації виконано успішно!")
        print("\n💡 Наступні кроки:")
        print("  1. Перегляньте результати в директорії results/")
        print("  2. Порівняйте метрики різних підходів")
        print("  3. Експериментуйте з параметрами")
        print("  4. Додайте власні документи в data/corporate_docs/")
        return 0
    else:
        print("\n⚠️  Деякі демонстрації завершились з помилками")
        print("Перевірте логи вище для деталей")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
