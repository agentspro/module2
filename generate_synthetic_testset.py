"""
Генерація синтетичного тестового датасету для RAG через RAGAS
==============================================================

RAGAS може автоматично згенерувати реалістичні тестові запити з PDF документів:
- Simple queries (прості факти)
- Reasoning queries (потребують роздумів)
- Multi-context queries (потребують кілька документів)
- Conditional queries (складні умовні запити)

Результат: 50-200+ якісних тестових кейсів замість 6 хардкоджених!
"""

import sys
import os
from pathlib import Path
import json
import time
from dotenv import load_dotenv

# Завантаження змінних середовища з .env (шукаємо в поточній та батьківській директорії)
load_dotenv()  # Спочатку поточна директорія
if not os.getenv('OPENAI_API_KEY'):
    # Якщо не знайшли, шукаємо в батьківській директорії
    load_dotenv(Path(__file__).parent.parent / '.env')

# RAGAS imports для генерації датасетів
try:
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    print("❌ RAGAS не встановлено")
    print("   Встановіть: pip install ragas langchain-openai langchain-community pypdf")
    sys.exit(1)


def load_documents_for_generation(pdf_dir: str = "data/pdfs"):
    """
    Завантажити PDF документи для генерації тестових запитів

    Args:
        pdf_dir: Директорія з PDF файлами

    Returns:
        List of Document objects для RAGAS
    """
    print(f"\n📂 Завантаження документів з {pdf_dir}...")

    loader = DirectoryLoader(
        pdf_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    documents = loader.load()
    print(f"✅ Завантажено: {len(documents)} сторінок з PDF")

    return documents


def generate_testset(
    documents,
    test_size: int = 50,
    distributions: dict = None
):
    """
    Генерує синтетичний тестовий датасет через RAGAS

    Args:
        documents: Список документів (LangChain Documents)
        test_size: Кількість тестових запитів для генерації
        distributions: Розподіл типів запитів

    Returns:
        Generated testset
    """
    print(f"\n🧪 Генерація {test_size} тестових запитів...")
    print("   Це займе 3-5 хвилин для 50 запитів")
    print()

    # Ініціалізація LLM та embeddings для RAGAS
    generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Створюємо генератор
    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings
    )

    # Розподіл типів запитів (за замовчуванням)
    if distributions is None:
        distributions = {
            simple: 0.4,        # 40% - прості фактичні запити
            reasoning: 0.3,     # 30% - потребують роздумів
            multi_context: 0.3  # 30% - потребують кількох документів
        }

    print("📊 Розподіл типів запитів:")
    print(f"   Simple (факти):          {distributions.get(simple, 0)*100:.0f}%")
    print(f"   Reasoning (роздуми):     {distributions.get(reasoning, 0)*100:.0f}%")
    print(f"   Multi-context (складні): {distributions.get(multi_context, 0)*100:.0f}%")
    print()

    # Генерація
    start_time = time.time()

    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=test_size,
        distributions=distributions,
        raise_exceptions=False  # Не падати на помилках
    )

    elapsed = time.time() - start_time

    print(f"✅ Згенеровано: {len(testset)} тестових кейсів за {elapsed:.1f}с")

    return testset


def save_testset(testset, output_file: str = "data/synthetic_testset.json"):
    """
    Зберегти згенерований датасет у JSON

    Args:
        testset: RAGAS testset object
        output_file: Шлях до вихідного файлу
    """
    print(f"\n💾 Збереження датасету: {output_file}")

    # Конвертуємо testset у DataFrame потім у dict
    df = testset.to_pandas()

    # Структура для збереження
    data = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_questions": len(df),
            "source": "RAGAS synthetic generation from PDFs"
        },
        "testcases": []
    }

    # Конвертуємо кожен тестовий кейс
    for idx, row in df.iterrows():
        testcase = {
            "question": row.get("question", ""),
            "ground_truth": row.get("ground_truth", ""),
            "contexts": row.get("contexts", []),
            "evolution_type": row.get("evolution_type", "unknown"),
            "metadata": row.get("metadata", {})
        }
        data["testcases"].append(testcase)

    # Зберігаємо
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Збережено: {len(data['testcases'])} тестових кейсів")

    # Статистика
    print(f"\n📊 Статистика датасету:")
    evolution_counts = df['evolution_type'].value_counts()
    for evo_type, count in evolution_counts.items():
        print(f"   {evo_type}: {count} ({count/len(df)*100:.1f}%)")


def preview_testset(testset, num_examples: int = 5):
    """
    Показати приклади згенерованих тестових кейсів

    Args:
        testset: RAGAS testset
        num_examples: Кількість прикладів для показу
    """
    print(f"\n{'='*80}")
    print(f"📝 ПРИКЛАДИ ЗГЕНЕРОВАНИХ ТЕСТОВИХ КЕЙСІВ (перші {num_examples})")
    print(f"{'='*80}\n")

    df = testset.to_pandas()

    for idx in range(min(num_examples, len(df))):
        row = df.iloc[idx]

        print(f"Кейс #{idx+1}")
        print(f"Тип: {row.get('evolution_type', 'unknown')}")
        print(f"Запит: {row.get('question', '')}")
        print(f"Ground Truth: {row.get('ground_truth', '')[:200]}...")
        print(f"Контексти: {len(row.get('contexts', []))} документів")
        print("-" * 80)
        print()


def main():
    """Головна функція - генерує синтетичний датасет"""
    print("="*80)
    print("🧪 RAGAS: ГЕНЕРАЦІЯ СИНТЕТИЧНОГО ТЕСТОВОГО ДАТАСЕТУ")
    print("="*80)
    print()
    print("Цей скрипт автоматично згенерує реалістичні тестові запити")
    print("з ваших PDF документів за допомогою RAGAS.")
    print()

    if not HAS_RAGAS:
        return

    # Параметри генерації
    PDF_DIR = "data/pdfs"
    TEST_SIZE = 50  # Кількість тестових запитів
    OUTPUT_FILE = "data/synthetic_testset.json"

    print(f"⚙️  Конфігурація:")
    print(f"   PDF директорія: {PDF_DIR}")
    print(f"   Кількість тестів: {TEST_SIZE}")
    print(f"   Вихідний файл: {OUTPUT_FILE}")
    print()

    try:
        # Крок 1: Завантаження документів
        documents = load_documents_for_generation(PDF_DIR)

        if not documents:
            print("❌ Документи не знайдено!")
            print(f"   Переконайтесь що PDF файли є в {PDF_DIR}")
            return

        # Крок 2: Генерація тестового датасету
        testset = generate_testset(
            documents,
            test_size=TEST_SIZE
        )

        # Крок 3: Показати приклади
        preview_testset(testset, num_examples=3)

        # Крок 4: Зберегти
        save_testset(testset, OUTPUT_FILE)

        print("\n" + "="*80)
        print("✅ ГЕНЕРАЦІЯ ЗАВЕРШЕНА!")
        print("="*80)
        print()
        print("Наступні кроки:")
        print(f"1. Перегляньте датасет: {OUTPUT_FILE}")
        print(f"2. Використайте для evaluation: python evaluate_with_synthetic_testset.py")
        print()

    except Exception as e:
        print(f"\n❌ Помилка генерації: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
