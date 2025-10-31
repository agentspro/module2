#!/bin/bash

# ВАЖЛИВО: Встановіть ваш OpenAI API ключ перед запуском
# export OPENAI_API_KEY="your-api-key-here"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ПОМИЛКА: OpenAI API ключ не встановлено!"
    echo "Встановіть змінну середовища: export OPENAI_API_KEY='your-key'"
    exit 1
fi

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Запуск всіх RAG тестів послідовно"
echo "=========================================="
echo ""

echo "[1/6] Naive RAG..."
python -u naive_rag/naive_rag_demo.py 2>&1 | tee -a /tmp/all_tests_output.log
echo ""

echo "[2/6] Advanced RAG..."  
python -u advanced_rag/advanced_rag_demo.py 2>&1 | tee -a /tmp/all_tests_output.log
echo ""

echo "[3/6] BM25 RAG..."
python -u bm25_rag/bm25_rag_demo.py 2>&1 | tee -a /tmp/all_tests_output.log
echo ""

echo "[4/6] FAISS RAG..."
python -u faiss_rag/faiss_rag_demo.py 2>&1 | tee -a /tmp/all_tests_output.log
echo ""

echo "[5/6] Hybrid RAG..."
python -u hybrid_rag/hybrid_rag_demo.py 2>&1 | tee -a /tmp/all_tests_output.log
echo ""

echo "[6/6] Corrective RAG..."
python -u corrective_rag/corrective_rag_demo.py 2>&1 | tee -a /tmp/all_tests_output.log
echo ""

echo "=========================================="
echo "Всі тести завершено! Порівняння результатів..."
echo "=========================================="
python -u compare_all_rag_approaches.py 2>&1 | tee -a /tmp/all_tests_output.log
