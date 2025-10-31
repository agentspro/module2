# Контрольний список для завантаження на GitHub

## ✅ ОБОВ'ЯЗКОВІ файли

### Код (демонстрації)
- [ ] `naive_rag/naive_rag_demo.py`
- [ ] `naive_rag/__init__.py`
- [ ] `advanced_rag/advanced_rag_demo.py`
- [ ] `advanced_rag/__init__.py`
- [ ] `bm25_rag/bm25_rag_demo.py`
- [ ] `bm25_rag/__init__.py`
- [ ] `faiss_rag/faiss_rag_demo.py`
- [ ] `faiss_rag/__init__.py`
- [ ] `hybrid_rag/hybrid_rag_demo.py`
- [ ] `hybrid_rag/__init__.py`
- [ ] `corrective_rag/corrective_rag_demo.py`
- [ ] `corrective_rag/__init__.py`

### Утиліти
- [ ] `utils/data_loader.py`
- [ ] `utils/__init__.py`

### Дані
- [ ] `data/test_queries_unified.json` (уніфікований датасет)
- [ ] `data/test_queries.json` (базовий датасет)
- [ ] `data/pdfs/README.md` (інструкція про PDF)

### Результати
- [ ] `results/naive_rag_results.json`
- [ ] `results/advanced_rag_results.json`
- [ ] `results/bm25_rag_results.json`
- [ ] `results/faiss_rag_results.json`
- [ ] `results/hybrid_rag_results.json`
- [ ] `results/corrective_rag_results.json`
- [ ] `results/final_comparison_report.txt`

### Скрипти
- [ ] `run_all_tests.sh` (БЕЗ API ключа!)
- [ ] `run_all_tests_v2.sh` (БЕЗ API ключа!)
- [ ] `check_progress.sh`
- [ ] `compare_all_rag_approaches.py`

### Документація
- [ ] `README_GITHUB.md` → перейменувати на `README.md`
- [ ] `UNIFIED_DATASET_README.md`
- [ ] `QUICK_START_UNIFIED.md`
- [ ] `.gitignore`

## ❌ НЕ ЗАВАНТАЖУВАТИ

### Секрети та ключі
- [ ] ~~Файли з API ключами~~
- [ ] ~~`.env` файли~~
- [ ] ~~Будь-які credentials~~

### Логи та тимчасові
- [ ] ~~`/tmp/*.log`~~
- [ ] ~~`*.log`~~
- [ ] ~~`*.tmp`~~

### Великі файли
- [ ] ~~`data/pdfs/*.pdf`~~ (занадто великі, студенти завантажать свої)

### Застарілі файли
- [ ] ~~`*_demo_clean.py`~~ (вже видалені)
- [ ] ~~`*_results_clean.json`~~ (вже видалені)

### Python артефакти
- [ ] ~~`__pycache__/`~~
- [ ] ~~`*.pyc`~~
- [ ] ~~`*.pyo`~~

### IDE
- [ ] ~~`.vscode/`~~
- [ ] ~~`.idea/`~~

## 🔧 ПЕРЕВІРКИ перед commit

1. **Видалено всі API ключі?**
   ```bash
   grep -r "sk-proj-" . --exclude-dir=.git
   # Має повернути: нічого
   ```

2. **Оновлено всі імпорти?**
   ```bash
   grep -r "_demo_clean" . --exclude-dir=.git
   # Має повернути: нічого
   ```

3. **Є .gitignore?**
   ```bash
   ls -la .gitignore
   ```

4. **Виконуються тести?**
   ```bash
   # Перевірте, що хоча б один demo запускається
   python naive_rag/naive_rag_demo.py --help
   ```

## 📦 СТРУКТУРА для GitHub

```
rag-demos-comparison/
├── .gitignore
├── README.md                          # Головний README
├── UNIFIED_DATASET_README.md
├── QUICK_START_UNIFIED.md
├── GITHUB_CHECKLIST.md               # Цей файл (опціонально)
│
├── naive_rag/
│   ├── __init__.py
│   └── naive_rag_demo.py
├── advanced_rag/
│   ├── __init__.py
│   └── advanced_rag_demo.py
├── bm25_rag/
│   ├── __init__.py
│   └── bm25_rag_demo.py
├── faiss_rag/
│   ├── __init__.py
│   └── faiss_rag_demo.py
├── hybrid_rag/
│   ├── __init__.py
│   └── hybrid_rag_demo.py
├── corrective_rag/
│   ├── __init__.py
│   └── corrective_rag_demo.py
│
├── utils/
│   ├── __init__.py
│   └── data_loader.py
│
├── data/
│   ├── test_queries_unified.json     # 100 запитів
│   ├── test_queries.json             # Базовий
│   └── pdfs/
│       └── README.md                 # "Помістіть ваші PDF тут"
│
├── results/
│   ├── naive_rag_results.json
│   ├── advanced_rag_results.json
│   ├── bm25_rag_results.json
│   ├── faiss_rag_results.json
│   ├── hybrid_rag_results.json
│   ├── corrective_rag_results.json
│   └── final_comparison_report.txt   # Головний звіт!
│
├── run_all_tests.sh                  # БЕЗ API ключа
├── run_all_tests_v2.sh              # БЕЗ API ключа
├── check_progress.sh
└── compare_all_rag_approaches.py
```

## 🚀 Команди для завантаження

```bash
# 1. Ініціалізація Git (якщо ще не зроблено)
git init

# 2. Додати .gitignore
git add .gitignore

# 3. Додати всі необхідні файли
git add README.md UNIFIED_DATASET_README.md QUICK_START_UNIFIED.md
git add */
git add utils/ data/ results/
git add *.sh *.py

# 4. Перевірка (що буде додано)
git status

# 5. Commit
git commit -m "Initial commit: RAG demos comparison with 6 approaches

- 6 working RAG implementations (Naive, Advanced, BM25, FAISS, Hybrid, Corrective)
- Unified dataset with 100 test queries
- Complete test results on 50 queries
- Comprehensive comparison report
- Ready-to-use demo scripts"

# 6. Додати remote (замініть на ваш URL)
git remote add origin https://github.com/your-username/rag-demos-comparison.git

# 7. Push
git push -u origin main
```

## 📝 Що побачать студенти

1. **README.md** - Головна сторінка з:
   - Описом проекту
   - Результатами порівняння
   - Інструкціями швидкого старту
   - Рекомендаціями

2. **6 робочих демонстрацій** - можна запустити одразу

3. **Готові результати** - results/*.json (не треба чекати 25 хвилин)

4. **Детальний звіт** - results/final_comparison_report.txt

5. **Уніфікований датасет** - для своїх експериментів

## ✨ Додатково (опціонально)

- [ ] `LICENSE` - ліцензія (MIT або освітня)
- [ ] `requirements.txt` - список залежностей
- [ ] `CONTRIBUTING.md` - як студенти можуть допомогти
- [ ] `.github/workflows/` - CI/CD (якщо потрібно)

