# Titanic ML Project

Полностью воспроизводимый ML-проект для задачи Titanic: EDA в ноутбуках, основной pipeline — в `main.py`.

## Что есть в проекте
- baseline: `LogisticRegression`
- boosting: `RandomForest`, `GradientBoosting`, `AdaBoost`, `LightGBM`, `CatBoost`, `XGBoost`
- ensembles: `Voting`, `WeightedVoting`, `Stacking`, `Bagging`
- deep learning: `TitanicNN` на PyTorch
- `StratifiedKFold` cross-validation
- сохранение метрик, моделей и submission

## Структура
- `configs/` — YAML-конфиги
- `data/raw/` — `train.csv`, `test.csv`
- `notebooks/` — EDA и черновики
- `src/` — production-код
- `artifacts/` — выходные файлы: модели, метрики, submission

## Запуск
```bash
pip install -r requirements.txt
python main.py --config configs/default.yaml
```

## Что делает main.py
1. Загружает train/test
2. Собирает единый dataframe
3. Выполняет preprocessing и feature engineering
4. Кодирует категории
5. Добавляет feature `cluster`
6. Обучает выбранные модели с CV
7. Выбирает лучшую модель
8. Дообучает лучшую модель на полном train
9. Сохраняет submission и таблицу метрик

## EDA
Отдельный ноутбук должен лежать в `notebooks/EDA.ipynb`.

## Замечания
- Для `CrossEntropyLoss` в DNN softmax в `forward()` не нужен.
- Все артефакты сохраняются в `artifacts/`.
