# Reproducibility TODO

Перед финальной отправкой и защитой нужно зафиксировать не только лучший CSV, но и полностью воспроизводимый путь его получения.

## Что уже зафиксировано

- Семейства моделей и эвристик: `results/model_registry/family_registry.csv`.
- Все найденные candidate rows из summary-файлов: `results/model_registry/candidate_registry.csv`.
- Известные public score: `results/model_registry/public_score_log.csv`.
- Скрипты экспериментов в `analysis/new_baseline/` и `analysis/new_strategy/`.
- Сгенерированные summary и candidate CSV в `results/` и `submissions/`.

## Что нужно дозаполнить после выбора финальной модели

1. Точный финальный candidate id и путь к CSV.
2. Точная команда запуска, которая этот CSV создает.
3. Версии Python и библиотек.
4. Seed/число folds/параметры обучения, если финальное решение использует ML.
5. Список входных файлов и их ожидаемые размеры/хэши.
6. Проверка формата submission: `18657` строк, колонки `new_id,rto`, без NaN и отрицательных значений.
7. Краткое описание алгоритма на естественном языке для организаторов.

## Минимальный шаблон для финального решения

```text
Final candidate:
CSV:
Public score:
Local OOF MAPE:
Family:
Main script:
Run command:
Input files:
Output files:
Python:
Libraries:
Seeds:
Notes:
```

## Важное замечание

Часть старых summary использует разные локальные протоколы: clean heuristic validation, March-2024 OOF, proxy MAPE, gated OOF. Поэтому `candidate_registry.csv` - это индекс и навигация, а не абсолютная таблица лидерборда. Для финального выбора нужно сравнить shortlist на одном согласованном локальном протоколе.
