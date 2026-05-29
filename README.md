# Growth Gradient

Финальная ветка с выбранным воспроизводимым решением для прогноза РТО магазинов на март 2025.

Полный исследовательский слепок со всеми старыми переборами, промежуточными CSV, OOF/test-prediction таблицами и скриптами сохранен в отдельной ветке:

```text
codex/best-9078-candidate-317-full-archive
```

В этой ветке оставлены только файлы, которые нужны для воспроизведения финального решения, проверки формата, защиты и краткого анализа.

## Финальное Решение

Выбранный вариант:

- candidate: `317`
- public score: `90.78`
- local OOF MAPE: `3.274858`
- family: `v15_meta_residual_on_265`
- base anchor: `pred265`
- residual model: `huber030`
- gamma: `0.80`
- clip: `0.008`
- seed: `2026`

Финальные CSV:

- `submissions/test.csv`
- `selected_solution/submission_candidate_317.csv`
- `submissions/stage2_candidate_317_v15_meta265_huber030_g0p80_c0p008.csv`

Эти файлы синхронизированы как финальные варианты candidate 317.

## Воспроизведение

Главная точка входа:

```text
notebooks/final_reproducible_solution.ipynb
```

Notebook самодостаточный: код вспомогательных этапов встроен внутрь, без импортов локальных project-модулей и без fallback на старые snapshot inputs или `git show`.

При последовательном запуске notebook:

1. выводит версии Python-библиотек;
2. фиксирует `RANDOM_SEED = 2026`;
3. строит промежуточные признаки и meta-inputs из входных данных;
4. обучает финальный residual-этап;
5. формирует `test.csv` и зеркало `submissions/test.csv`;
6. проверяет формат submission.

Для локального запуска на текущих данных нужен файл:

```text
train_2.csv
```

Если организаторы подставляют закрытые входные данные в том же формате, notebook должен строить новый `test.csv` из этих данных без обращения к сохраненным промежуточным CSV.

## Структура

- `notebooks/` - финальный reproducible notebook.
- `selected_solution/` - manifest, версии библиотек и финальный выбранный CSV.
- `submissions/` - только финальный `test.csv` и полное имя candidate 317.
- `results/new_baseline/` - компактные summary финального v15-этапа.
- `results/model_registry/` - компактный журнал public/local результатов.
- `analysis/` - EDA-отчет и графики для защиты.

## Что Убрано

Из текущей ветки удалены старые исследовательские скрипты, сотни candidate CSV, snapshot inputs, OOF/test-prediction таблицы и промежуточные результаты переборов. Они не потеряны: все это осталось в archive-ветке `codex/best-9078-candidate-317-full-archive`.
