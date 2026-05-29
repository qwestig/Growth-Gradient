# Growth Gradient

Рабочий репозиторий для задачи прогнозирования РТО магазинов сети Пятерочка на март 2025.

Эта ветка хранит не только финальные кандидаты, но и исследовательские артефакты: скрипты, summary-таблицы, OOF/test predictions и submission CSV. Это сделано для того, чтобы можно было выбрать лучшую модель по локальным метрикам, public leaderboard и воспроизводимости.

## Текущий якорь сравнения

- Current public best visible score: `90.78`.
- Current public MAPE anchor: `4.7214609684`.
- Current local `pred265` OOF MAPE anchor: `3.314574`.
- Текущий public-anchor: candidate `265`, он же `pred265` / temporal204 v11 GRU stack.

Новые или старые кандидаты удобно сравнивать с `pred265`, потому что более ранние clean baseline заметно слабее.

## Быстрая навигация

Главная точка входа для анализа моделей:

- `analysis/model_registry/README.md` - как устроен реестр моделей.
- `analysis/model_registry/MODEL_FAMILY_NOTES.md` - краткая интерпретация семейств моделей.
- `analysis/model_registry/REPRODUCIBILITY_TODO.md` - что нужно дозаполнить после выбора финального решения.
- `results/model_registry/family_registry.csv` - один ряд на семейство моделей.
- `results/model_registry/candidate_registry.csv` - нормализованный индекс кандидатных строк из summary-файлов.
- `results/model_registry/public_score_log.csv` - известные public-score результаты.
- `results/model_registry/registry_snapshot.md` - быстрый markdown-снимок лучших локальных строк.

Обновление реестра после добавления новых summary:

```bash
python3 analysis/model_registry/build_model_registry.py
```

Если появляется новый public score, он вручную добавляется в `PUBLIC_SCORE_LOG` внутри `analysis/model_registry/build_model_registry.py`, после чего реестр пересобирается той же командой.

## Структура репозитория

- `train_2.csv` - исходные обучающие данные.
- `notebooks/` - исследовательские ноутбуки.
- `analysis/new_baseline/` - baseline и ML-эксперименты ранней/средней фазы.
- `analysis/new_strategy/` - поздние альтернативные эвристики, residual-модели и overlay-кандидаты.
- `analysis/model_registry/` - документация и скрипт сборки реестра экспериментов.
- `results/` - summary, diagnostics, OOF predictions, test predictions, model registry.
- `submissions/` - generated candidate CSV. Обычно есть полное имя вида `stage2_candidate_<id>_<description>.csv`; для части отправленных кандидатов есть короткие alias-файлы вроде `472.csv`.

## Известные public результаты

Полный журнал лежит в `results/model_registry/public_score_log.csv`.

Ключевые точки:

- `265`: `90.78`, текущий anchor.
- `390`: `90.59`, aggressive feb-anomaly GRU, хуже anchor.
- `398`: `90.45`, safe feb-anomaly GRU, хуже anchor.
- `430`: `90.58`, uncapped feb-anomaly GRU, хуже anchor.
- `467`: `90.78`, micro overlay on 265, public-neutral.
- `472`: `90.78`, stronger micro overlay on 265, public-neutral.

## Текущая картина по семействам

По `results/model_registry/family_registry.csv` лучшие локальные семейства сейчас:

1. `micro_overlays_on_265` - лучший local OOF, но public-neutral на `467` и `472`.
2. `multimonth_gru_seasonal` - сильное локальное ML-семейство.
3. `multimonth_gru` - более простой GRU residual вариант.
4. `similarity_overlays_on_265` - маленький независимый KNN-similarity signal.
5. `anchor_temporal204_v11_gru_stack` - текущий public-anchor.

Отдельно сохранены проигрышные, но важные для анализа и защиты семейства: `shrunk_yoy`, `share_of_market`, `march_share_q1`, `deseasonalized_normal`, `lifecycle`, `hierarchical_cohort_expert`, `conditional_feb_anomaly`.

## Как читать артефакты

`candidate_registry.csv` является индексом, а не единой абсолютной leaderboard-таблицей. Причина: разные старые эксперименты могли использовать разные локальные протоколы (`clean heuristic validation`, `March-2024 OOF`, proxy MAPE, gated OOF).

Для выбора финального решения стоит сначала собрать shortlist, затем сравнить выбранные модели на одном согласованном локальном протоколе и только после этого фиксировать финальный CSV и воспроизводимость.

## Воспроизводимость финального решения

После выбора финального кандидата нужно заполнить `analysis/model_registry/REPRODUCIBILITY_TODO.md`:

- final candidate id и путь к CSV;
- точная команда запуска;
- Python и версии библиотек;
- seed/fold/training параметры;
- входные файлы и ожидаемые размеры/хэши;
- выходные файлы;
- проверка формата submission.
