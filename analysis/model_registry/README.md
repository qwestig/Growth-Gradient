# Model Experiment Registry

Цель этой ветки - зафиксировать все накопленные семейства решений в одном месте, чтобы дальше выбирать не по памяти и не по отдельным CSV, а по воспроизводимому реестру.

Текущий якорь сравнения:

- public score: `90.78`
- public MAPE: `4.7214609684`
- local `pred265` OOF MAPE: `3.314574`

## Что где лежит

- `analysis/model_registry/build_model_registry.py` - собирает реестр из уже существующих summary-файлов.
- `results/model_registry/family_registry.csv` - один ряд на семейство моделей/эвристик.
- `results/model_registry/candidate_registry.csv` - нормализованный реестр всех найденных кандидатных строк из summary-файлов.
- `results/model_registry/public_score_log.csv` - ручной журнал известных public score.
- `results/model_registry/registry_snapshot.md` - быстрый markdown-снимок лучших локальных строк.
- `analysis/model_registry/MODEL_FAMILY_NOTES.md` - интерпретация семейств: что проверяли, что сработало, что не сработало.
- `analysis/model_registry/REPRODUCIBILITY_TODO.md` - что нужно дозаполнить перед финальным воспроизведением решения.

## Как обновить реестр

После появления новых summary-файлов или новых public submit-результатов:

```bash
python3 analysis/model_registry/build_model_registry.py
```

Если появился новый public score, его нужно вручную добавить в `PUBLIC_SCORE_LOG` внутри `build_model_registry.py`, затем перезапустить скрипт.

## Как читать локальные результаты

Главный файл для выбора кандидатов:

```text
results/model_registry/candidate_registry.csv
```

Важные поля:

- `family_id` - семейство решения.
- `source_file` - откуда взята строка.
- `candidate_id` - номер candidate CSV, если его удалось извлечь.
- `local_mape` - локальная OOF/proxy MAPE из исходного summary.
- `local_delta` - локальное изменение относительно якоря/базы, если исходный summary это содержит.
- `public_score` - известный public score, если кандидат отправлялся.
- `params_json_like` - параметры строки, которые удалось извлечь из summary.

Важно: `local_mape` у разных старых семейств не всегда построен одним и тем же протоколом. Для финального выбора нужно смотреть не только ранжирование, но и протокол в соответствующем скрипте/summary.

## Текущая картина

Лучший public-якорь остается `265`/`pred265`: score `90.78`.

По локальному OOF самые сильные строки сейчас выглядят так:

- `micro_overlays_on_265`: лучший локальный MAPE около `3.2992`, но public `467` и `472` дали тот же видимый score `90.78`.
- `multimonth_gru_seasonal`: локально около `3.3044`, требует аккуратной проверки public/private transfer.
- `multimonth_gru`: локально около `3.3089`.
- `similarity_overlays_on_265`: локально около `3.3097`, новый независимый сигнал, public еще не проверен.
- `cross_similarity_expert`: сам по себе слабый, но полезен как источник независимой поправки.

Семейства вроде `shrunk_yoy`, `share_of_market`, `march_share_q1`, `deseasonalized_normal`, `lifecycle` сохранены не потому, что они лучшие, а потому что они показывают проверенные и отвергнутые гипотезы. Это полезно для защиты: можно объяснить, почему финальная формула не пошла в сторону YoY/Q1/share allocation.

## Следующий шаг

Дальше надо выбрать shortlist моделей по двум осям:

1. Минимальная локальная MAPE на нашем протоколе.
2. Известная устойчивость на public leaderboard.

После выбора shortlist уже нужно фиксировать точные команды запуска, seed, версии библиотек и итоговый CSV, чтобы организаторы могли воспроизвести финальное решение.
