# Selected 90.78 Public Candidate

Цель этой папки - зафиксировать лучший известный вариант среди решений, которые уже дали видимый public score `90.78`, и при этом имеют известный локальный OOF MAPE.

## Выбранное решение

| Поле | Значение |
|---|---|
| candidate_id | `317` |
| public score | `90.78` |
| local OOF MAPE | `3.2748580735` |
| delta vs pred265 local OOF | `-0.0397163531` |
| family | `v15_meta_residual_on_265` |
| model/source | `huber030` |
| gamma | `0.80` |
| clip | `0.008` |
| submission | `selected_solution/submission_candidate_317.csv` |

## Почему не чистый 265

`265` остается главным public anchor: он дал `90.78` на public и имеет известный public MAPE `4.7214609684`. Но его локальный OOF MAPE равен `3.3145744266`.

`317` тоже дал видимый public score `90.78`, но локально лучше:

| candidate | public score | local OOF MAPE | delta vs 265 |
|---:|---:|---:|---:|
| `317` | `90.78` | `3.274858` | `-0.039716` |
| `319` | `90.78` | `3.282847` | `-0.031728` |
| `472` | `90.78` | `3.299217` | `-0.015357` |
| `467` | `90.78` | `3.304258` | `-0.010316` |
| `265` | `90.78` | `3.314574` | `0.000000` |

Поэтому при равном видимом public score текущий критерий выбора - минимальный local OOF MAPE среди уже проверенных public-90.78 решений.

## Как воспроизвести

Рекомендуемый способ для проверки организаторами:

```bash
jupyter notebook notebooks/reproduce_candidate_317.ipynb
```

При последовательном выполнении всех ячеек notebook:

- выводит версии Python-библиотек и полный `pip freeze`;
- фиксирует `RANDOM_SEED = 2026`;
- обучает нужный `huber030` residual expert;
- формирует `selected_solution/reproduced_submission_candidate_317.csv`;
- проверяет byte-to-byte совпадение с `selected_solution/submission_candidate_317.csv`.

Основной скрипт:

```bash
python3 analysis/new_baseline/train_v15_meta_residual_on_265.py
```

Ключевые summary-файлы:

- `notebooks/reproduce_candidate_317.ipynb`
- `selected_solution/inputs/oof_meta_inputs_v11_v14.csv.gz`
- `selected_solution/inputs/test_meta_inputs_v11_v14.csv.gz`
- `results/new_baseline/v15_meta265_search_summary.csv`
- `results/new_baseline/v15_meta265_submission_candidates.csv`
- `results/model_registry/candidate_registry.csv`
- `results/model_registry/family_registry.csv`

Кандидат `317` соответствует строке:

```text
stage2_candidate_317_v15_meta265_huber030_g0p80_c0p008.csv
```

Ветка фиксирует два одинаковых CSV-алиаса:

- `selected_solution/submission_candidate_317.csv` - короткий путь для выбранного образца.
- `submissions/stage2_candidate_317_v15_meta265_huber030_g0p80_c0p008.csv` - полное имя из summary.

## Важное ограничение

Внутри `v15_meta_residual_on_265` есть варианты с локальным MAPE ниже `317`, например `310-316`, но в текущем журнале у них нет подтвержденного public score `90.78`. Они не выбраны как финальный образец, потому что текущая задача - выбрать лучшее решение среди уже подтвержденных public-90.78 кандидатов.
