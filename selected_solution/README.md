# Selected Candidate 317

Эта папка фиксирует выбранное решение среди кандидатов с подтвержденным visible public score `90.78`.

## Параметры

| Поле | Значение |
|---|---|
| candidate_id | `317` |
| public score | `90.78` |
| local OOF MAPE | `3.274858` |
| delta vs pred265 local OOF | `-0.039716` |
| family | `v15_meta_residual_on_265` |
| base anchor | `pred265` |
| residual model | `huber030` |
| gamma | `0.80` |
| clip | `0.008` |
| seed | `2026` |

## Почему 317

`pred265` остается сильным public anchor: он дал `90.78` на public и имеет local OOF MAPE `3.314574`.

Candidate `317` использует `pred265` как внутренний anchor и применяет небольшой `huber030` residual. Public score остался `90.78`, но local OOF MAPE ниже:

| candidate | public score | local OOF MAPE | delta vs 265 |
|---:|---:|---:|---:|
| `317` | `90.78` | `3.274858` | `-0.039716` |
| `319` | `90.78` | `3.282847` | `-0.031728` |
| `472` | `90.78` | `3.299217` | `-0.015357` |
| `467` | `90.78` | `3.304258` | `-0.010316` |
| `265` | `90.78` | `3.314574` | `0.000000` |

При равном public score выбран кандидат с меньшим локальным OOF MAPE.

## Артефакты

- `manifest.json` - машинно-читаемое описание выбранной конфигурации.
- `reproduction_package_versions.csv` - версии библиотек, зафиксированные для воспроизводимости.
- `submission_candidate_317.csv` - финальный выбранный CSV.

Полный воспроизводимый pipeline находится в:

```text
notebooks/final_reproducible_solution.ipynb
```

Старые snapshot inputs и вспомогательные скрипты удалены из текущей ветки, чтобы не было скрытых источников данных. Полный исследовательский архив доступен в ветке `codex/best-9078-candidate-317-full-archive`.
