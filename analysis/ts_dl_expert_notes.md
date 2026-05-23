# Notes From TS_DL Expert Notebook

## What the notebook emphasizes

- Use global models across many time series instead of fitting one local model per store.
- Normalize target dynamics before feeding the model:
  - last-known normalization,
  - delta/difference normalization,
  - standard scaling by series/id.
- For ML models, use recursive / Rec-MIMO forecasting with a fixed history window.
- For DL models, DLinear is a strong simple baseline because it explicitly separates trend and residual via moving average.
- Chronos-style zero-shot forecasting can be tested, but it is local per-series and requires a pretrained model download/runtime.

## What already overlaps with our best GRU

The current best `GRU alpha=0.70` already uses two core ideas from the expert notebook:

- Last-known normalization: `rel_log_rto = log(history) - log(last_history_value)`.
- Difference normalization: `log_growth = log(y_t) - log(y_{t-1})`.

So the best model is not trained on fully raw RTO levels. It is closer to the expert recommendation than it first looked.

## DLinear check

I implemented an expert-inspired DLinear residual experiment:

- Same handcrafted March baseline.
- Same March-2024 store-fold validation.
- Same alpha residual application.
- DLinear block decomposes the sequence into moving-average trend and residual.
- Tested moving-average kernels `3`, `5`, and `7`.

Best local result:

- DLinear kernel `7`, alpha `0.70`: MAPE `3.4779`.
- Current GRU alpha `0.70`: MAPE `3.4417`.

DLinear is useful as a sanity check, but it is not stronger than the current GRU branch.

## Blend check

GRU + DLinear blend gives only a tiny local improvement:

- GRU only: `3.4417`.
- Best GRU/DLinear blend: about `3.4406`.

This is too small to be a confident public-test candidate, especially because a previous similarly small GRU/CatBoost blend degraded on public.

## Practical next ideas

- Do not spend time on another generic neural architecture unless it changes the data view.
- Focus on segment-specific bias analysis: region, store size, revenue level, volatility, and momentum clipping.
- Test transformations not as new models, but as diagnostics for groups where GRU over/under-corrects.
- Consider Chronos only if we can run it locally without breaking iteration speed; it is unlikely to use static store features and may be weak for one-month March correction.
