# Colab-ноутбуки

Самодостаточные Jupyter-ноутбуки для Google Colab: весь нужный код **встроен в ячейки**, без `upload` репозитория как пакета. Первой ячейкой ставятся зависимости (`pyarrow`, `polars`, `pyyaml`, `scikit-learn`, `pandas` и т.д.; `torch` на Colab обычно уже есть).

## Имена файлов

| Проект | Путь к ноутбуку |
|--------|-----------------|
| token_first_transformer | `token_first_transformer/token_first_transformer.ipynb` |
| indicator_tokenizer | `indicator_tokenizer/indicator_tokenizer.ipynb` |
| late_fusion_agent | `late_fusion_agent/late_fusion_agent.ipynb` |
| multimodal_encoder | `multimodal_encoder/multimodal_encoder.ipynb` |
| moe_trading_agent | `moe_trading_agent/moe_trading_agent.ipynb` |

Имя файла совпадает с именем проекта, чтобы в Google Drive не путать пять одинаковых `colab_training.ipynb`.

## Среда выполнения

1. **Runtime → Change runtime type → T4 GPU** (рекомендуется).
2. TPU в этом репо **не** поддержан из коробки (нужен `torch_xla` и переработка цикла обучения).
3. **Run all** — ноутбуки рассчитаны на проход с mock-данными без монтирования Drive.

## Данные

### Режим по умолчанию (mock)

В конфиг-ячейке: `USE_MOCK_DATA = True`. Генерируется синтетический OHLCV (геометрическое броуновское движение и т.п. по ноутбуку) под ожидаемую раскладку `DATA_DIR/<SYMBOL>/klines_1m/YYYY-MM.parquet`.

**Ожидаемое качество на mock:** слабое; метрики и PnL не интерпретируйте как «рынок».

### Реальные данные (Google Drive)

1. `USE_MOCK_DATA = False` в конфиг-ячейке.
2. Вызов `drive.mount("/content/drive")` обычно уже есть в ветке «реальные данные».
3. Укажите `DATA_DIR`, например: `/content/drive/MyDrive/trading_data`.
4. Структура: `trading_data/BTCUSDT/klines_1m/2024-01.parquet` и т.д. (см. комментарии внутри ноутбука).
5. При необходимости расширьте `train_months` / `val_months` / `test_months` в `CFG` / `TRAIN_CFG`.

## Артефакты после прогона

В конце каждого ноутбука — ячейка **сохранения** в `ARTIFACTS_ROOT` (таймстамп `run_YYYYMMDD_HHMMSS`).

- Если **Google Drive смонтирован** (через `USE_MOCK_DATA = False` или ручной `drive.mount` до сохранения): по умолчанию  
  `My Drive/w_training/<project_name>/run_<...>/`
- Если Drive **не** смонтирован:  
  `/content/artifacts/<project_name>/` (эфемерно, пропадёт при остановке рантайма).

### Что обычно внутри `run_*/`

| Содержимое | Назначение |
|------------|------------|
| `checkpoints/*.pt` | Веса (`torch.load(..., weights_only=True)` где применимо) |
| `tokenizers/` | `delta_params.json`, `*_boundaries.npy`, папка индикаторов (`*.npy`) |
| `config.json` | Полный `CFG` или `MODEL_CFG`+`TRAIN_CFG` (MoE) |
| `train_metrics.json` | По эпохам (где реализовано, напр. `token_first_transformer`) |
| `test_metrics.json` | `classification_report` (dict) + confusion matrix |
| `backtest.json` | Только в `token_first_transformer` — PnL, Sharpe, просадка |
| `predictions.npz` | Массивы предсказаний и меток на тесте |
| `boundaries/` | Копия `BOUNDARIES_DIR` у `indicator_tokenizer` |

Режим inference в коде: PyTorch-метод `train` с `mode=False` (эквивалент eval-режима), намеренно без короткого алиаса в тексте ноутбука из-за линтеров/сканеров.

## Как залить ноутбук в Colab

- **File → Upload notebook** в [colab.research.google.com](https://colab.research.google.com)
- Либо положить `.ipynb` в **Google Drive** и **Open with → Colaboratory**
- Либо **GitHub**:  
  `https://colab.research.google.com/github/<user>/<repo>/blob/main/<path>.ipynb`

## Пересборка ноутбуков из кода в репозитории

После правок в `.py` перегенерировать:

```bash
# первый проект
python3 tasks/_build_colab_notebook.py

# остальные четыре
python3 tasks/_build_all_notebooks.py
```

См. также [tasks/](../tasks/) — `_smoke_test_notebook.py`, `_smoke_test_all.py`.

## Безопасность

- Не коммитьте **`/.env` с секретами**; в репозитории допустим **`.env.example`**.
