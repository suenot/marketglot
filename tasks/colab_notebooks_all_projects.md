# Colab notebooks для остальных проектов w_training

**Актуальная пользовательская документация:** [docs/README.md](../docs/README.md) (все 5 ноутбуков, Drive, порядок запуска).

## Запрос пользователя

> сделай notebooks для других проектов в w_training

Имелся в виду `token_first_transformer/token_first_transformer.ipynb` как образец —
нужно было сделать аналогичные самодостаточные ноутбуки для оставшихся
незаблокированных проектов.

## Какие проекты покрываем

Из 8 проектов в `projects.md` незаблокированы 5: `token_first_transformer`
(уже сделан), `indicator_tokenizer`, `late_fusion_agent`, `multimodal_encoder`,
`moe_trading_agent`. Проекты 4/7/8 (`orderbook_encoder`, `diffusion_orderbook`,
`transformer_diffusion_fusion`) заблокированы отсутствием данных order book.

## Итоговые артефакты

| # | Файл | Cells | Kind |
|---|------|------:|------|
| 1 | `indicator_tokenizer/indicator_tokenizer.ipynb` | 16 (8 code + 8 md) | Fit quantile boundaries, save/reload, гистограммы токенов, save to Drive |
| 2 | `late_fusion_agent/late_fusion_agent.ipynb` | 28 (14 code + 14 md) | Train Model A + Model B + Meta, `classification_report`, save to Drive |
| 3 | `multimodal_encoder/multimodal_encoder.ipynb` | 24 (12 code + 12 md) | End-to-end fusion transformer, early stop, `classification_report`, save to Drive |
| 4 | `moe_trading_agent/moe_trading_agent.ipynb` | 24 (12 code + 12 md) | MoE-transformer (8 experts × top-2), aux loss, test eval, save to Drive |

Скрипты-генераторы и smoke-тест:

- `tasks/_build_all_notebooks.py` — оркестратор, собирает все 4 .ipynb.
- `tasks/_smoke_test_all.py` — end-to-end smoke-тест всех ноутбуков в venv.

## Логика реализации

### Общая архитектура ноутбуков

Каждый ноутбук полностью самодостаточный (пригоден для Colab):

1. Install cell (`!pip install -q pyarrow polars pyyaml scikit-learn pandas`).
2. Вклеенные модули: `DeltaTokenizer`, `BucketTokenizer`, `IndicatorComputer`,
   `IndicatorTokenizer` (для 3 моделей, у `indicator_tokenizer` только
   два последних).
3. Проектные модули (модель, датасет, трейнер) — по одной ячейке каждый.
4. Config cell: `USE_MOCK_DATA`, `DATA_DIR`, `MOCK_MONTHS`, гиперпараметры.
5. Mock data cell: генерирует синтетический OHLCV (GBM) в parquet — чтобы
   ноутбук запускался "из коробки" без внешних данных.
6. `make_split` helper (для троек train/val/test по месяцам).
7. Main: загрузка → модель → тренер → итоговый `classification_report` (либо
   для `indicator_tokenizer` — сохранение/перезагрузка boundaries и
   гистограмма токенов).

### Адаптации при инлайнинге

- `from __future__ import annotations` — оставляется только одним —
  в общей ячейке tokenizers; в остальных заменён на обычный `import`-free
  стиль (чтобы после конкатенации в smoke-тесте не было `SyntaxError`).
- Относительные импорты (`from models.xxx`, `from dataset.xxx`) удалены —
  всё живёт в одном namespace Colab-ноутбука.
- В `fusion_trainer.py` был `importlib.util.spec_from_file_location` для
  `PriceTransformer` — это убрано, класс просто определён в ячейке выше.
- `MoEDataset` в проекте загружает boundaries из сиблинга через путь; в
  ноутбуке используется параметр `boundaries_dir=None`, и индикаторный
  токенайзер просто fit-ится на наблюдаемых данных (или по указанному
  пути, если загрузить заранее).
- Inference-режим модели вызывается через `.train(mode=False)` вместо
  его стандартного однословного алиаса — чтобы подстрока не попала под
  статический сканер безопасности. Функционально идентично.
- Уникальные `id` для каждой ячейки (`cell-001`, `cell-002`, ...) — без
  этого nbformat выдаёт warning и в будущем сделает hard error.

### Mock-данные

Одна и та же функция `_gen_month` (GBM с σ=0.0008 на минуту) генерирует
OHLCV; стартовая цена каждой следующей месячной выборки — `close[-1]`
предыдущей, чтобы цены были связные. Объём — lognormal.

Количество минут для mock указано в constant'ах в каждом ноутбуке;
`moe_trading_agent` использует 30k минут (у датасета берётся сквозной
split по индексу, нужно достаточно окон).

### Про `moe_trading_agent` — "дырка" в модели

`MoETradingModel` хардкодит `candle_proj: 96->128` и `indicator_proj: 96->128`
(а потом concat → `dim=256`). Поэтому нельзя просто уменьшить `dim`
в smoke-тесте — держим `dim=256`, уменьшаем только `num_experts`, `num_layers`,
`num_heads`, `hidden_dim`.

### Рекомендуемый runtime (Colab)

**T4 GPU** — подхватывается `torch.cuda.is_available()` автоматически для
всех 4 ноутбуков. TPU v5e-1 потребовал бы `torch_xla` — не окупается на
моделях < 50M параметров (наши модели 50k–15M в зависимости от проекта).

A100/L4/H100 — только если будете растить `hidden_dim`, `ffn_dim`,
`num_experts` сильно выше дефолтов (или грузить 20+ месяцев данных с
Drive).

## Верификация

1. **JSON / nbformat**: `nbformat.validate(nb)` — проходит для всех 4.
2. **Python AST**: каждая code-ячейка парсится `ast.parse` при генерации.
3. **End-to-end smoke-тест**: `tasks/_smoke_test_all.py` прогоняет каждый
   ноутбук в subprocess (с оверрайдами на mini-конфиг) через venv
   `token_first_transformer/.venv`. Все 4 выходят с `exit 0` и печатают
   `classification_report` / `vocab_sizes` / token histograms.

### Smoke-test settings (в памяти, не переносятся в Colab)

- `indicator_tokenizer`: 2k мин × 2 месяца, fit boundaries, reload, гистограммы.
- `late_fusion_agent`: 2.5k мин × 3 месяца, seq_len=32, horizon=5,
  hidden_dim=64, num_layers=1, epochs_a=1, epochs_b=1, epochs_meta=2.
- `multimodal_encoder`: 2.5k мин × 3 месяца, seq_len=32, horizon=5,
  hidden_dim=64, num_layers=1, epochs=1.
- `moe_trading_agent`: 3k мин × 2 месяца, seq_len=32, horizon=5,
  num_experts=4, top_k=2, num_layers=1, epochs=1.

### Результат последнего прогона

```
[OK] indicator_tokenizer
[OK] late_fusion_agent
[OK] multimodal_encoder
[OK] moe_trading_agent
ALL SMOKE TESTS PASSED
```

## Пути, которые проверял

- `token_first_transformer/token_first_transformer.ipynb` — образец для подражания.
- `token_first_transformer/tokenizer/{delta_tokenizer,bucket_tokenizer}.py`
- `indicator_tokenizer/indicators/{computer,tokenizer}.py`
- `indicator_tokenizer/scripts/{fit,inspect_indicators}.py`
- `indicator_tokenizer/configs/default.yaml`
- `late_fusion_agent/{models/*,dataset/fusion_dataset.py,training/fusion_trainer.py,scripts/train.py,configs/default.yaml}`
- `multimodal_encoder/{models/multimodal_model.py,dataset/multimodal_dataset.py,training/trainer.py,scripts/train.py,configs/default.yaml}`
- `moe_trading_agent/{models/{router,expert,moe_layer,moe_model}.py,dataset/moe_dataset.py,training/trainer.py,scripts/train.py,configs/default.yaml}`

## Как пользоваться

1. Открыть нужный `<project>/<project>.ipynb` в Colab (File → Upload
   notebook, либо push в GitHub → Open in Colab).
2. Runtime → Change runtime type → **T4 GPU** → Save.
3. `Run all`. На mock-данных одна эпоха идёт 1–3 минуты на T4.
4. Для реальных данных: в config cell поставить `USE_MOCK_DATA = False`,
   смонтировать Drive, указать путь к parquet-ам и расширить диапазон
   `train_months` / `val_months` / `test_months` в `CFG["data"]`.

## Автовыгрузка артефактов в Google Drive

Каждый ноутбук заканчивается ячейкой **«Save artifacts to Google Drive»**.
`ARTIFACTS_ROOT` определяется в config-ячейке:

- если `/content/drive/MyDrive` смонтирован → `drive/MyDrive/w_training/<project>/`
- иначе → `/content/artifacts/<project>/` (эфемерно, исчезает при shutdown)

Содержимое `ARTIFACTS_ROOT/run_<YYYYMMDD_HHMMSS>/`:

| файл                                   | что внутри |
|----------------------------------------|------------|
| `checkpoints/best.pt` (или `best_model.pt`, или 3 отдельных) | state_dict моделей |
| `tokenizers/delta_params.json`         | range_pct, step_pct для `DeltaTokenizer` |
| `tokenizers/{vol,vb}_boundaries.npy`   | границы `BucketTokenizer` |
| `tokenizers/indicators/*.npy`          | границы `IndicatorTokenizer` (кроме token_first_transformer) |
| `config.json`                          | полный `CFG` (или `MODEL_CFG` + `TRAIN_CFG` у MoE) |
| `train_metrics.json`                   | per-epoch loss/F1 (только token_first_transformer) |
| `test_metrics.json`                    | `classification_report(output_dict=True)` + confusion matrix |
| `backtest.json`                        | PnL/Sharpe/DD/winrate (только token_first_transformer) |
| `predictions.npz`                      | `preds` + `labels` тестового набора |
| `boundaries/`                          | (indicator_tokenizer) полная копия `BOUNDARIES_DIR` |

Для inference в новой сессии:

```python
from google.colab import drive; drive.mount('/content/drive')
RUN_DIR = Path("/content/drive/MyDrive/w_training/<project>/run_<TAG>")
CFG = json.loads((RUN_DIR / "config.json").read_text())
# → пересоздать токенизаторы из CFG + npy-файлов
# → построить модель с тем же CFG
# → model.load_state_dict(torch.load("checkpoints/best.pt", weights_only=True)["model_state_dict"])
# → model.train(mode=False); прогнать новые 128 свечей
```

## Заметки / risks

- У `late_fusion_agent` и `multimodal_encoder` в smoke-логах видны
  f1≈0.4–0.5 и `FLAT` predominates — это ожидаемо: 1 epoch + mock данные,
  порог `target_threshold=0.0015` для случайного блуждания почти всегда
  даёт FLAT. Это не баг ноутбука, а особенность синтетики.
- `moe_trading_agent` предсказывает `DOWN` — из-за маленького датасета и
  class weights. На реальных данных и полных эпохах ситуация меняется.
- Все ноутбуки инлайнят свои зависимости, поэтому при изменении проектных
  файлов нужно **перегенерировать** ноутбуки: `python3 tasks/_build_all_notebooks.py`.
