# orderbook_encoder — спецификация (контракт модулей)

Проект 4 из `projects.md`: энкодер L2-ордербука. Deep MLP → компактный эмбеддинг,
обучение через предсказание движения mid-price (3 класса DOWN/FLAT/UP, порядок
классов `[DOWN=0, FLAT=1, UP=2]` — как в остальных проектах).

## Источник данных

Прод-сервис warehouse (см. `../docs/data_sources.md`):

- API (без авторизации): `GET {api_base}/data/{SYMBOL}/orderbook` → `{"exchanges": {"bybit": {"days": N, "first": "YYYY-MM-DD", "last": "..."}}}`;
  `GET {api_base}/data/{SYMBOL}/orderbook/{exchange}` → дни; `GET .../{exchange}/{date}` → `[{"file","size_bytes","url"}]`.
- Файлы (анонимный S3): `{s3_base}/{SYMBOL}/orderbook/{exchange}/{YYYY-MM-DD}/{HH}_delta.parquet.zst`
  и `{HH}_snapshot.parquet.zst`. Бывают ротации `{HH}_delta.parquet.zst.1`, `.2`, … (после реконнекта коллектора).
- Это обычные parquet с zstd-компрессией колонок — читаются `pd.read_parquet(path)` напрямую.

Схемы:
- snapshot: `ts:int64(ms), last_update_id:int64, side:category(bid|ask), price:float64, qty:float64` (~200 уровней на сторону).
- delta: `event_time:int64(ms), recv_time:int64, first_update_id:int64, final_update_id:int64, side, price, qty` (qty=0 — уровень удалён). ~500k строк/час.

Восстановление книги на момент T: последний снапшот ≤ T, затем проигрыш дельт по
возрастанию (`event_time`, `final_update_id`) до T.

## Структура проекта (как у соседей: token_first_transformer и др.)

```
orderbook_encoder/
  pyproject.toml          # готов
  configs/default.yaml    # готов
  warehouse/client.py     # агент data-pipeline
  book/book.py            # агент data-pipeline
  book/sampler.py         # агент data-pipeline
  scripts/download_data.py, build_samples.py   # агент data-pipeline
  dataset/orderbook_dataset.py                 # агент model-training
  models/orderbook_mlp.py                      # агент model-training
  training/trainer.py                          # агент model-training
  scripts/train.py                             # агент model-training
  tests/                  # оба агента, файлы не пересекаются
```

Только stdlib + deps из pyproject (numpy, pandas, pyarrow, torch, pyyaml, sklearn).
Для скачивания — `urllib.request` (requests не в зависимостях).

## Контракты

### warehouse/client.py
```python
class WarehouseClient:
    def __init__(self, api_base: str, s3_base: str): ...
    def get_orderbook_info(self, symbol) -> dict        # exchanges -> {days, first, last}
    def list_days(self, symbol, exchange) -> list[str]  # ["2026-06-01", ...]
    def list_files(self, symbol, exchange, date) -> list[dict]  # {file,size_bytes,url}
    def download_day(self, symbol, exchange, date, dest_root: Path,
                     skip_existing=True) -> list[Path]
```
`download_day` кладёт файлы в `{dest_root}/{symbol}/{exchange}/{date}/{filename}`,
качает во временный `.part` и атомарно `os.replace`. Непустой существующий файл —
пропуск. Ретраи на 5xx/таймаут: 3 раза (10/30/60 c).

### book/book.py
```python
class LocalBook:
    def apply_snapshot(self, prices, qtys, sides) -> None   # полная замена состояния
    def apply_delta(self, prices, qtys, sides) -> None      # qty=0 удаляет уровень
    def top_levels(self, depth) -> tuple[list[tuple[float,float]], list[tuple[float,float]]]
        # (bids по убыванию цены, asks по возрастанию), каждый список ≤ depth
    def mid(self) -> float | None                           # None если сторона пуста
    def is_valid(self) -> bool                              # обе стороны непусты и best_bid < best_ask
```
Внутри — dict price→qty на сторону; сортировка только в `top_levels`/`mid`
(семплируем 1 Гц, это дёшево).

### book/sampler.py
```python
def features_from_book(book: LocalBook, depth: int) -> tuple[np.ndarray, float] | None
    # (вектор float32 формы (4*depth,), mid); None если book невалиден
def sample_day(day_dir: Path, depth: int, interval_sec: float) -> dict | None
    # {'ts': int64 (T,) ms, 'features': float32 (T, 4*depth), 'mid': float64 (T,)}
    # None если в дне нет ни одного валидного часа
```
Раскладка фич: `[bid_off×D, bid_qty×D, ask_off×D, ask_qty×D]`, где
`off = |price - mid| / mid`, `qty = log1p(qty)`. Если уровней меньше D — паддинг:
off повторяет самый глубокий доступный уровень, qty = 0.

`sample_day`: часы в отсортированном порядке; на каждый час — если есть снапшот,
применить его (ресинк); дельты часа (включая ротации `.1`, `.2` — конкатенировать
и отсортировать по `event_time, final_update_id`) проигрывать по строкам; часы по
своему clock'у `event_time`. Семпл на каждый тик сетки `interval_sec` (по
event_time): эмитим текущее состояние книги. Невалидная книга → тик пропускается.

### scripts/download_data.py / build_samples.py
CLI (argparse): `--config configs/default.yaml`, `--dates 2026-06-01:2026-06-09`
(диапазон включительно) или `--dates 2026-06-09` (один день); пути и
symbol/exchange — из конфига. `build_samples.py` пишет
`{samples_dir}/{symbol}/{exchange}/{date}.npz` (np.savez_compressed, ключи
`ts, features, mid`), пропускает уже существующие npz если не задан `--force`.

### dataset/orderbook_dataset.py
```python
class OrderbookDataset(torch.utils.data.Dataset):
    def __init__(self, npz_paths: list[Path], horizon_sec: float,
                 threshold_pct: float, interval_sec: float): ...
    # __getitem__ -> (torch.float32 (4*depth,), torch.int64 label)
def build_splits(cfg: dict) -> tuple[OrderbookDataset, OrderbookDataset, OrderbookDataset]
    # по спискам дат cfg['split']['train_days'|'val_days'|'test_days']
```
Метка: `h = round(horizon_sec / interval_sec)`; `ret = mid[i+h]/mid[i] - 1`;
UP(2) если `ret > threshold_pct/100`, DOWN(0) если `< -…`, иначе FLAT(1).
Окна не пересекают границу npz-файла. Если `ts[i+h] - ts[i]` отличается от
`horizon_sec*1000` больше чем вдвое (дырка в данных) — семпл выбрасывается
(индексация валидных пар строится в `__init__`).

### models/orderbook_mlp.py
```python
class OrderbookEncoder(nn.Module):   # input_dim -> hidden_dims[...] -> embedding_dim
    def forward(self, x) -> Tensor   # (B, embedding_dim)
class OrderbookClassifier(nn.Module):
    def __init__(self, encoder: OrderbookEncoder, num_classes: int): ...
    def forward(self, x) -> Tensor   # (B, num_classes) логиты
    def encode(self, x) -> Tensor    # эмбеддинг без головы
```
MLP-блок: Linear → LayerNorm → ReLU (или GELU) → Dropout. Размеры из конфига.

### training/trainer.py + scripts/train.py
```python
def train(cfg: dict) -> dict   # возвращает итоговые метрики
```
AdamW (lr, weight_decay из конфига), CrossEntropy с весами классов (обратно
пропорциональны частотам в train), early stop по val loss
(`early_stop_patience`), device auto (mps → cuda → cpu). Артефакты в
`artifacts/run_YYYYMMDD_HHMMSS/`: `best.pt`, `config.json`, `test_metrics.json`
(`sklearn.metrics.classification_report(..., output_dict=True)` + confusion
matrix) — формат как у соседних проектов.

## Тесты

pytest, без сети, синтетические данные. Прогон:
`/Users/suenot/projects/w_trading/w_training/token_first_transformer/.venv/bin/python -m pytest tests/ -x -q`
из корня `orderbook_encoder` (в этой виртуалке есть torch/pandas/pyarrow/pytest;
свою .venv не создавать — torch тяжёлый).

- data-pipeline: `test_book.py` (снапшот, дельта, удаление уровня qty=0, top_levels,
  mid, is_valid на пустой/скрещенной книге), `test_sampler.py` (синтетические
  parquet-файлы во tmp_path: снапшот+дельты → ожидаемые фичи/mid/паддинг; ротация
  `.1`), `test_client.py` (формирование URL и путей, skip_existing — без сети,
  download мокается через monkeypatch urlopen).
- model-training: `test_dataset.py` (метки UP/FLAT/DOWN по синтетическому mid,
  выбрасывание семплов с дыркой по ts, непересечение границ файлов),
  `test_model.py` (shape forward/encode, backward проходит), `test_trainer.py`
  (1 эпоха на крошечных синтетических npz во tmp_path, артефакты создаются).

## Стиль

Как в соседних проектах: короткие модули, докстринги на английском, type hints,
без лишних абстракций. Импорты внутри проекта — относительно корня проекта
(`from book.book import LocalBook`), как у соседей.
