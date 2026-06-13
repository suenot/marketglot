# Источники рыночных данных

Откуда брать данные для обучения моделей w_training.

## OHLCV-свечи (klines) — локально

- **Путь:** `w_trender/backtests/data/` (43 GB, 262 символа).
- **Гранулярность:** `klines_1m`, `klines_1s`, `klines_100ms`.
- **Формат:** parquet, помесячно — `{SYMBOL}/klines_1m/YYYY-MM.parquet`.

## L2 ордербук — прод warehouse (локально данных НЕТ)

Сервис `w_warehouse/warehouse/` (Go API + collector), прод: `https://warehouse.marketmaker.cc`.
Коллектор пишет L2-стаканы по WebSocket с 6 бирж (bybit, okx, mexc, bitget, kucoin, gate)
с **2026-06-01**, история прирастает ежедневно.

### API (без авторизации)

```text
GET https://warehouse.marketmaker.cc/api/v1/data/{SYMBOL}/orderbook
    → {"exchanges": {"bybit": {"days": 10, "first": "2026-06-01", "last": "2026-06-10"}, ...}}
GET .../orderbook/{exchange}          → список дней
GET .../orderbook/{exchange}/{date}   → [{"file", "size_bytes", "url"}, ...]
```

### Скачивание файлов (анонимный S3)

```text
https://s3.warehouse.marketmaker.cc/warehouse-data/{SYMBOL}/orderbook/{exchange}/{YYYY-MM-DD}/{HH}_delta.parquet.zst
https://s3.warehouse.marketmaker.cc/warehouse-data/{SYMBOL}/orderbook/{exchange}/{YYYY-MM-DD}/{HH}_snapshot.parquet.zst
```

Бывают ротации `{HH}_delta.parquet.zst.1`, `.2`, … (реконнект коллектора).
`.parquet.zst` — это обычный parquet с zstd-компрессией колонок, читается
`pd.read_parquet()` напрямую.

### Модель хранения и схемы

- **Снапшот** (раз в час, ~400 строк = 200 уровней на сторону):
  `ts:int64(ms), last_update_id, side(bid|ask), price, qty`.
- **Дельты** (~500k строк/час, ~3–4 MB zst):
  `event_time:int64(ms), recv_time, first_update_id, final_update_id, side, price, qty`;
  `qty=0` — уровень удалён.
- **Книга на момент T** = последний снапшот ≤ T + проигрыш дельт по
  (`event_time`, `final_update_id`) до T.
- Объём: ~85 MB/день на пару символ-биржа.

### Быстрый путь: rsync с сервера по локальной сети

Если вы в одной сети с server1 (см. `w_server/docs/deployment.md`), данные можно
тянуть напрямую с диска сервера — на порядок быстрее, чем через публичный S3
(~25 MB/с по LAN против ~1.5 MB/с через CDN):

```sh
# server1: LAN 192.168.28.72 (публичный 89.179.247.80), SSH-порт по LAN — 4242
rsync -a --exclude="*.part" -e "ssh -p 4242" \
  root@192.168.28.72:/mnt/second/trender/backtests/data/XRPUSDT/orderbook/bybit/2026-06-05 \
  orderbook_encoder/data/raw/XRPUSDT/bybit/
```

Корень данных на сервере: `/mnt/second/trender/backtests/data/` (тот же layout,
что и в S3: `{SYMBOL}/orderbook/{exchange}/{YYYY-MM-DD}/`). `*.part` — недокачанные
файлы коллектора, исключать. С публичного IP порт 22 тоже работает (`ssh root@89.179.247.80`).

### Покрытие (на 2026-06-10)

| Символ | Биржи | С |
|--------|-------|---|
| XRPUSDT, DRIFTUSDT | 5–6 бирж | 2026-06-01 |
| BTCUSDT | bybit, mexc | 2026-06-02 |

Готовый загрузчик: `orderbook_encoder/scripts/download_data.py`.

## L2 ордербук — вендор CryptoHFTData (глубокая история)

**<https://www.cryptohftdata.com/>** — платный вендор почасовых L2-ордербуков
(binance_futures и др.), история глубже, чем у нашего коллектора. Нужен API-токен
(`CRYPTOHFTDATA_API_TOKEN`).

В warehouse уже есть демон-бэкфиллер `w_warehouse/warehouse/warehouse-collector/collector/hftdata.py`:
качает вендорные часы в то же дерево `{SYMBOL}/{data_type}/{exchange}/{YYYY-MM-DD}/{HH}.parquet.zst`,
цель бэкфилла — до 2025-08-01 (`HFT_BACKFILL_START`). На 2026-06-10 вендорные данные
в API warehouse ещё не видны — при необходимости глубокой истории качать у вендора напрямую.

## Funding rates

В S3-каталоге warehouse лежат `{SYMBOL}_funding.csv` (см. `catalog.json` в корне бакета).

## Каталог всей публичной раздачи

```text
https://warehouse.marketmaker.cc/api/v1/distribution        → дескриптор (S3, IPFS, объёмы)
https://s3.warehouse.marketmaker.cc/warehouse-data/catalog.json → все символы + manifest.json на символ
```

В публичную S3-раздачу klines/trades попадают через publisher, а ордербук — только
через `/orderbook` API-эндпоинты (из bulk-каталога он исключён).
