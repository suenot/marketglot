# Карта репозитория

## Назначение

`w_training` — набор **изолированных Python-проектов** (каждый в своей папке) для обучения классификаторов движения цены (UP / FLAT / DOWN) и вспомогательных токенизаторов индикаторов. Общий источник данных по умолчанию описан в [projects.md](../projects.md) (парquet с `w_trender` и т.п.); в Colab-ноутбуках по умолчанию используется **синтетика**, чтобы прогон был «из коробки».

## Структура верхнего уровня

| Путь | Назначение |
|------|------------|
| `<project_name>/` | Код, конфиги, тесты, **Colab-ноутбук** `<project_name>.ipynb` |
| `docs/` | Документация (этот раздел) и черновые планы/спеки |
| `tasks/` | Скрипты генерации Colab, смоук-тесты, заметки по задачам |
| `projects.md` | Краткое резюме всех направлений и таблица статусов |

## Семь учебных проектов (код + тесты)

| Проект | Суть | Colab-файл |
|--------|------|------------|
| `token_first_transformer` | Трансформер на токенах свечи (delta + vol + volume buckets) + бэктест | `token_first_transformer/token_first_transformer.ipynb` |
| `indicator_tokenizer` | Квантильные границы для индикаторов, сохранение `boundaries/` | `indicator_tokenizer/indicator_tokenizer.ipynb` |
| `late_fusion_agent` | Две отдельные модели (свечи + индикаторы) + meta-модель | `late_fusion_agent/late_fusion_agent.ipynb` |
| `multimodal_encoder` | Один end-to-end трансформер по свечам и индикаторам | `multimodal_encoder/multimodal_encoder.ipynb` |
| `moe_trading_agent` | MoE-трансформер, sparse top-k эксперты | `moe_trading_agent/moe_trading_agent.ipynb` |
| `orderbook_encoder` | L2-ордербук: загрузка с warehouse, реконструкция книги, MLP-энкодер | — (ноутбука пока нет) |

Данные ордербука — через прод warehouse API / CryptoHFTData, см. [data_sources.md](data_sources.md).
**Не начаты:** `diffusion_orderbook`, `transformer_diffusion_fusion` (блокер по данным снят, см. [projects.md](../projects.md)).

## Граф идейных зависимостей (исходный код)

Исходники `late_fusion` и `multimodal` опираются на **идеи** `token_first_transformer` и `indicator_tokenizer` (ретушь и импорты), но **Colab-ноутбуки не требуют** предварительного запуска друг друга: всё встроено.

```text
token_first_transformer ──┐
                          ├── multimodal_encoder ── moe_trading_agent
indicator_tokenizer ──────┘
                          ├── late_fusion_agent
```

## Конвенции

- **Данные:** `YYYY-MM.parquet`, путь `.../<SYMBOL>/klines_1m/`.
- **Стек:** PyTorch, Python ≥ 3.11 (локально; в Colab — версия среды Google).
- **Конфиги:** YAML в репо; в Colab — inline `CFG` / `MODEL_CFG` в ячейке конфигурации.

Полная таблица тестов и статусов — в [projects.md](../projects.md).
