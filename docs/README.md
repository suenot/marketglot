# Документация w_training

Монорепозиторий с проектами обучения нейросетей на рыночных данных (OHLCV, индикаторы).

## Оглавление

| Раздел | Содержание |
|--------|------------|
| [Карта репозитория](repository.md) | Структура каталогов, связи между проектами |
| [Источники данных](data_sources.md) | Klines локально, L2 ордербук через warehouse API / CryptoHFTData |
| [Colab-ноутбуки](colab.md) | Пути к `.ipynb`, среда, Google Drive, артефакты, пересборка |
| [Порядок запуска экспериментов](training_workflow.md) | Рекомендуемая последовательность и что сравнивать |
| [Обзор проектов](../projects.md) | Краткое описание всех 9 направлений (в корне) |
| [Ресёрч: diffusion-модели](research/diffusion-llms.md) | Diffusion LM (Gemini Diffusion, LLaDA) и diffusion для рынков — направление проектов 7–8 |
| [Спека token-first (design)](superpowers/specs/2026-04-21-token-transformer-trading-agent-design.md) | Исторический design-doc |

## Быстрые ссылки

- **Пять self-contained Colab-ноутбуков** (код внутри, без `pip install` локальных пакетов):
  - `token_first_transformer/token_first_transformer.ipynb`
  - `indicator_tokenizer/indicator_tokenizer.ipynb`
  - `late_fusion_agent/late_fusion_agent.ipynb`
  - `multimodal_encoder/multimodal_encoder.ipynb`
  - `moe_trading_agent/moe_trading_agent.ipynb`

- **Пересборка ноутбуков** из исходников: `python3 tasks/_build_all_notebooks.py` и `python3 tasks/_build_colab_notebook.py` (только `token_first_transformer`).

- **Смоук-тесты** (виртуалка: `token_first_transformer/.venv/bin/python`):

  ```text
  python3 tasks/_smoke_test_notebook.py
  python3 tasks/_smoke_test_all.py
  ```

Планы реализации по дате лежат в `docs/superpowers/plans/`.
