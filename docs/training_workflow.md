# Порядок запуска и сравнение моделей

## Важно

Пять Colab-ноутбуков **не зависят** друг от друга: в каждом токенизаторы и модель инициализируются и обучаются внутри. Порядок ниже — **логический** (от диагностики к сложным архитектурам), а не технический.

## Рекомендуемая последовательность

1. **`indicator_tokenizer`** — быстро (минуты): фит квантилей по 6 индикаторам, проверка распределения токенов. Модель не обучается; это sanity-check данных.
2. **`token_first_transformer`** — baseline на одних токенах свечи; есть **backtest** (`backtest.json`) — ориентир по Sharpe/PnL на вашем сплите.
3. **`late_fusion_agent`** — независимые Model A (свечи) + B (индикаторы) + meta; сравнение с baseline по `test_metrics.json`.
4. **`multimodal_encoder`** — раннее слияние модальностей в одном трансформере; сравнение с (2) и (3).
5. **`moe_trading_agent`** — тяжелее и чувствительнее к данным; имеет смысл после того, как (2)–(4) показали, что сигнал на тесте не случайный.

**Параллельно:** на разных вкладках Colab / разных аккаунтах — можно, если лимит GPU позволяет; результаты в Drive независимы.

## Таблица: что с чем сравнивать

| Модель (ноутбук) | Главные метрики | С чем сравнивать |
|------------------|----------------|------------------|
| indicator_tokenizer | гистограммы токенов, `vocab_sizes` | — (не классификация) |
| token_first_transformer | `backtest.json` + `test_metrics.json` | — baseline |
| late_fusion_agent | `test_metrics.json` | token_first |
| multimodal_encoder | `test_metrics.json` | late_fusion, token_first |
| moe_trading_agent | `test_metrics.json` | multimodal, token_first |

Прямой финансовой метрики в `backtest` есть только у **token_first**; остальные — классификация на тесте. Для единообразия смотрите `macro avg` / `f1` в `test_metrics.json` у всех `run_*`.

## Быстрый пример агрегации (в Colab)

```python
from pathlib import Path
import json

root = Path("/content/drive/MyDrive/w_training")
for proj in ["token_first_transformer", "late_fusion_agent", "multimodal_encoder", "moe_trading_agent"]:
    runs = sorted((root / proj).glob("run_*"))
    if not runs:
        print(proj, "— нет run_*"); continue
    r = json.loads((runs[-1] / "test_metrics.json").read_text())["report"]
    print(proj, "macro F1 =", f'{r["macro avg"]["f1-score"]:.4f}', "| run =", runs[-1].name)
```

## Ограничения

- **Mock-данные:** обучающие цифры и confusion matrix **не** отражают реальный рынок; используйте для проверки пайплайна.
- **Реальные данные:** согласуйте сплит по времени, утечки (`target` из будущего), комиссии в бэктесте (у отдельных ноутбуков бэктеста может не быть).
- **GPU:** T4 обычно достаточно; при росте `hidden_dim` / длины последовательности смотрите OOM.

Подробности по путям и артефактам: [colab.md](colab.md).
