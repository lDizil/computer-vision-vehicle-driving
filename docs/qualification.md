# Контролируемая визуальная квалификация Slow

Этот модуль проверяет, зависит ли структурированный ответ VLM от текущей картинки и конкретного кандидата. Он не обучает Reason2Drive, не заменяет driving benchmark и не включает semantic veto в `HybridDriveAgent`.

## Владение и поток данных

```text
known camera geometry + static boxes + candidate paths
    → raw RGB views → projected candidate overlays
    → SlowContext (without labels/titles)
    → fixed projected prompt + local VLM + strict parser
    → complete model × case × seed records
    → metrics + counterfactual checks → standalone HTML
```

- `geometry.camera` владеет pinhole/SE3 проекцией, near-plane/viewport clipping и resize калибровки.
- `qualification.cases/render` владеют только контролируемыми сценами и явно объявленными ожидаемыми исходами.
- `slow.llama` владеет local transport и CPU/Metal execution profile; `slow.protocol` — строгим output contract. Варианты grammar ограничивают связь status/IDs; independent parser всё равно проверяет ответ.
- `qualification.runner` выполняет offline-ожидание с bounded worker; не вызываем его из fast decision path.
- `qualification.metrics` владеет полнотой матрицы, знаменателями и diagnostic gate. HTML только отображает пересчитанный summary, не интерпретирует Safety policy.

## Геометрия

Ego: x вперёд, y влево, z вверх; camera: x вправо, y вниз, z вперёд. `ego_to_camera` — proper SE3; K имеет положительные fx/fy, поддерживаем skew и principal point вне кадра. Матрицы owned/immutable после pickle. Resize использует PIL pixel-centre convention: `u'=(u+0.5)*sx-0.5`.

Соседние отрезки сначала обрезаем по near plane, затем по viewport; невидимые части не соединяем. Проекция свыше `1e12` pixels и near-crossing с `depth/near>1e12` явно отвергаются как численно неустойчивые. Входы для реальной камеры должны иметь проверенную калибровку; synthetic rig не выдаём за NAVSIM extrinsics.

## Что именно размечено

Полный набор — 10 детерминированных **схематических** случаев:

- Препятствие слева, справа, отсутствует, на обоих путях.
- Две привязки одних и тех же candidate IDs к left/right геометрии.
- Два контроля недостаточных данных: raw views чёрные, known path overlays сохранены; ожидаем UNKNOWN, не выдуманную свободную дорогу.

Восемь известных случаев дают 16 candidate-units: 8 пересечений и 8 свободных центральных линий. Метки объявлены по независимой проверке: matching path проходит через центр box footprint при x=8; противоположный находится по другую сторону y=0. **Это centreline/box test, не swept vehicle collision, динамика или дорожная безопасность.**

В model input входят только изображения, motion и сами кандидаты. Ground truth, заголовки и friendly names сцен остаются в отчёте. Оpaque plan/snapshot IDs выводятся из pixels/geometry; ожидаемый ответ в prompt не кодируем. Пары проверяют неизменность raw scene при смене привязки кандидатов и неизменность кандидатов при смене препятствия.

## Одинаковый вход для сравнения

`input_hash` включает actual overlaid RGB, motion, IDs, позы и sampling. Отдельно сохраняем `scene_hash`, `candidate_hash` и **фактический `prompt_sha256`**. Сравнение разных prompt hashes отклоняется, даже если картинка совпала.

Projected prompt v2 не содержит зависящий от модели/request clocks hash. В grammar поле `context_hash` стоит **после** status/IDs/explanation: принудительные request-dependent токены не должны причинно влиять на уже сгенерированное риск-решение. Runner проверяет этот порядок по фактическому raw JSON завершённого ответа; нарушение является технической ошибкой, а не отрицательной модельной оценкой. Worker всё равно связывает ответ с полным контекстом через собственный IPC envelope и parser; модельное повторение hash не считаем аутентификацией.

Все сравниваемые модели получают один prompt profile, calibration/rendering, seed и resource profile. Tokenizer/веса остаются частью модели. На temperature=0 повторения — проверка стабильности, не независимые сцены и не доверительные интервалы качества.

## Метрики и честный отрицательный результат

Матрица обязана содержать ровно model × case × seed; пропуски/повторы — ошибка отчёта. UNKNOWN и failed output не исключаются из знаменателей. Считаем TP/FP/TN/FN, abstained/failed positive/negative, recall на всех рискованных попытках, false-veto rate, coverage и точность полного candidate-specific ответа. Insufficient-evidence controls считаем отдельно.

Контрфактическая пара засчитывается, только если **обе** стороны дали правильные структурированные ответы; изменение произвольного текста не является чувствительностью к сцене. Latency — время всей попытки, включая подготовку, startup при необходимости, preprocessing, generation и parse. p50/p95/p99 описывают малую выборку; это не tail-confidence или runtime guarantee.

Diagnostic gate требует полностью правильного controlled suite. Даже его прохождение всегда возвращает `not_road_qualified`, `advisory_only`, `automatic_veto_allowed=false`. UNKNOWN/неправильный veto/invalid JSON — полноценные отрицательные наблюдения, не повод подменять результат безопасным ответом.

CLI успешно завершает **измерение** после полученных valid/invalid model outputs и полного cleanup; backend/timeout/process failures дают ненулевой exit. Incomplete input construction/неподтверждённый cleanup сохраняют partial records и failure. Если cleanup последней модели сорвался после полной матрицы, отчёт сохраняется, но execution status остаётся failed. Raw truncated output сохраняем bounded, никогда не применяем его как SlowResult. HTML отдельно показывает технический статус, в том числе ошибку cleanup. `commit_reference_supplied` означает переданный commit ID, не автоматическое доказательство чистоты checkout.

## CPU и native Metal

CPU defaults не изменены: 1 thread, `gpu-layers=0`, no mmproj offload. Отдельный Metal profile требует macOS и обнаруженный `MTL*` device; silent CPU fallback запрещён. Runtime запускается с локальными проверенными GGUF, loopback и `--offline`. Native OS network isolation не заявляем; в Docker inference запускаем `--network none`.

Native bootstrap сохраняет venv/CMake/source/build/cache в выбранном каталоге, не устанавливает глобальный Python/Homebrew-пакеты и не меняет Docker VM:

```bash
bash scripts/build_native_slow.sh "$HOME/Desktop/СМЗ" 6
PYTHONPATH=src "$HOME/Desktop/СМЗ/.venv/bin/python" -m scripts.run_slow_qualification \
  --assets-root "$HOME/Desktop/СМЗ/assets" --output-root "$HOME/Desktop/СМЗ/reports" \
  --models 256m 500m qwen3vl-2b --execution-profile metal --threads 6 \
  --server-path "$HOME/Desktop/СМЗ/runtime/build-metal/bin/llama-server" \
  --revision "$(git rev-parse HEAD)"
```

Каждой модели нужен **свой** vision projector. Общий registry используется подготовкой и runner, чтобы не смешивать файлы семейств.

| `--variant` / `--models` | Файлы text + projector, bytes | Transfer cap | Каталог внутри assets-root |
|---|---:|---:|---|
| `256m` | 278824384 | 320 MiB | `smolvlm-256m` |
| `500m` | 545590272 | 600 MiB | `smolvlm-500m` |
| `qwen3vl-2b` | 1552463168 | 1536 MiB | `qwen3vl-2b` |

Официальный [Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/tree/52d6c8ffea26cc873ac5ad116f8631268d7eb503) использует text Q4_K_M и projector Q8_0; Smol-модели — Q8_0. Это сравнение доступных комплектов, не изолированный эксперимент размера модели. Все source revisions/SHA256/размеры/ETag закреплены. Подготовка single-owner: один процесс на каталог; inference только проверяет assets и ничего не скачивает/не ремонтирует.

```bash
PYTHONPATH=src "$HOME/Desktop/СМЗ/.venv/bin/python" -m scripts.prepare_slow_sample \
  --variant qwen3vl-2b --root "$HOME/Desktop/СМЗ/assets/qwen3vl-2b" --rate-limit-mib 16
```

Darwin teardown учитывает EPERM гонки/ограничения: сигнал leader — best effort, освобождение только после смерти leader и bounded `ps`-подтверждения отсутствия живой группы. Ошибка проверки остаётся fail-closed, cleanup имеет общий бюджет. Native RSS — максимум reaped child process, **не** aggregate/unified GPU memory. Для device memory отдельного измерения пока нет.

CI запускает полный dataset-free suite и настоящие обе Smol-модели на явно названном smoke subset q00/q01/q08. Это не тихий skip полного набора; полный controlled run и реальные hardware результаты фиксируем в MV-146. Qwen — отдельный native Metal experiment; CI не заявляет его weighted CPU-проверку в лимите 2 GiB.

## Двухшаговый projected profile

`--profile grounded-v1` отделяет оценку наблюдаемости сцены от классификации перекрытия пути. `--suite validation` выбирает фиксированные новые геометрии, сохраняя старый controlled default. [Контракт, trace, границы и native profile](slow-grounding.md).
