# Projected visual grounding Slow

Этот вариант Slow решает ограниченную задачу: по текущим калиброванным изображениям с наложенными кандидатами определяет наблюдаемость сцены и перекрытие центральной линии пути видимым box. Он не является воспроизведением Reason2Drive или learned Prior Tokenizer Bridge.

## Ответственность и поток

```text
CandidateSet + known camera calibration
    → actual projected overlays: slot 0 cyan, slot 1 magenta
    → Qwen: visible / missing / uncertain
    → if visible, Qwen: cyan / magenta / both / none / unknown
    → strict projected protocol → canonical SlowResult
    → existing Safety applicability checks (only when explicitly wired)
```

`slow/grounding.py` владеет двумя вопросами, schemas, строгим разбором и единственным отображением категорий в `SlowResult`. `slow/llama.py` владеет только local model transport. Worker передаёт неизменённый bounded trace и вызывает явно переданный parser; не интерпретирует категории. Safety не знает о цветах или внутренних полях visual protocol.

На вход LLM идут сами проекции фактических путей. Повторный длинный список числовых координат в этом profile не передаём: он не заменяет наблюдения препятствий. Полный план, время и pixels остаются в `SlowContext` и связаны hash. Slot/color contract обязателен; произвольный кадр с цветными объектами не считается корректным projected input.

## Протокол и отказы

Каждый этап возвращает только категорию и точный `context_hash`, причём hash идёт последним. Первый ответ `missing/uncertain` даёт `UNKNOWN`, второй запрос не выполняется. Для видимой сцены `cyan/magenta/both` отображаются в фактические candidate IDs соответствующих слотов; `none` даёт `NO_VETO`, `unknown` — `UNKNOWN`. Текст объяснения не парсим: итоговое explanation — короткое детерминированное отображение категорий.

Worker raw response для этого profile — JSON trace с `protocol`, `visibility_raw`, `overlap_raw`. Последние два поля содержат точные исходные модельные строки; это не канонический ответ, выданный за ответ модели. Повторные/лишние ключи, неверный hash/order/enum и противоречивая ветка отклоняются без ремонта. На всю trace действует прежний лимит 8192 UTF-8 bytes.

Невалидная генерация остаётся `invalid_output`; provider/HTTP/transport failure — `backend_error`, с сохранением уже полученной bounded trace. Ошибки не превращаем в успешный `UNKNOWN`. Общий worker deadline включает startup и оба запроса; при timeout прекращается вся группа процессов. При таком hard timeout trace незавершённого backend-вызова может быть потеряна вместе с child: parent сохраняет timeout и сам контекст, но не обещает промежуточный ответ первого этапа. Catchable HTTP/provider failures сохраняют уже полученные этапы. Новый protocol имеет отдельную model version, чтобы его ответы не смешивались с direct baseline.

## Воспроизводимость и проверка

- Direct profile и стандартный HybridDriveAgent не меняют поведение автоматически.
- Qwen grounding на Mac использует явный visual budget: min=max 1024, context 8192, cache off; CPU defaults прежние: max 64/context 4096.
- `prompt_sha256` для одного вопроса сохраняет прежний смысл; для двух этапов это SHA256 canonical JSON списка вопросов. Это hash объявленной условной программы, а не утверждение, что оба вопроса всегда выполнены. Реально выполненные этапы видны в trace.
- Старые q00–q09 используем только как development. `synthetic-validation-v1` содержит 30 заранее фиксированных новых геометрических случаев; metadata/labels не входят в model input. Renderer и rig остаются synthetic общими, поэтому это не real-road holdout.
- До первого validation запуска фиксируем profile, commit, suite manifest hash и критерии. После просмотра результатов profile под этот набор не меняем; дальнейшая настройка потребует новой проверки.
- Метрики считаем для всей pipeline, включая UNKNOWN и отказы. Дополнительно доступны точные категории обоих модельных этапов. Даже diagnostic PASS не разрешает automatic veto или утверждение о дорожной безопасности.

```bash
PYTHONPATH=src "$HOME/Desktop/СМЗ/.venv/bin/python" -m scripts.run_slow_qualification \
  --assets-root "$HOME/Desktop/СМЗ/assets" --output-root "$HOME/Desktop/СМЗ/reports/grounding" \
  --models qwen3vl-2b --profile grounded-v1 --suite validation --seeds 0 1 2 \
  --execution-profile metal --threads 6 --max-tokens 256 --deadline-seconds 120 \
  --image-min-tokens 1024 --image-max-tokens 1024 --context-size 8192 --no-prompt-cache \
  --server-path "$HOME/Desktop/СМЗ/runtime/build-metal/bin/llama-server" \
  --revision "$(git rev-parse HEAD)"
```

CPU CI отдельно исполняет настоящий двухшаговый smoke на уже закреплённом SmolVLM-256M; его semantic результат не выдаём за качество Qwen. Qwen с указанным visual budget проверяем native на Mac.
