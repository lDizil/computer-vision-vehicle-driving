# Image-input Slow baseline

Экспериментальная Slow-ветка получает текущие камеры, измеренное движение и полный набор Fast-траекторий. Локальный SmolVLM-256M Q8 возвращает candidate-specific `UNKNOWN / NO_VETO / VETO`. Это доступный image-input baseline с отдельным vision encoder, **не** обученный Reason2Drive и **не** shared-feature Prior Tokenizer Bridge.

## Поток данных

```text
AgentInput + один immutable Fast plan
    → context_from_navsim (три RGB-view, явный resize)
    → SlowContext + content hash + полный prompt
    → AsyncSlowWorker (отдельный процесс → локальный llama-server)
    → строгий JSON parser → SlowResult
    → SafetyArbiter → candidate / braking fallback / explicit failure
```

`HybridDriveAgent` по умолчанию сохраняет прежний Fast → checks путь. `SlowPlanningSession` — отдельный runtime с явной активацией плана. Новая сцена/новый план требуют нового запроса; старый ответ не переносится и не перебазируется. Первый вызов `decide` окончателен для активного плана: поздний ответ остаётся только аудитом. Session не ждёт модель.

Для offline-пилота `run_slow_sample` явно ждёт **вне** fast decision path, удерживая записанную сцену и `now_sim_s` неизменными. Такой эксперимент проверяет соединение моделей и evaluator, но не возможность своевременно применять Slow при движении машины. Temporal replay проверяет временной контракт отдельно.

## Контекст и ответ

- Передаём текущие `CAM_L0`, `CAM_F0`, `CAM_R0`; resize каждой отдельно — RGB 384×216, PIL bilinear. Это не panorama и не калиброванная проекция траектории на изображение.
- Immutable pixels, motion, все точки/ID/временная сетка кандидатов, request/model identity входят в hash. Ограничения: три view ≤640×480, ≤8 кандидатов ×32 точки, metadata/prompt ≤32 KiB.
- Будущие человеческие траектории, map/box-аннотации и evaluator labels не входят в Slow input.
- Ответ содержит ровно `context_hash`, `status`, `vetoed_candidate_ids`, `explanation`; максимум 8192 UTF-8 bytes, объяснение ≤512 символов. `VETO` называет известные уникальные ID; другие статусы требуют пустого списка.
- Parser отклоняет fences/prose, повторы/лишние ключи, чужой hash, неверные ID и nonfinite JSON. Runtime сам добавляет trusted IDs и времена. Не исправляем повреждённый ответ по смыслу текста.
- JSON grammar ограничивает форму, а не достоверность. `NO_VETO` означает только отсутствие veto модели, не доказательство безопасности; без достаточных данных ожидаем `UNKNOWN`.

## Асинхронный runtime

Одновременно держим не более одного active и одного pending запроса. Новый pending вытесняет прежний. Большие данные проходят через фиксированные shared buffers (input 4 MiB / output 8192 bytes), уведомления — короткие Pipe frames; бесконечных очередей и feeder threads нет.

Один владелец вызывает `submit/poll` на fast ticks; именно эти вызовы обслуживают deadline. Самостоятельного watchdog нет. Model startup входит в deadline. Timeout завершает всю POSIX process group, включая llama-server; новый worker запускаем только после завершения старой группы. `close` ограничен 0.5 с и сообщает ошибку, если завершение не подтверждено. Подтверждение отсутствия живых потомков рассчитано на Linux/Docker. В контейнерных запусках используем `--init`, чтобы reaping orphan children не зависел от Python как PID 1.

Диагностический буфер хранит максимум два completion. Для полного журнала нужно регулярно `poll`; потери отражаются в `dropped_completion_count`, totals — в `outcome_counts`. Ошибки backend не пересылают произвольный текст исключений. Invalid/timeout/неприменимый ответ остаётся `UNKNOWN`, кинематика и fallback не отключаются.

## Модель и изоляция

`scripts/smolvlm-sources.json` закрепляет оба GGUF по commit/size/SHA256/ETag: text model + vision projector, всего 278824384 bytes. Tokenizer/template встроены в GGUF. Подготовка использует существующий bounded downloader; inference только проверяет readonly assets и не скачивает/не ремонтирует их.

`Dockerfile.slow` собирает llama.cpp по закреплённому commit на той же Debian-базе, затем добавляет бинарник в отдельный NAVSIM-образ. PyTorch и базовые locks не меняются. `SOURCE_REVISION` проверяется до запуска. Runtime работает на loopback внутри `--network none`, не публикует порты, отключает proxy и получает base64 PNG, не внешние URL. CPU threads=1, context=4096, image-max-tokens=64, generation ≤256 tokens. Ограничение vision compute задаём отдельно: уменьшение JPEG само по себе не исключает upstream upsampling/tiling.

Подготовка и smoke:

```bash
docker build -t hybrid-drive:cpu .
docker build -f Dockerfile.slow -t hybrid-drive:slow .
docker run --rm -v "$SLOW_ASSETS:/slow-assets" hybrid-drive:slow \
  python -m scripts.prepare_slow_sample --root /slow-assets
docker run --rm --init --cpus 1 --memory 2g --memory-swap 2g --network none \
  hybrid-drive:slow python -m scripts.run_slow_replay --output /tmp/replay.json
```

Полный реальный pilot и сохранение артефактов воспроизводит `.github/workflows/slow-smoke.yml`. Отдельные данные/volumes удаляются после проекта; глобальные зависимости и Docker VM не настраиваем. Результаты/измерения ведём в MV-130, MV-142–MV-145, а не в README.

## Источники

- [SmolVLM GGUF, Apache-2.0](https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/tree/b9e4379657e1450d04d02eec8e345667265b0a00).
- [llama.cpp: поддержка multimodal](https://github.com/ggml-org/llama.cpp/blob/372dc5ca157276beb6c67b2e92477e8d662e964a/docs/multimodal.md).
- [Закреплённый server API](https://github.com/ggml-org/llama.cpp/blob/372dc5ca157276beb6c67b2e92477e8d662e964a/tools/server/README.md).
