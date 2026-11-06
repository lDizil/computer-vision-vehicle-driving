# CPU/Linux окружение NAVSIM

Первый проверяемый этап MV-135 / MV-105: воспроизводимая CPU-среда и dataset-free smoke. Это не FSDrive inference, не mini PDMS и не проверка NVIDIA GPU.

## Запуск

Нужен работающий Docker с Linux containers. Из корня репозитория:

```bash
docker build --tag hybrid-drive:cpu .
docker run --rm --cpus 1 --memory 2g --memory-swap 2g --network none hybrid-drive:cpu
```

Сборка загружает исходники upstream и полный набор Python-зависимостей, но не модели и датасеты. Тесты выполняются без сети и не должны пропускаться при отсутствии нужной зависимости. В GitHub CI используется тот же Dockerfile и та же команда.

## Что закреплено

| Компонент | Версия / источник |
|---|---|
| Base image | Python 3.10 slim-bookworm, immutable multi-architecture digest в Dockerfile |
| NAVSIM | `3e8291bfa89ff247231e0227778840cd0a036896`, maintained branch v1.1, не SHA одноимённого release tag |
| nuplan-devkit | `ce3c323af01c0d7ec5672f7832ef53f9c679aab0`, tag nuplan-devkit-v1.2 |
| Torch / torchvision | 2.0.1 / 0.15.2, официальный CPU index |
| Python dependencies | Отдельные полные version locks для Linux arm64 и amd64 |

NAVSIM и nuplan установлены editable в `/opt/navsim` и `/opt/nuplan-devkit`; upstream metadata не изменяем. Устанавливаем полный dependency graph NAVSIM, а не скрытый урезанный subset. Дополнительный `requirements.txt` nuplan не устанавливаем: он конфликтует со стеком NAVSIM. `pip check` обязателен во время сборки.

Python 3.10 и часть библиотек — legacy-зависимости этого NAVSIM. Python 3.10 уже EOL; среда предназначена для изолированного воспроизведения benchmark, не для открытого production-сервиса. Перенос на поддерживаемый Python требует отдельной проверки совместимости Hydra/upstream. FSDrive с torch 2.5.1/CUDA 12.4 сюда не устанавливаем.

## Что проверяет smoke

- Реальные импорты обоих editable upstream.
- Импорт `GPKGMapsDB`, POSIX-зависимости и запись/чтение маленького временного GeoPackage через Fiona. Конструктор `GPKGMapsDB` не вызываем: ему нужны реальные карты. Чтение настоящей nuPlan-карты этим тестом не подтверждается.
- CPU tensor operation PyTorch.
- Реальный `ConstantVelocityAgent`: стоящий автомобиль, скорость 5 м/с и использование последнего ego-state. Ожидаемые координаты заданы независимо от реализации агента.

## Один реальный сценарий

```bash
python3 scripts/prepare_navsim_sample.py --root "$PWD/data/navsim" --rate-limit-mib 4
mkdir -p outputs
docker run --rm --cpus 1 --memory 2g --memory-swap 2g --network none \
  --user "$(id -u):$(id -g)" \
  --volume "$PWD/data/navsim:/data" \
  --volume "$PWD/outputs:/results" \
  hybrid-drive:cpu python -m scripts.run_navsim_sample --data-root /data --output-root /results
```

Helper загружает один log и четыре векторные карты: до 800 MiB по сети, около 1.2 GiB после распаковки. Все четыре города нужны из-за eager initialization upstream nuPlan. Камеры, LiDAR и отдельные raster-архивы не используются. Публикация/распространение данных определяется условиями источника, указанными в `scripts/navsim-sample-sources.json`; в Git сами данные не добавляем.

Проверяются точные HTTP Range/ETag, размеры, SHA256 log и CRC32 ZIP members. Полученные SHA256 карт сохраняются в manifest; полный архивный SHA256 при частичной загрузке не объявляется проверенным. Повреждённые или незавершённые existing-файлы требуют явного удаления перед повтором.

Runner проверяет входы, выбирает одну допустимую сцену, запускает штатные metric caching и scoring с `worker=sequential gpu=false`. Успех требует ровно одного requested token, `valid=True`, конечного score в `[0,1]` и корректной average-строки, а не только exit code процесса. Результат: исходный CSV, `result.json`, PNG с predicted/reference траекториями и два лога. Каждый запуск создаёт отдельный каталог.

Для сквозного `HybridDriveAgent` добавляем `--agent hybrid` к команде runner. Он передаёт результат единственного вызова агента напрямую штатному `pdm_score` и рисует именно оценённые точки. Дополнительный `decision.json` содержит причины выбора, кинематические проверки, Slow-статус и failure. Неудачный fallback сохраняется как неуспешная сцена, а не успешный неподвижный baseline. Лимиты этого sample — экспериментальная конфигурация проверки соединения модулей, не откалиброванная политика безопасного движения.

В CI выполняется тот же реальный сценарий. Для него выделено 1 CPU / 4 GiB на GitHub runner, а не на Mac; smoke ограничен 1 CPU / 2 GiB. Артефакты содержат только результаты/логи/manifest, без самих maps/log pickle. Один сценарий не заменяет оценку всего mini.

Pickle исходных данных десериализуется только внутри контейнера. Монтируем только каталог этого датасета и каталог результатов; рабочие репозитории и credentials контейнеру не передаём.

## Обновление зависимостей

`requirements/navsim.in` соответствует полному requirements NAVSIM по закреплённому SHA, кроме nuplan: его checkout установлен отдельно. Добавлены только инструменты сборки `wheel` и `pip`. При обновлении upstream сверяем исходный список и metadata; не меняем только SHA.

Locks сгенерированы uv 0.13.6. После согласованного изменения входного списка:

```bash
uv pip compile requirements/navsim.in --python-version 3.10 --python-platform aarch64-unknown-linux-gnu --torch-backend cpu --no-annotate --no-header --emit-index-url -o requirements/linux-arm64.lock
uv pip compile requirements/navsim.in --python-version 3.10 --python-platform x86_64-unknown-linux-gnu --torch-backend cpu --no-annotate --no-header --emit-index-url -o requirements/linux-amd64.lock
```

После обновления обязательны новая сборка, `pip check` и smoke на соответствующих архитектурах. Python-пакеты закреплены по версиям; APT mirror и байтовое содержимое всех package-артефактов не зафиксированы, поэтому побайтовую воспроизводимость образа не заявляем.

Build dependencies native geospatial-пакетов закреплены отдельно в `requirements/build-constraints.txt`. Они применяются через pip 25.3 `--build-constraint`, не меняя setuptools, требуемый upstream в runtime.
