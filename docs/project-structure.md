# Структура проекта и движение данных

Проект связывает готовый визуальный planner, анализ риска и выбор траектории. Мы не пишем собственные PyTorch, NAVSIM или базовую VLM. Собственная часть — согласование данных, применимость slow-ответа, Safety Veto и воспроизводимые сравнения.

## Что переиспользуем

| Готовый компонент | Его ответственность | Наша граница |
|---|---|---|
| NAVSIM `AgentInput`, `Trajectory`, `TrajectorySampling` | Вход/выход benchmark | `adapters/navsim.py`: перевод в независимые контракты и обратно |
| NAVSIM `ConstantVelocityAgent` | Простой воспроизводимый baseline | Запускаем upstream, не переписываем его как «нашу модель» |
| NAVSIM/nuPlan maps, metric caching, PDMS evaluator | Данные, карты и оценка результата | `scripts/`: подготовка небольшого sample, запуск и сбор артефактов |
| nuPlan SE(2)/trajectory utilities | Преобразования и интерполяция объектов benchmark | Используем в benchmark-части; компактный NumPy-контракт не импортирует весь benchmark/ML-стек |
| FSDrive/PyTorch и выбранная slow VLM | Нейросетевые вычисления | Будущие тонкие model adapters, а не копия исходников моделей |

`contracts/Trajectory` не заменяет весь NAVSIM `Trajectory`. Это неизменяемое значение, общее для Fast, Slow и Safety. В нём фиксируем владение данными, координаты и время; на внешней границе используем настоящий NAVSIM-объект. Веса и обучение не являются частью этих контрактов.

Источники: [NAVSIM types](https://github.com/autonomousvision/navsim/blob/3e8291bfa89ff247231e0227778840cd0a036896/navsim/common/dataclasses.py), [nuPlan SE(2)](https://github.com/motional/nuplan-devkit/blob/ce3c323af01c0d7ec5672f7832ef53f9c679aab0/nuplan/common/geometry/convert.py), [FSDrive](https://github.com/MIV-XJTU/FSDrive), [Reason2Drive](https://github.com/LogosIntelligenceLab/Reason2Drive).

## Дерево исходников

```text
src/hybrid_drive/
├── __init__.py                 # публичные Pose2D / Trajectory
├── contracts/
│   ├── trajectory.py           # ego-позы, время, SE(2), явное resample
│   ├── motion.py               # измеренные скорость/ускорение в ego-frame
│   ├── planning.py             # Candidate, CandidateSet, content hash
│   ├── slow.py                 # SlowRequest, SlowResult, RiskStatus
│   └── _validation.py          # общие проверки scalar/ID/sequence
├── adapters/
│   └── navsim.py               # настоящий NAVSIM ↔ наши траектории
├── agents/
│   └── hybrid.py               # Fast NAVSIM agent → SafetyArbiter → native output
├── fast/
│   └── latent_transfuser.py    # официальный pretrained LTF, strict/offline CPU
├── assets/                    # bounded download, atomic files, JPEG allowlist/verification
├── experiments/
│   ├── protocol.py            # manifests, парность, ADE/FDE, failures в знаменателе
│   └── splits.py              # аудит official splits и recording-level validation
└── safety/
    ├── kinematics.py           # дискретные проверки скорости/ускорения/jerk
    ├── fallback.py             # аналитический тормозной профиль и failure
    ├── selection.py            # выбор кандидата либо fallback
    └── arbiter.py              # применимость slow-ответа и объединение решения

scripts/
├── prepare_navsim_sample.py    # проверяемые реальные logs/maps
├── prepare_ltf_sample.py       # pinned checkpoint/encoder + три настоящие камеры
├── sample_support.py           # общий verified-data setup
├── run_navsim_sample.py        # caching → CV/checks → CSV/JSON/PNG
└── run_neural_sample.py        # один neural Fast → парный baseline/checks + CV reference

tests/
├── unit/                      # наша логика и независимые fixtures
├── smoke/                     # настоящие установленные upstream, без весов
└── integration/               # обязательные реальные weights, явный отдельный запуск

requirements/                  # закреплённый CPU dependency graph
Dockerfile                     # одна Linux-среда для arm64/amd64
.github/workflows/             # те же тесты + реальный sample
```

Публичный импорт остаётся `from hybrid_drive import Pose2D, Trajectory`. Типы планов и slow-протокола импортируем из `hybrid_drive.contracts.planning` и `hybrid_drive.contracts.slow`.

## Куда добавляем следующие блоки

Это карта будущих файлов, а не существующие пустые модели. Каталог появляется вместе с рабочей реализацией и тестами.

| Файл / модуль | Ответственность | Задача |
|---|---|---|
| `contracts/sensors.py` | CameraSnapshot: время, rig, калибровка, маски; не обучаем perception | MV-131 |
| `adapters/sensors.py` | Явный camera mapping/preprocessing конкретного датасета | MV-131 |
| `fast/fsdrive.py` | Загрузка проверенного checkpoint, upstream inference → CandidateSet | MV-116, MV-117, MV-118 |
| `bridge/context.py` | Контекст выбранной slow-модели; shared-feature bridge только после alignment | MV-120 |
| `slow/worker.py` | Модель, structured output, ограниченная очередь и ошибки inference | MV-130, MV-120 |
| `agents/hybrid.py` | Расширение сквозного агента до целевых Fast/Slow backend | MV-122 |
| `runtime/replay.py` | Сценарии timeout, stale/out-of-order и all-veto | После offline integration |

Направление зависимостей: модели и адаптеры → `contracts`; Safety → `contracts`; сборка агента → модели/адаптеры/Safety. Контракты не импортируют PyTorch, NAVSIM, конкретные модели или runtime. Не создаём отдельный DI-framework: зависимости передаём явно при сборке агента.

Дискретные кинематические проверки реализованы в `safety/kinematics.py`; [формулы и семантика результата](kinematics.md). [Fallback, offline-выбор и SafetyArbiter](fallback.md) объединяют проверки с применимым slow-ответом. Очередь/вызов моделей и подключение evaluator остаются ответственностью runtime и сборки агента.

## Сквозной NAVSIM-агент

`HybridDriveAgent` принимает готовый Fast NAVSIM agent, `KinematicLimits` и `BrakingConfig`. Последний ego-state передаём в Safety без обнуления боковой скорости или ускорения. Fast возвращает один кандидат; его проверка, выбор и торможение принадлежат существующему `SafetyArbiter`, а не копируются в агент.

Каждый вызов — независимый offline-сценарий с локальным началом координат и `t_ref_s=0`. NAVSIM `AgentInput` не содержит глобального timestamp/token: их не выдумываем и не выдаём эти вызовы за непрерывную симуляцию. Новые ID связывают снимок и план внутри вызова; фактический token добавляет evaluator. Backend с `requires_scene=True` не подключаем: это потребовало бы привилегированные данные сцены.

`compute_trajectory()` возвращает настоящий NAVSIM `Trajectory`. `last_decision` хранит только последнее решение и очищается **перед** следующим вызовом, включая ошибочный. Вызываем агент последовательно. Ошибки входа/Fast/формата не заменяем выдуманной траекторией; явный `decision.failed` сохраняем отдельно от валидности массива. Обёртка предназначена для inference, не для обучения или сохранения Fast через её `state_dict`.

Fast выбираем явно: штатный `ConstantVelocityAgent` либо официальный pretrained `LatentTransfuserAgent`; Slow фиксируется как `UNKNOWN/not_requested`. FSDrive остаётся целевой интеграцией, LTF не подменяет его названием. [Нейросетевой вход, парные сравнения и протокол](experiments.md). Проверка препятствий и семантический veto без подключённого Slow этим прогоном не выполняются.

## Один план: что куда движется

```text
CameraSnapshot + ego-state
          │
          ▼
FastAdapter ───► CandidateSet ───────────────────────► SafetyArbiter
                     │                                    ▲
                     ▼                                    │
         SlowRequest + связанный scene context             │
                     │                                    │
                SlowWorker ───► SlowResult ────────────────┘
                                                          │
                                             выбранная Trajectory
                                                          │
                                             NAVSIM adapter/evaluator
```

Например, Fast предлагает кандидат `turn-right` с будущими `(x,y,heading)`. Мы сохраняем `plan_id`, `snapshot_id`, simulation `t_ref_s`, reference pose, временную сетку и сами точки в неизменяемом `CandidateSet`.

Slow-запрос содержит этот набор целиком, а не только ID/hash. `candidates_hash` — SHA256 канонического JSON с версией схемы, контекстом и упорядоченными кандидатами. Изменение геометрии, времени, системы отсчёта или состава набора даёт другую привязку ответа. Hash не является подписью или средством авторизации.

`SlowResult` явно содержит `UNKNOWN`, `NO_VETO` либо `VETO` с конкретными `candidate_id`. Пояснение — только текст: слово «опасно» в нём не подменяет структурированный статус. `validate_for(request)` проверяет соответствие ID/hash/model version и известность veto-ID, но **не** принимает решение и **не** проверяет свежесть.

Процессные сроки `submitted_at_monotonic_s` / `deadline_monotonic_s` сравниваем только в одной monotonic clock domain. Simulation `t_ref_s` / `valid_until_sim_s` — другая шкала. Monotonic deadline не передаём другой машине как универсальный timestamp. Проверка срока по доверенному времени получения и защита от replay принадлежат runtime/арбитру, а не ответу VLM.

Текущий `SlowRequest` — траекторная часть протокола. Привязку фактических изображений/признаков оформляем вместе с CameraSnapshot/ContextAdapter; пустой контекст не считаем полноценным vision-запросом. Карта выше показывает целевое соединение; runtime Fast–Slow этим набором типов не запускается.

После выбора NAVSIM оценивает траекторию. Это offline benchmark, не команды реальной машине и не реализация `Vehicle Controller` из исходной схемы. Для будущего CARLA будет отдельный контроллерный адаптер.

## Окружение и запуск

CPU Docker уже содержит настоящие NAVSIM/nuPlan, PyTorch, NumPy и устанавливаемый `hybrid_drive`. Не создаём второй образ только ради файловой структуры. [Команды и закреплённые версии](environment.md) едины для Mac/Linux и CI. Локальные тесты запускаем с 1 CPU / 2 GiB и без сети.

FSDrive со своим CUDA/PyTorch-стеком позднее изолируем от benchmark-образа. Не устанавливаем несовместимые ML-зависимости в существующую рабочую CPU-среду и не скачиваем веса при обычном запуске тестов.

## Image-input Slow baseline

`slow/` владеет immutable camera/plan context, строгим JSON protocol, bounded subprocess worker и planning session. `adapters/slow_context.py` читает только текущий NAVSIM AgentInput. Локальный llama.cpp backend и offline/replay runners отделены от Fast и Safety; shared-feature Bridge и Reason2Drive не подменяются этим baseline. [Границы и поток данных](slow-path.md).

## Геометрия и визуальная квалификация

`geometry/` владеет калибровкой и проекцией ground trajectories в изображение. `qualification/` отделяет контролируемые сцены и их labels от model inputs, считает matched-input метрики и формирует самостоятельный HTML-отчёт. Runner поддерживает одинаковые CPU или native Metal profiles для сравниваемых моделей; semantic veto автоматически не включает. [Контракт, ограничения и запуск](qualification.md).
