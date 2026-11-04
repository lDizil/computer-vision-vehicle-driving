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
└── safety/
    └── kinematics.py           # дискретные проверки скорости/ускорения/jerk

scripts/
├── prepare_navsim_sample.py    # проверяемые реальные logs/maps
└── run_navsim_sample.py        # caching → scoring → CSV/JSON/PNG

tests/
├── unit/                      # наша логика и независимые fixtures
└── smoke/                     # настоящие установленные upstream

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
| `safety/fallback.py` | Воспроизводимая тормозная траектория и failure semantics | MV-132 |
| `safety/arbiter.py` | Принадлежность/свежесть ответа, veto и выбор кандидата | MV-121 |
| `agent.py`, `adapters/navsim_agent.py` | Сборка зависимостей и реализация NAVSIM Agent API без доменной логики в адаптере | MV-122 |
| `runtime/replay.py` | Сценарии timeout, stale/out-of-order и all-veto | После offline integration |

Направление зависимостей: модели и адаптеры → `contracts`; Safety → `contracts`; сборка агента → модели/адаптеры/Safety. Контракты не импортируют PyTorch, NAVSIM, конкретные модели или runtime. Не создаём отдельный DI-framework: зависимости передаём явно при сборке агента.

Дискретные кинематические проверки реализованы в `safety/kinematics.py`; [формулы и семантика результата](kinematics.md). Fallback и выбор кандидата — отдельные потребители этой проверки.

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
