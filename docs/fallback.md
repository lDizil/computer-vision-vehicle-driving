# Тормозной fallback и offline-выбор

`select_trajectory(plan, motion, limits, config, vetoed_candidate_ids)` проверяет все кандидаты и выбирает первый неотклонённый вариант, прошедший [кинематические проверки](kinematics.md). Порядок `CandidateSet` считается порядком предпочтения Fast. Сохраняем отчёт каждого кандидата, включая vetoed и невыбранные.

If no candidate is eligible, the selected fallback policy handles every cause consistently. The default remains longitudinal braking; the curved snapshot model requires explicit opt-in. Within one policy, all ablations retain the same model and limits.

## Границы API

- Используем явный `BrakingConfig(deceleration_mps2, jerk_mps3)`. Значения положительные и не превышают соответствующие `KinematicLimits`.
- Шаг плана должен совпадать с конфигурацией. Неизвестные/повторные veto-ID и неверная конфигурация — `ValueError`, а не молчаливое игнорирование.
- `selection.py` получает **уже авторизованные для этого плана** veto-ID. Он не принимает свободный текст или `SlowResult`: применимость проверяет `SafetyArbiter`, очередь и жизненный цикл запросов — runtime.
- Движение, план и система отсчёта должны относиться к одному `t_ref`; эта функция не синхронизирует сенсоры.

## Поддерживаемый профиль

Начальный вариант — прямолинейное движение вперёд: `vx>=0`, `vy=ay=0`, `|ax|<=A`, где `A` — заданное максимальное тормозное ускорение, `J` — максимальный jerk. Боковые компоненты и движение назад не отбрасываем и не проецируем на x: возвращаем явную ошибку fallback.

Для начальных `v,a`:

```text
b = min(A, sqrt(J*v + a²/2))
t_down = (a+b)/J
t_hold = (v + a²/(2J) - b²/J)/b   для трапецеидального профиля
t_hold = 0                       для треугольного профиля
t_up = b/J
```

Фазы имеют jerk `-J, 0, +J`. Интегрируем их аналитически, сохраняем начальные скорость/ускорение; после завершения `v=a=0`, позиция постоянна. При положительном начальном ускорении машина сначала может ускоряться: отдельно проверяем аналитический пик `v + a²/(2J)`, даже если грубая сетка его пропускает.

При `a<0` и `v<a²/(2J)` невозможно одновременно сохранить начальное ускорение, ограничить jerk, избежать reverse и завершить плавную остановку. Возвращаем `INFEASIBLE_STOP`, а не делаем `max(v,0)`. Равенство допустимо: остаётся только фаза снятия торможения `+J`. Настоящий покой — только точно нулевые `v,a`, не epsilon-обнуление малой скорости.

Вычисления используют устойчивую ветку около этой границы и оценку последней фазы от конечной остановки. Непредставимые времена, расстояния или точки дают `NUMERIC_RANGE`. Остановка может находиться за горизонтом плана: траекторию не растягиваем и не обрываем мгновенно. `stop_time_s` и `stop_distance_m` описывают полный профиль.

## Результат и failure accounting

`TrajectoryDecision` содержит:

| `status` | Что возвращаем |
|---|---|
| `CANDIDATE` | Выбранную исходную траекторию и её ID |
| `FALLBACK` | Тормозной профиль, прошедший sampled-проверку и проверку аналитического пика скорости |
| `FAILURE` | Траекторию для offline-протокола и обязательный `failed=True` с причиной в `fallback.failure_reason` |

Если тормозной профиль построен, но нарушает лимиты, сохраняем именно его с `KINEMATIC_LIMITS`, не выдавая за допустимый. Если поддерживаемый профиль построить невозможно, возвращаем `EMERGENCY_HOLD`: конечный массив нулевых поз нужного размера, **только как offline-placeholder с failure**. Это не команда остановки движущейся машины и не безопасная политика реального контроллера.

Траектория присутствует и при failure, имеет тот же горизонт/число точек. `plan` в решении сохраняет исходный неизменяемый контекст, ID/hash и reference pose. Штатный NAVSIM-адаптер отдельно проверяет диапазон `float32` при преобразовании. В результатах сохраняем `status`, `failed`, причины и отчёты; такие сцены нельзя удалять из сравнения или считать fallback-успехом. Smoke проверяет, что все три исхода сохраняют настоящий NAVSIM `Trajectory` и что failure остаётся посчитанным.

## SafetyArbiter

`SafetyArbiter(limits, braking).decide(...)` объединяет текущий план, доверенные часы, активный `SlowRequest` и опциональный `SlowResult` с выбором выше. Он не ждёт модель и не запускает inference. Зависимости передаём явно, mutable-кеша ответов нет.

Перед применением slow проверяем:

1. Активный запрос относится к текущему `plan_id`, `snapshot_id` и полному hash кандидатов. Пересозданный эквивалентный план допустим; изменённая геометрия при тех же ID — нет.
2. Текущий monotonic time не раньше отправки и не позже `deadline_monotonic_s`; simulation time не позже `valid_until_sim_s`. Границы включительные, float-допуск сроки не продлевает. Позднее принятие даже своевременно полученного ответа отклоняем.
3. Доверенное время получения присутствует и лежит между отправкой и текущим временем. Часы monotonic одного домена; simulation time с ними численно не сравниваем.
4. Ответ соответствует request/snapshot/plan/hash/model version и содержит только известные veto-ID.

Нет запроса/ответа, неверный формат, чужой план, истёкший срок или статус `UNKNOWN` дают эффективный `UNKNOWN` с `SlowDisposition`, а не `NO_VETO`. Кинематические проверки и fallback при этом работают. Только корректный актуальный `VETO` передаёт маску в selection; текст пояснения не управляет решением. Актуальный `NO_VETO` тоже сохраняем как применённый ответ, не объявляя траекторию безопасной.

В `ArbitrationDecision` сохраняем исходный `TrajectoryDecision`, оценку slow, применённый ответ, возраст ответа по monotonic clock и обе шкалы времени решения. Неверные типы доверенных входов/часов и simulation time раньше текущего плана — `ValueError`; некорректный внешний ответ — деградация до `UNKNOWN`.

Арбитр stateless: повтор того же допустимого ответа к тому же неизменяемому плану идемпотентен. Он не переносит veto на новый план и не заменяет транспортную защиту от replay, ограниченную очередь и дедлайны выполнения модели. Выпуск уникальных request ID и передача текущего плана/запроса принадлежат runtime.

`now_sim_s` используем для применимости slow-ответа, не для сдвига точек траектории: caller передаёт актуальный Fast-план и согласованное движение на его `t_ref`. В offline NAVSIM это время текущего кадра; rebasing и управление с непрерывно идущими часами здесь не реализованы.

Политика проверяет только заявленные ограничения. Дорога, препятствия, yaw/кривизна, real-time и управление CARLA — отдельные слои. `FALLBACK` не означает доказанную дорожную безопасность. Подключение моделей/evaluator-агента относится к MV-122.

## Opt-in curved snapshot model

`SafetyArbiter(limits, braking, FallbackModel.CURVED)` accepts a fresh
`fallback_observation` on each call. `FallbackObservation` binds the associated
heading rate to `snapshot_id`, `t_ref_s`, the complete `EgoMotion` and a source
ID. A different snapshot/time/motion gives `OBSERVATION_MISMATCH`; a missing
rate gives `MISSING_HEADING_RATE`. An eligible Fast candidate does not need
unused fallback inputs.

The [planar hypothesis](planar-motion-model.md) preserves both initial velocity
and acceleration components, with independent path and body-heading curvature.
The scalar stopping kernel is shared with the legacy generator. A predeclared
finite search uses budgets `1, 1/2, ..., 2^-12`, testing each independently
against continuous full-vector acceleration/jerk bounds and the original
`BrakingConfig` and `KinematicLimits`. Limits are not relaxed or reinterpreted
as tangential-only caps.

Admission additionally checks the complete unwrapped sideslip range until the
full stop, unambiguous heading increments and sampled kinematics. No certified
candidate gives `NO_CERTIFIED_PROFILE`, not proof of global infeasibility.
`model_report` retains attempted budgets, selected scale, continuous bounds
and model assumptions. `FallbackResult` retains full stopping time/path even
beyond the output horizon.
These conditional point-mass references do not certify tyres, steering,
actuators, obstacle clearance or driving safety.
