# Нейросетевой Fast и воспроизводимые сравнения

`LatentTransfuserAgent` — официальный NAVSIM v1.1 Latent TransFuser с готовым checkpoint. Это отдельный baseline, не реализация или переименование FSDrive. Используем CPU, `latent=True` и нативные граф/предобработку закреплённого upstream.

## Вход и веса

Модель получает текущие `CAM_L0`, `CAM_F0`, `CAM_R0` и ego-state: navigation command из четырёх one-hot значений, скорость и ускорение. Crop/stitch/resize 1024×256 выполняет штатный feature builder. Проверяем RGB uint8, доступность камер и форму данных до inference.

В latent-ветке upstream заменяет LiDAR-признаки обученным параметром, используя от входного LiDAR только размер batch. Наш адаптер передаёт пустой batch-carrier, не искусственное облако точек. Отдельный weighted integration сравнивает **все** выходы с нативной веткой, получившей ненулевой LiDAR histogram. Для обычного TransFuser этот обход запрещён.

Upstream не использует camera intrinsics/extrinsics или timestamps в этой модели. Мы не выдаём кадр NAVSIM за измеренную синхронизацию экспозиций и не переименовываем этот rig в шесть камер FSDrive. Полный camera/clock-контракт целевой интеграции остаётся отдельной задачей.

`scripts/ltf-sample-sources.json` фиксирует revision, размеры и SHA256 LTF checkpoint и offline ResNet34. Encoder сначала создаёт исходный граф через поддерживаемый timm `local-dir`; затем **все** параметры модели загружаются из полного LTF checkpoint с `strict=True`. Никаких случайных весов вместо отсутствующего checkpoint и скрытых сетевых загрузок при inference.

Загрузчик потоково читает официальный camera shard, сохраняет только три allowlisted JPEG и прекращает чтение после последнего. Проверяем commit/ETag/размер источника, JPEG decode и сохраняем индивидуальные hashes. Полный SHA архива при чтении префикса **не** считается проверенным. Архив не сохраняется. Повреждённые существующие данные требуют явного исправления; они не заменяются молча.

## Один вызов модели — парное сравнение

`scripts/run_neural_sample.py` один раз вызывает HybridDriveAgent с нейросетевым Fast:

1. Сохраняем исходный Fast-кандидат из immutable `CandidateSet` решения.
2. Оцениваем этот кандидат как `baseline`.
3. Оцениваем выбранную Safety траекторию как `checks_only`.
4. Отдельно оцениваем `ConstantVelocityAgent` как `cv_reference`, не выдавая другую модель за абляцию.

Все режимы используют одну сцену, sampling и metric cache. Native `pdm_score` получает именно сохраняемые/отрисованные точки. Будущая человеческая траектория используется только evaluator для ADE/FDE, после получения ответа агента.

Качество не является условием успешности исследования: PDMS может быть нулевым или ухудшиться. При этом ошибка inference/scorer или failed fallback остаётся отдельной неуспешной попыткой. Форма конечного placeholder не превращает её в успешную сцену. CLI сохраняет результаты и возвращает ошибку, если любой обязательный режим не отработал.

## Протокол абляций

| Режим | Дополнительные deterministic checks | Структурированный Slow veto |
|---|---|---|
| baseline | нет | нет |
| checks_only | да | нет |
| slow_only | нет | да |
| combined | да | да |

Во всех режимах обязательны корректный формат/время/конечные числа. Для режимов с veto используем одинаковый тормозной fallback. `slow_only` не отключает валидацию результата или проверку самого fallback. Пока Slow-модель не подключена, `slow_only`/`combined` явно **недоступны**, а не воспроизводятся через фиктивные ответы. Текущий experiment manifest это отражает.

Парные абляции требуют одинаковых code/upstream revisions, весов, кадров, preprocessing, seed, sampling, Fast-кандидата и fallback policy. Проверка `require_paired` отклоняет изменение любого из этих условий. `cv_reference` — только внешняя точка сравнения.

## Данные и отсутствие утечки

Аудит читается из закреплённых официальных NAVSIM YAML; SHA256 каждого источника проверяется. `navtrain` и `navtest` не должны пересекаться по logs, tokens или recording-группам. Recording — имя без двух заключительных диапазонов кадров; соседние сегменты одной записи не распределяем между ролями.

Для собственного обучения/подбора параметров делим **только navtrain**: сортируем recording-группы по SHA256 `hybrid-drive-navtrain-val-v1:0:<recording>`, первые `ceil(20% × recordings)` назначаем validation, остальные train. Это доля recording-групп, не кадров. `navtest` оставляем для фиксированной финальной оценки; thresholds/prompts на нём не настраиваем. Полные log allowlists и hashes сохраняются в отчёте. Число tokens внутри собственного train/validation не выдумываем без соответствующей полной metadata.

Текущий OpenScene mini используется только как **smoke/debug**. Выбранный лог и token входят в официальный navtrain; этот прогон нельзя называть holdout-оценкой. Состав обучения конкретных pretrained-весов известен не полностью. Сама mini-выборка не доказывает научный выигрыш.

## Что сохраняем и считаем

- `experiment-manifest.json`: точный код/upstream, веса и hashes, snapshot manifest входов, политика, seed, sampling, исходный кандидат и split audit.
- `decision.json`: причины выбора, кинематические отчёты, Slow status и фактические native-точки.
- `result.json`, `comparison.csv`, per-mode CSV и `trajectory.png`: все обязательные попытки, включая failures.
- PDMS и его NC/drivable-area/progress/TTC/comfort/direction-компоненты; не смешиваем PDMS и EPDMS.
- ADE — средняя XY Euclidean ошибка по совпадающим будущим временам; FDE — XY ошибка последней точки. Метры, без heading и без неявной интерполяции.

В сводке отдельно показываем valid/failed и failure rate. `score_mean_all_attempts_failure_zero` штрафует failure нулём; valid-only средние подписаны отдельно. Failure не превращаем в измеренное столкновение или нулевую L2. All-veto и стоимость обработки также сохраняем; прогресс нужен, чтобы постоянное торможение не выглядело улучшением.

Для детерминированного CPU smoke фиксируем seed и deterministic algorithms. Для будущего стохастического Slow заранее задаём не менее пяти seeds/repeats; сравниваем парные разности на одинаковых сценах и bootstrap-интервалы по целым recording-группам, не по соседним кадрам. На одной debug-сцене статистические выводы/интервалы выигрыша не строим.

## Запуск и изоляция

Команды находятся в [описании окружения](environment.md). Dataset-free suite запускается отдельно от `tests/integration`; обычный тест не скачивает веса. Neural workflow обязательно получает настоящие assets и запускает weighted test без skip. Inference/evaluator работают без сети с readonly data/assets; в артефакты не выгружаем веса и оригинальные кадры.
