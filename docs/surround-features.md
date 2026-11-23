# Общее изображение сцены для Fast и Bridge

Шесть направлений в `CameraSnapshot`: front, front-left, front-right, rear-left, rear-right, rear. Внутри сохраняем реальные имена камер. Экспозиция, опорное время и калибровка — разные поля: нельзя присвоить шести изображениям одно время кадра и назвать их синхронизированными. Калибровка переводит **метрические** координаты из единой опорной ego-системы в систему выбранной камеры.

```text
6 rectified RGB + ego reference + per-view calibration
  → one batched encoder call → [1,6,C,H,W] + resized intrinsics
                             → Fast(features) → forecast
                             → Bridge(the same features, the same forecast)
```

После resize на полной картинке обновляем intrinsics по той же пиксельной геометрии. Размер пиксельной калибровки и размер решётки признаков не взаимозаменяемы. `route_surround_scene` передаёт обоим потребителям тот же тензор, не пускает его через второй энкодер и отклоняет in-place запись обычными PyTorch-операциями. Это контракт однократного инференса; функции не должны изменять данные через внешние алиасы.

## Два входа

| Вход | Калибровка и время |
|---|---|
| **nuScenes v1.0-mini**, исходно извлечённый каталог | Шесть JPEG keyframe и соответствующие pose/exposure timestamps. Референс — **ego-поза** ключевого `LIDAR_TOP`, не его экстрики. Внешние калибровки каждой камеры приводим через её capture ego-позу к этому референсу. Лимит проверяет диапазон шести экспозиций и опорного времени. Скорость/ускорение не выводим из поз соседних кадров и не подставляем нули. |
| **OpenScene Mini**, закреплённый smoke-снимок | F0, L0, R0, L2, R2, B0 из восьмикамерного nuPlan rig. Проверяем локальные архивы и точные JPEG; выпрямляем исходную оптику. Общая привязка к LiDAR известна, отдельные экспозиции — нет: `source_association`, без измеренного skew. Это не nuScenes. |

[Схема nuScenes](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md), [времена камер и калибровки](https://www.nuscenes.org/data-format), [официальный OpenScene Mini](https://huggingface.co/datasets/OpenDriveLab/OpenScene).

## Диагностика

В закреплённом `Dockerfile.alignment` (Linux, сеть отключена, данные и веса смонтированы read-only) запускаем с локальным DINOv2:

```sh
python -m scripts.run_surround_probe --model-root /models --output /results/openscene-six \
  --revision '<commit SHA>' openscene --metadata-archive /data/openscene_metadata_mini.tgz \
  --camera-archive /data/openscene_sensor_mini_camera_16.tgz
```

Для исходного nuScenes Mini вместо `openscene …` передаём `nuscenes --root /data --sample-token TOKEN --max-skew-us LIMIT`. `result.json` сохраняет порядок камер, калибровки, часы, хеши карт и токенов. `cameras.png` остаётся локально.

Диагностика запускает **настоящие** DINOv2 и ROI-токенизатор. Fast здесь — проверка его входа, а ROI — фиксированные области для проверки стыка, не предсказанные объекты. Прогноз будущей 3D-сцены, кандидаты движения и VQA в этом прогоне не запускаются; обычный NAVSIM-агент не меняется.
