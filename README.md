# Тестовое задание для BND 

## Описание

Дан видеофайл crowd.mp4. Вам необходимо написать программу на языке Python, которая будет выполнять детекцию людей и их отрисовку на этом видео. Также нужно проанализировать полученный результат и сформулировать шаги по дальнейшему улучшению качества распознавания.

## Установка

1) Клонируйте данный репозиторий
2) Установить все необходимые библиотеки

```bash
pip install -r requirements.txt
```
## Запуск
1) Поменяйте путь до файла в глобальной переменной INPUT_VIDEO_PATH
2) Запустите программу
```bash
python main.py
```
## Выводы по результатам работы программы
Результаты работы программы показывают, что модель YOLOv5 успешно обнаруживает людей в видеофайле crowd.mp4. Однако, в процессе тестирования были замечено, что в  некоторых случаях модель могла пропустить людей или ошибочно идентифицировать объекты, что требует дальнейшей оптимизации.
## Шаги по дальнейшему улучшению:
1) Настройка модели: провести обучение на специализированных данных для повышения точности.
2) Анализ ошибок: проанализировать случаи, когда модель ошибается, чтобы понять, как улучшить алгоритм.
