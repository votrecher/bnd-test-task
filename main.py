import cv2
import torch


INPUT_VIDEO_PATH = "crowd.mp4"
OUTPUT_VIDEO_PATH = "crowd_detected.mp4"


def load_model(model_name):
    """
    Загружает модель YOLOv5.

    :return: Загруженная модель YOLO.
    """
    return torch.hub.load("ultralytics/yolov5", "yolov5s")


def open_video(input_video_path):
    """
    Открывает видеофайл.

    :param input_video_path: Путь к видеофайлу.
    :return: Объект VideoCapture для чтения видео.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Ошибка при открытии видео: {input_video_path}")
    return cap


def create_video_writer(output_video_path, width, height, fps):
    """
    Создает объект для записи видео.

    :param output_video_path: Путь для сохранения выходного видео.
    :param width: Ширина видео.
    :param height: Высота видео.
    :param fps: Частота кадров видео.
    :return: Объект VideoWriter для записи видео.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


def process_frame(model, frame):
    """
    Обрабатывает кадр для детекции объектов.

    :param model: Загруженная модель YOLO.
    :param frame: Кадр видео для обработки.
    :return: Обработанный кадр с нарисованными bounding boxes.
    """
    results = model(frame)

    detections = results.xyxy[0]

    for *xyxy, conf, cls in detections.tolist():
        if int(cls) == 0:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"Person {conf * 100:.1f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return frame


def process_video(cap, model, out):
    """
    Обрабатывает видео в цикле, читая кадры, обрабатывая их и отображая результат.

    :param cap: Объект VideoCapture для чтения видео.
    :param model: Загруженная модель YOLO для обработки кадров.
    :param out: Объект VideoWriter для записи выходного видео.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(model, frame)
        out.write(processed_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = load_model()
    cap = open_video(INPUT_VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = create_video_writer(OUTPUT_VIDEO_PATH, width, height, fps)

    process_video(cap, model, out)
