import cv2
import torch
import numpy as np


def run_yolov5_webcam_detection(conf_threshold=0.5):
    # Загрузка модели
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()

    # Открытие веб-камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть веб-камеру")
        return

    print("Детекция запущена. Нажмите 'q' для выхода...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Детекция объектов
            results = model(frame)

            # Получение результатов в формате pandas DataFrame
            df = results.pandas().xyxy[0]

            # Визуализация результатов
            for _, row in df[df['confidence'] > conf_threshold].iterrows():
                x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{row['name']} {row['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Webcam Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    run_yolov5_webcam_detection(conf_threshold=0.6)