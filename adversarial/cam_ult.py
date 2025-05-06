import cv2
from ultralytics import YOLO
import os

def run_yolov5_webcam_detection(conf_threshold=0.5):
    model = YOLO("yolov5s.pt")

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

            results = model.predict(source=frame, conf=conf_threshold, verbose=False)

            # Визуализация результатов
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, class_ids):
                    if conf < conf_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{result.names[cls_id]} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv5 Ultralytics', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    run_yolov5_webcam_detection(conf_threshold=0.6)


if __name__ == '__main__':
    main()
