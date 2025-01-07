import cv2
from ultralytics import YOLO
import torch
from SSD_Model.ssd_model import SSDLite


def run_ssd_detection(model_path='ssd_best_model.pth'):
    device = torch.device('cpu')
    model = SSDLite(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            face_roi = frame[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (320, 320))
            face_normalized = face_resized / 255.0
            face_tensor = torch.FloatTensor(face_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                loc_preds, conf_preds = model(face_tensor)

            loc_preds = loc_preds.view(-1, 4)
            conf_preds = conf_preds.view(-1, 3)

            best_conf = 0
            best_box = None
            best_label = None

            for loc, conf in zip(loc_preds, conf_preds):
                confidence = torch.max(conf).item()
                if confidence > 0.7 and confidence > best_conf:
                    best_conf = confidence
                    best_box = loc.tolist()
                    best_label = torch.argmax(conf).item()

            if best_box:
                x_min = x
                y_min = y
                x_max = x + w
                y_max = y + h

                label = "Masked" if best_label == 0 else "Unmasked"
                box_color = (0, 255, 0) if best_label == 0 else (0, 0, 255)  # Kırmızı renk unmasked için
                text_color = (255, 255, 255)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 2)
                cv2.putText(frame, f"{label} ({best_conf:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        cv2.imshow("SSD Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_yolo_prediction():
    model = YOLO(fr'best.pt')
    CLASS_NAMES = ['Incorrectly Weared Mask', 'With Mask', 'Without Mask']

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        boxes = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
        class_ids = results[0].boxes.cls

        iou_threshold = 0.4
        conf_threshold = 0.5

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                                   score_threshold=conf_threshold, nms_threshold=iou_threshold)

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                conf = confidences[i]
                cls_id = class_ids[i]

                x1, y1, x2, y2 = map(int, box)
                class_name = CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else "Unknown"

                color = (0, 255, 0) if class_name == "With Mask" else (0, 0, 255)  # Kırmızı renk Without Mask için
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = f"{class_name}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('YOLOv8 Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def main():
    print("Choose a model to run:")
    print("1. SSD")
    print("2. YOLO")

    choice = input("Enter your choice (1/2): ").strip()

    if choice == '1':
        print("Running SSD Model...")
        run_ssd_detection()
    elif choice == '2':
        print("Running YOLO Model...")
        run_yolo_prediction()
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
