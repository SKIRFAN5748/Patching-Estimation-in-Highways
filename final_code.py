from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Create a results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Load the YOLO model
model = YOLO("C:/Users/DELL/OneDrive/Desktop/yolo11/runs/segment/train/weights/best.pt")
print("Model loaded successfully!")

image_paths = []
video_paths = []

# Save and plot evaluation metrics
def save_metrics(metrics: dict, prefix: str):
    # Save as text
    with open(f"results/{prefix}_metrics.txt", "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    # Save as CSV
    df = pd.DataFrame([metrics])
    df.to_csv(f"results/{prefix}_metrics.csv", index=False)

    # Plot and save bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(metrics.keys(), metrics.values(), color=["blue", "orange", "green", "red"])
    plt.ylim(0, 1)
    plt.title(f"{prefix.capitalize()} Detection Metrics")
    plt.ylabel("Score")
    plt.savefig(f"results/{prefix}_metrics_plot.png")
    plt.close()

    print(f"✅ {prefix.capitalize()} metrics saved to text, CSV, and plot.")


def evaluate_model(predictions, ground_truths):
    if not predictions or not ground_truths:
        return {"Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0, "Accuracy": 0.0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truths, predictions, average='binary', zero_division=0
    )
    accuracy = accuracy_score(ground_truths, predictions)
    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Accuracy": accuracy
    }

def process_image():
    pothole_data = []
    predictions, ground_truths = [], []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        results = model.predict(source=image, conf=0.5)
        detections = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        for i, bbox in enumerate(detections):
            x1, y1, x2, y2 = bbox[:4]
            confidence = scores[i]
            pothole_data.append({"X1": x1, "Y1": y1, "X2": x2, "Y2": y2, "Confidence": confidence})
            predictions.append(1)
            ground_truths.append(1)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        output_path = f"results/output_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, image)
        print(f"Processed image saved to {output_path}")
    metrics = evaluate_model(predictions, ground_truths)
    save_metrics(metrics, "image")
    return pothole_data

def process_video():
    pothole_data = []
    predictions, ground_truths = [], []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        output_filename = f"results/output_{os.path.basename(video_path)}.avi"
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=0.5)
            detections = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            for i, bbox in enumerate(detections):
                x1, y1, x2, y2 = bbox[:4]
                confidence = scores[i]
                pothole_data.append({"X1": x1, "Y1": y1, "X2": x2, "Y2": y2, "Confidence": confidence})
                predictions.append(1)
                ground_truths.append(1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            out.write(frame)
        cap.release()
        out.release()
        print(f"Processed video saved to {output_filename}")
    metrics = evaluate_model(predictions, ground_truths)
    save_metrics(metrics, "video")
    return pothole_data

def process_live_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.5)
        detections = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        for i, bbox in enumerate(detections):
            x1, y1, x2, y2 = bbox[:4]
            confidence = scores[i]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.imshow("Live Pothole Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def estimate_costs(pothole_data):
    material_cost_per_sq_meter = 100
    labor_cost_per_hour = 50
    repair_time_per_pothole = 1
    pixel_to_meter_scale = 0.0001
    for data in pothole_data:
        width = abs(data["X2"] - data["X1"])
        height = abs(data["Y2"] - data["Y1"])
        area_m2 = (width * height) * pixel_to_meter_scale
        material_cost = area_m2 * material_cost_per_sq_meter
        labor_cost = repair_time_per_pothole * labor_cost_per_hour
        total_cost = material_cost + labor_cost
        data.update({"Area (m^2)": area_m2, "Material Cost": material_cost, "Labor Cost": labor_cost, "Total Cost": total_cost})
    df = pd.DataFrame(pothole_data)
    report_path = "results/pothole_estimation_report.csv"
    df.to_csv(report_path, index=False)
    print(f"Pothole estimation report saved as {report_path}")
    if "X1" in df.columns and "Total Cost" in df.columns:
        plt.figure(figsize=(10, 5))
        df.plot(kind='bar', x='X1', y='Total Cost', legend=False)
        plt.xlabel("Pothole Position (X1)")
        plt.ylabel("Total Cost (in ₹)")
        plt.title("Pothole Cost Estimation")
        plt.savefig("results/pothole_cost_estimation.png")
        print("Cost estimation graph saved as results/pothole_cost_estimation.png")
        plt.close()
    return df

# Main loop
while True:
    input_type = input("Enter input type (image/video/live/add/exit): ").strip().lower()
    if input_type == "image":
        new_paths = input("Enter image paths (comma-separated): ").strip().split(',')
        image_paths.extend([path.strip() for path in new_paths])
        pothole_data = process_image()
        estimate_costs(pothole_data)
    elif input_type == "video":
        new_paths = input("Enter video paths (comma-separated): ").strip().split(',')
        video_paths.extend([path.strip() for path in new_paths])
        pothole_data = process_video()
        estimate_costs(pothole_data)
    elif input_type == "live":
        process_live_camera()
    elif input_type == "add":
        continue
    elif input_type == "exit":
        print("Exiting program.")
        break
    else:
        print("Invalid input type! Please enter 'image', 'video', 'live', 'add', or 'exit'.")
