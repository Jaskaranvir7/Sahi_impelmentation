# Import required libraries
from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.file import download_from_url
import json
import csv

# Step 1: Download YOLOv8 model weights
yolov8_model_path = "models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)

# Step 2: Download a sample image
image_url = "https://c8.alamy.com/comp/2A68664/birds-eye-view-of-a-crowd-standing-close-together-in-a-grassy-area-on-a-sunny-day-during-a-college-orientation-week-event-and-looking-up-toward-the-camera-at-the-johns-hopkins-university-baltimore-maryland-september-4-2006-from-the-homewood-photography-collection-2A68664.jpg"
image_path = "demo_data/sample_image.jpeg"
download_from_url(image_url, image_path)

# Step 3: Load the YOLOv8 detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="CUDA"
)

# Step 4: Perform object detection with SAHI (Slicing Aided Hyper Inference)
sliced_result = get_sliced_prediction(
    image_path,
    detection_model=detection_model,
    slice_height=195,
    slice_width=195,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# Step 5: Extract predictions and convert to JSON
object_predictions = sliced_result.object_prediction_list

# Debugging: Print the bbox object to inspect its attributes
for obj in object_predictions:
    x_min, y_min, x_max, y_max = obj.bbox.to_xyxy()
    print(x_min, y_min, x_max, y_max)
# Convert object predictions to a dictionary format
predictions_dict = [
    {
        "category_id": int(obj.category.id),         # Convert to int if necessary
        "category_name": obj.category.name,
        "score": float(obj.score.value),              # Convert to native float
        "bounding_box": {
            "x_min": float(obj.bbox.minx),
            "y_min": float(obj.bbox.miny),
            "x_max": float(obj.bbox.maxx),
            "y_max": float(obj.bbox.maxy),
            "width": float(obj.bbox.maxx - obj.bbox.minx),
            "height": float(obj.bbox.maxy - obj.bbox.miny),
        }
    }
    for obj in object_predictions
]

# Convert dictionary to JSON string
predictions_json = json.dumps(predictions_dict, indent=4)

# Step 6: Print the JSON string
print(predictions_json) 
# Step 7: Export predictions to a CSV file
csv_file_path = "predictions.csv"
with open(csv_file_path, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header
    csv_writer.writerow(["category_id", "category_name", "score", "x_min", "y_min", "x_max", "y_max", "width", "height"])
    # Write prediction rows
    for obj in predictions_dict:
        bbox = obj["bounding_box"]
        csv_writer.writerow([
            obj["category_id"],
            obj["category_name"],
            obj["score"],
            bbox["x_min"],
            bbox["y_min"],
            bbox["x_max"],
            bbox["y_max"],
            bbox["width"],
            bbox["height"]
        ])

print(f"Predictions exported to {csv_file_path}")