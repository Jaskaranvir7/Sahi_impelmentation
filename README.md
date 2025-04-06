# YOLOv8 Object Detection with SAHI

## Overview
This project demonstrates object detection using the YOLOv8 model integrated with SAHI (Slicing Aided Hyper Inference) for enhanced detection capabilities on large or complex images. It uses slicing techniques to improve prediction accuracy and outputs results in JSON format.

---

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- CUDA for GPU support (if available)
- Virtual environment manager (`uv` is recommended)
- Required libraries (`sahi`, `torch`, etc.)

---

## Setting Up the Environment

To set up the environment using `uv`:

1. Create and activate a virtual environment:
    ```bash
    uv new yolo_env
    uv activate yolo_env
    ```

2. Install the required dependencies:
    ```bash
        uv sync --dev
    ```

---

## Project Structure
The key steps in the script are:

1. **Download YOLOv8 Model**: Fetch the YOLOv8 model weights (`yolov8s.pt`).
2. **Download Sample Image**: Retrieve a sample image from a specified URL.
3. **Load Detection Model**: Initialize the YOLOv8 model with custom settings.
4. **Perform Object Detection**: Use SAHI slicing methods to detect objects.
5. **Generate JSON Output**: Extract bounding boxes and prediction details, then save the output in JSON format.

---

## How to Run
1. Clone this repository or copy the script locally.
2. Execute the script with Python:
    ```bash
    python script_name.py
    ```
3. The script will:
   - Download the YOLOv8 model weights.
   - Fetch the sample image and perform object detection.
   - Output prediction results in JSON format.

---

## Output
The output is printed to the console in JSON format, containing:
- **Category ID**
- **Category Name**
- **Prediction Score**
- **Bounding Box Dimensions** (x_min, y_min, x_max, y_max, width, height)

---

## Debugging
For troubleshooting, bounding box coordinates are printed for each detected object. This ensures proper loading and inference of the YOLOv8 model.

---

## Notes
- Update `image_url` and `image_path` variables for custom images.
- Modify slicing parameters (`slice_height`, `slice_width`, etc.) to match image dimensions and inference requirements.
- Ensure GPU compatibility for faster processing by setting `device="CUDA"`.

---

## License
This project follows an open-source framework for educational and non-commercial use.

---

## Credits
- Developed with [SAHI](https://github.com/obss/sahi) for object detection slicing.
- Powered by YOLOv8 detection model.