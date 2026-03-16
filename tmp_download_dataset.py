from roboflow import Roboflow
import os

# Initialize with the API key identified
rf = Roboflow(api_key="HAeSNKfVyi8ZXZ330Y9J")

# Access the project and version
project = rf.workspace("harshith-xfwsa").project("lanyard-nb33q")
version = project.version(4)

# Download the dataset in YOLOv8 format to the datasets directory
os.makedirs("datasets", exist_ok=True)
dataset = version.download("yolov8", location="datasets/roboflow_id_dataset")

print(f"Dataset downloaded to: {dataset.location}")
