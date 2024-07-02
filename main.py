from ultralytics import YOLO
import torch

# If Ensemble is no longer available, comment it out or find the new equivalent
# from ultralytics.nn.modules import Ensemble

if __name__ == "__main__":

    # 0. There are a lot of requirements to create/use OpenCV projects. I used the following video to get started
    # https://youtu.be/WgPbbWmnXJ8

    # 1. Download the dataset specified in config.yaml
    # 2. Put the downloaded folders into your project and change absolute paths in config.yaml
    # 3. Change the following text to the correct relative paths for your project
    # 4. Run this
    # 5. Go to runs/detect/train/weights and use best.pt as the model while running shot_detector.py

    # Load a model# Set to your desired GPU number
    model = YOLO('Yolo-Weights/yolov8n.pt')
    print(torch.cuda.is_available())

    # Train the model
    results = model.train(data='config.yaml', epochs=100, imgsz=640)
