# BITNO!! INSTALIRATI NAVEDENO U TERMINALU: pip install 'git+https://github.com/facebookresearch/segment-anything.git'
import os
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import supervision as sv

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

predictor = SamPredictor(sam)
mask_generator = SamAutomaticMaskGenerator(sam)

def upload_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Image File")
    return file_path
  
print("Upload an image:")
IMAGE_PATH = upload_image()
if IMAGE_PATH:
    img = cv2.imread(IMAGE_PATH)
   
else:
    print("No image uploaded.")

points = []
segmented_masks = []

def mouse_click(event, x, y, flags, param):
    global points, segmented_masks

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))  
        cv2.circle(image_rgb, (x, y), 10, (255, 0, 0), -1) 
        cv2.imshow('image', image_rgb)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) >= 2:
            segmented_masks.append(np.array(points))  #korisnicki oznacene tocke se spremaju kao maska
            points = []
            cv2.circle(image_rgb, (x, y), 10, (0, 0, 255), -1)  
            cv2.imshow('image', image_rgb)
        else:
            print("At least two points are needed to define a bounding box.")
            points = []

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.imread(IMAGE_PATH)

cv2.imshow('image', image_rgb)
cv2.setMouseCallback('image', mouse_click) 
cv2.waitKey(0)
cv2.destroyAllWindows()

for mask_points in segmented_masks:
    x_min = min([point[0] for point in mask_points])
    y_min = min([point[1] for point in mask_points])
    x_max = max([point[0] for point in mask_points])
    y_max = max([point[1] for point in mask_points])
    box = np.array([x_min, y_min, x_max, y_max])

    predictor.set_image(image_rgb)

    masks, _, _ = predictor.predict(box=box, multimask_output=True)       

    mask_annotator = sv.MaskAnnotator(color=sv.Color.RED, color_lookup=sv.ColorLookup.INDEX)

    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks=masks),
        mask=masks
    )
    detections = detections[detections.area == np.max(detections.area)]
    segmented_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    # Da ne bude vise segmentiranih slika, dodano da sve slike segmentiranja spoje u jednu
    alpha = 0.5 
    cv2.addWeighted(segmented_image, alpha, image_bgr, 1 - alpha, 0, image_bgr)

cv2.imshow('Segmented Image', image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
