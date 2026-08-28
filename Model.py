import os
import glob
import pandas as pd
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
import yaml
import random
from PIL import Image

Class_Names = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'No Tumor',
    3: 'Pituitary'
}


dataset_path = r"C:\Users\omar\Desktop\MLA Project\Brain Tumor with Bounding Boxes"
train_path = r"C:\Users\omar\Desktop\MLA Project\Brain Tumor with Bounding Boxes\Train"
val_path = r"C:\Users\omar\Desktop\MLA Project\Brain Tumor with Bounding Boxes\Val"


def verify_dataset(base_path, split_name):

    image_paths = glob.glob(os.path.join(base_path, '**', 'images', '*.jpg'), recursive=True)

    clean_images_paths = []
    corrupt_images = []
    missing_labels = []
    empty_labels = []
    invalid_boxes = []


    for img_path in image_paths:
        
        
        cv_img = cv2.imread(img_path)
        if cv_img is None:
            corrupt_images.append(img_path)
            continue

        
        txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        if not os.path.exists(txt_path):
            missing_labels.append(img_path)
            continue

        
        with open(txt_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            empty_labels.append(txt_path)
            continue

        
        has_invalid_box = False
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                invalid_boxes.append((txt_path, "Wrong column count"))
                has_invalid_box = True
                break

            try:
                cls_id, x, y, w, h = map(float, parts)
                if int(cls_id) not in Class_Names.keys():
                    invalid_boxes.append((txt_path, f"Unknown Class ID: {cls_id}"))
                    has_invalid_box = True
                    break
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                    invalid_boxes.append((txt_path, f"Coordinates out of bounds: x={x}, y={y}, w={w}, h={h}"))
                    has_invalid_box = True
                    break
            except ValueError:
                invalid_boxes.append((txt_path, "Non-numeric values in label"))
                has_invalid_box = True
                break

        if not has_invalid_box:
            clean_images_paths.append(img_path)

    #print(f"Total Scanned:  {len(image_paths)}")
    #print(f"Clean Kept:     {len(clean_images_paths)}")
    #print(f"Corrupt Images: {len(corrupt_images)}")
    #print(f"Missing Labels: {len(missing_labels)}")
    #print(f"Empty Labels:   {len(empty_labels)}")
    #print(f"Invalid BBoxes: {len(invalid_boxes)}\n")

    return clean_images_paths



train_images_paths = verify_dataset(train_path, "Train")
val_images_paths = verify_dataset(val_path, "Val")


train_counts = [len(os.listdir(os.path.join(train_path, cls, 'images'))) for cls in Class_Names.values()]
val_counts = [len(os.listdir(os.path.join(val_path, cls, 'images'))) for cls in Class_Names.values()]


'''
train_txt_path = os.path.abspath('train_list.txt')
val_txt_path = os.path.abspath('val_list.txt')

with open(train_txt_path, 'w') as f:
    f.write('\n'.join(train_images_paths))

with open(val_txt_path, 'w') as f:
    f.write('\n'.join(val_images_paths))
'''

'''
yaml_content = {
    'train': train_txt_path,
    'val': val_txt_path,
    'names': Class_Names
}

with open('data.yaml', 'w') as f:
    yaml.dump(yaml_content, f ,default_flow_style=False )

'''



'''
classes = list(Class_Names.values())
x = np.arange(len(classes))  
            
plt.figure(figsize=(9, 5))

bars_train = plt.bar(x - 0.2, train_counts, 0.4, label='Train', color='#4C72B0')

bars_val = plt.bar(x + 0.2, val_counts, 0.4, label='Validation', color='#DD8452')

plt.title("Data Distributionb:Train & Val", fontsize=14, fontweight="bold")
plt.xticks(x, classes)
plt.legend()

for bar in bars_train:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 15, str(yval), ha='center', fontweight='bold')

for bar in bars_val:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 15, str(yval), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
'''


'''
box_widths = []
box_heights = []


for img_path in train_images_paths + val_images_paths:
    txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    box_widths.append(float(parts[3]))
                    box_heights.append(float(parts[4]))


plt.figure(figsize=(7, 5))
plt.scatter(box_widths, box_heights, alpha=0.3, color='purple')

plt.title("Tumor Dimensions (Width vs Height)", fontsize=14, fontweight="bold")
plt.xlabel("Width ", fontsize=12)
plt.ylabel("Height ", fontsize=12)
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.tight_layout()
plt.show()
'''


'''
image_shapes = []

for img_path in np.array(train_images_paths + val_images_paths):
    with Image.open(img_path) as img:
        image_shapes.append(img.size) 

widths, heights = zip(*image_shapes)

plt.figure(figsize=(7, 4))
plt.hist2d(widths, heights, bins=20, cmap='Purples')
plt.colorbar(label='Image Count')
plt.title("Native Image Resolution Distribution", fontsize=12, fontweight="bold")
plt.xlabel("Width (Pixels)", fontsize=10)
plt.ylabel("Height (Pixels)", fontsize=10)
plt.tight_layout()
plt.show()
'''

'''
def visualize_samples(image_list, num_samples):
    sample_paths = random.sample(image_list, num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(16, 4))
    
    for ax, img_path in zip(axes, sample_paths):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_img, w_img, _ = img.shape
    
        txt_path = img_path.replace('images', 'labels').replace('.jpg', '.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, x, y, w, h = map(float, parts)
                        x1 = int((x - w / 2) * w_img)
                        y1 = int((y - h / 2) * h_img)
                        x2 = int((x + w / 2) * w_img)
                        y2 = int((y + h / 2) * h_img)
                        
                        label_name = Class_Names.get(int(cls_id), 'Unknown')
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(img, label_name, (x1, max(y1 - 5, 15)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(os.path.basename(img_path), fontsize=10)
        
    plt.tight_layout()
    plt.show()


visualize_samples(train_images_paths, num_samples=2)
'''


"""
model = YOLO('yolov8n.pt')  

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    workers=4,
    name='yolov8n_baseline',
    seed=42,
    patience = 10,
)
"""
