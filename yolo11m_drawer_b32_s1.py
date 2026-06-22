from ultralytics import YOLO
import glob
import os
import pandas as pd
from datetime import datetime

print("\n")
full_date = datetime.now()
date = datetime.now().strftime("%d-%m-%Y")
print("Current date and time:", full_date)
print("\n")


print("INFO - yolov11m model, split 1 (75%, 15%,10%), class 4, pretrained weights, batchsize 32, 640 size")
print("\n")


#############################################

## import data
train_impath = "/cluster/home/sbhansali/digi_hiwi/drawers2/datasets/all_jpg_qr_scale_whitebal_crop_pad_1280_newSplit/train_s1/images/"
val_impath = "/cluster/home/sbhansali/digi_hiwi/drawers2/datasets/all_jpg_qr_scale_whitebal_crop_pad_1280_newSplit/val/images/"
test_impath = "/cluster/home/sbhansali/digi_hiwi/drawers2/datasets/all_jpg_qr_scale_whitebal_crop_pad_1280_newSplit/test/images/"
print("path to training image = ",train_impath)
print("path to val images = ",val_impath)
print("path to test images = ",test_impath)

tpath = []
tpath = glob.glob(train_impath + '*.jpg')
print("number of train images s1 =",len(tpath))
 
vpath = []
vpath = glob.glob(val_impath + '*.jpg')
print("number of val images =",len(vpath))

tspath = []
tspath = glob.glob(test_impath + '*.jpg')
print("number of test images =",len(tspath))

del tpath
del vpath
del tspath

#############################################


os.chdir('/cluster/home/sbhansali/digi_hiwi/drawers2/ultralytics')

print("\n")
print("STARTING TRAINING...")
print("\n")

# Load a model
model = YOLO("yolo11m.pt")
folder_name = f"yolo11m_drawer_b32_s1_{date}"

# Train the model
train_results = model.train(
    data = "/cluster/home/sbhansali/digi_hiwi/drawers2/ultralytics/data_s1.yaml",  # path to dataset YAML
    epochs =300,  # number of training epochs
    batch = 32, 
    save = True,
    imgsz = 640,  # training image size
    device= 'cpu',  # for CPU ='cpu'. For device to run on GPU, i.e. device=0 or device=0,1,2,3 ( depeneding on number of GPU you want to use.)
    name=folder_name,          # name of the experiment
)


print("\n\n")
print("done")

full_date = datetime.now()
print("Current date and time:", full_date)
print("\n")
