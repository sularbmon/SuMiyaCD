import torch
import gc
gc.collect()
torch.cuda.empty_cache()

from ultralytics import YOLO

model = YOLO('runs/segment/train4/weights/best.pt')  # load a custom model

# Predict with the model
results = model('D:\\815_CowDataChecking\\20221228\\20221228_E_cow\\resize\\',save=False,retina_masks=False,show=False,device=0,imgsz=(1080,1080))