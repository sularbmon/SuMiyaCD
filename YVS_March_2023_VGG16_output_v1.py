### run here first
import argparse
import os
import sys
from pathlib import Path
import imutils
import numpy as np
from PIL import Image
import math
from collections import deque
import cv2

import glob
import cv2
import torch
import torch.backends.cudnn as cudnn
import tensorflow as tf 
import pandas as pd
import nas_video_module as nas
from re import match

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from datetime import datetime
from timer import Timer
from matplotlib import pyplot 
import tkinter
import matplotlib
matplotlib.use( 'tkagg' )
import time
import pickle
#load model
#vgg_filename='vgg16_nov_model.h5'
#svm_filename ='svm_nov_model.pkl'
#le_filename = 'label_encode.le'

vgg_filename='ALane_jan_2023_v3_models/vgg16_jan_v3_model.h5'
svm_filename = 'ALane_jan_2023_v3_models/svm_jan_v3_model.pkl'
le_filename = 'ALane_jan_2023_v3_models/label_jan_v3_encode.le'


clf = pickle.load(open(svm_filename, 'rb'))
le = pickle.load(open(le_filename, 'rb'))
VGG_model =  tf.keras.models.load_model(vgg_filename, compile = False)

##############

import gc

gc.collect()

torch.cuda.empty_cache()
def Predict_SVM(image):  ## new with vgg
    square = 4
    ix = 1
#Check results on a few select images
    #n=np.random.randint(0, x_test.shape[0])
    img = image

    #plt.imshow(img)
    input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
    input_img_feature=VGG_model.predict(input_img)
    
    for fmap in feature_maps:
        #plot cattle image
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(fmap[0, :, :, ix-1], cmap='viridis')
                ix += 1
        # show the figure
        pyplot.show()
    
    
    input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
    prediction_RF = clf.predict(input_img_features)[0] 
    prediction_RF = le.inverse_transform([prediction_RF])  #Reverse the label encoder to original name
    #label = [str(COW_MAPPER[int(prediction_RF)][0])]
    #print("The prediction for this image is: ", prediction_RF)
    #predict_proba= clf.predict_proba(input_img_features)
    #print(predict_proba)
    #print("max predict value is "+str(predict_proba.max()) )
    
    #decision_svc= clf.predict_proba(input_img_features)
    #probs_svc = (decision_svc - decision_svc.min()) / (decision_svc.max() - decision_svc.min())
    #print("decision probability "+str(probs_svc))
    
    return prediction_RF
print("defined RF new VGG SVM")
def Is_Duplicate_Id(y1,y2,id):
    global PREVIOUS_ID
    global PREVIOUS_Y1
    global PREVIOUS_Y2
    global PREVIOUS_LOCAL_IDS
    global CATTLE_LOCAL_ID
    
    try: 
        index = PREVIOUS_ID.index(id)
        #print('I reached here')
        if(PREVIOUS_Y1[index]+321<=y1 and PREVIOUS_Y2[index]+371<y2): #duplicate from bottom
            #if(id in PREVIOUS_LOCAL_IDS):
            #print('id: ',id,' LOCAL_ID: ',CATTLE_LOCAL_ID)
            #return PREVIOUS_LOCAL_IDS[id][0]
            
            #print('This is not gonna happen again')
            #PREVIOUS_LOCAL_IDS.append([id,LOCAL_ID])
            #PREVIOUS_Y[index]=
            #print('PREVIOUS ID')
            #CATTLE_LOCAL_ID +=1
            #CATTLE_LOCAL_ID += 1
            
            #print('except')
            PREVIOUS_ID.append(CATTLE_LOCAL_ID)
            PREVIOUS_Y1.append(y1)
            PREVIOUS_Y2.append(y2)
            #print('New Cattle Id')
            return CATTLE_LOCAL_ID
        #elif(PREVIOUS_Y[index]+400<center): #stepping back
        #    if(id in PREVIOUS_LOCAL_IDS):
        #        return PREVIOUS_LOCAL_IDS[id][0]
        else:
            #print('Oh. here ? really?')
            PREVIOUS_Y1[index]=y1 #duplicate is solved or no duplicate and just need for last y 
            PREVIOUS_Y2[index]=y2
            #return PREVIOUS_LOCAL_IDS[index][1]
            
            #update('PREVIOUS Y')
            return PREVIOUS_ID[index]
    except:
        #print(PREVIOUS_ID)
        #print(id)
        CATTLE_LOCAL_ID += 1
        #print('except')
        PREVIOUS_ID.append(CATTLE_LOCAL_ID)
        PREVIOUS_Y1.append(y1)
        PREVIOUS_Y2.append(y2)
        return id
#def Take_Prev_Label(y2,id,cow_srno):
def Take_Prev_Label(y,h,id,cow_srno):
    global STORED_IDS
    global STORED_MID_Y
    global STORED_MID_Y1
    global STORED_MID_Y2
    global STORED_MISS
    global LAST_SEEN_IDS
    global LAST_SEEN_ID_CENTROIDS
    global CATTLE_LOCAL_ID
    global IS_FIRST_CATTLE 
    y1 , y2 = y , y+h
    
    if IS_FIRST_CATTLE:
        IS_FIRST_CATTLE = False
        id = CATTLE_LOCAL_ID
    #mid_y = y2
    mid_y = int(2*y + h)/2
    IS_NEW = True
    last_id = 999
    last_y1 = 0
    last_y2 = 0
    if(len(STORED_IDS)>0): 
        last_id = STORED_IDS[len(STORED_IDS)-1]
        last_y1 = STORED_MID_Y1[len(STORED_MID_Y1)-1]
        last_y2 = STORED_MID_Y2[len(STORED_MID_Y2)-1]
        MISSED_LEN = len(STORED_MISS)
        #if(IS_NEW):
        
        #    MISSED_LEN -=1
        removed = 0
        for i in range(MISSED_LEN):
            #print(i, ' missed index checking' )
            missed = STORED_MISS[i-removed]
            #print('checking ',i-removed, 'to remove')
            if((missed>100 and len(STORED_MISS)>0) or int(last_id)-1>int(STORED_IDS[i-removed])): #if missed 35 frames
    
                del STORED_MISS[i-removed]  
                del STORED_MID_Y[i-removed]
                del STORED_MID_Y1[i-removed]
                del STORED_MID_Y2[i-removed]
                del STORED_IDS[i-removed]
                removed+=1
                #print('removed')
                
    #clear misses
   
    
    threshold_1 = 250 #300
    threshold_2 = 300  #230
    Distance = 2000
     
    if mid_y <= 1300 or mid_y >= 700:
        threshold_1 = 320 #350
        threshold_2 = 370 #280
    for i in range(1,len(STORED_MID_Y)+1):
        #print(STORED_IDS[-i-1],STORED_MID_Y[-i-1],' ',i)
        
        
        #if(STORED_MID_Y[-i]+threshold_2>=mid_y and STORED_MID_Y[-i]-threshold_1<=mid_y): # and IS_NEW): #previous 150 #200
        if(STORED_MID_Y1[-i]-threshold_1<=y1 and STORED_MID_Y1[-i]+threshold_1-100>=y1) or (STORED_MID_Y2[-i]-threshold_2<=y2 and STORED_MID_Y2[-i]+threshold_2-100>=y2): # and IS_NEW): #previous 150 #200
            if(IS_NEW):
                #print('mid_y ',mid_y,'existing y ',STORED_MID_Y[-i])
                #print('all mid_y ',STORED_MID_Y) 
                #print("Old")
                #print("STORED_MID_Y1",STORED_MID_Y1[-1], " and STORED_MID_Y2", STORED_MID_Y2)
                #print("Y!",y1, " and Y@", y2)
                
                Distance = abs(STORED_MID_Y1[-i] - y1)
                if(abs(STORED_MID_Y2[-i] - y2)<Distance):
                    Distance = abs(STORED_MID_Y2[-i] - y2)
                IS_NEW = False
                STORED_MID_Y1[-i] = y1
                STORED_MID_Y2[-i] = y2
                
                STORED_MISS[-i]=1
                id= STORED_IDS[-i]
                #print(Distance)
                #print(id)
                
            #try:
            #    exist_index = LAST_SEEN_IDS.index(id)
            #    if(LAST_SEEN_ID_CENTROIDS[exist_index]+200>y): # showing old id
            #        LAST_SEEN_ID_CENTROIDS[exist_index] = y
            #except:
            #print('corrected id :',STORED_IDS[-i])
            elif Distance >60:
                STORED_MISS[-i]+=1
            else:
                STORED_MISS[-i]= 15 #reset count to 2 when not moving
        elif(STORED_MID_Y1[-i]<=y1 and STORED_MID_Y2[-i]>=y2):
                STORED_MISS[-i]=5
        else:
            STORED_MISS[-i]+=1    
                
        #elif(cow_srno==1):
                
    #print(STORED_IDS,' IDS ',STORED_MID_Y,' SMY ',mid_y,' mid_y')
    if(IS_NEW):
        #print('SMY: ',STORED_MID_Y,', new my:',mid_y) 
        #print('new id: ',id)
        updatedID = Is_Duplicate_Id(y1,y2,id)
        if(int(last_id) <int(updatedID) and y1<last_y1-150 and y2<last_y2-150): # duplicate cattle with increased cattleID
            CATTLE_LOCAL_ID-=1
            for i in range(len(STORED_MID_Y)-1,0,-1):
                STORED_MISS[i]=15
            return -1
        if(int(last_id)-1>int(updatedID)):
            return -1
            
    #if(updatedID!=id):
    #    print('orgID: ',id,' updated ID: ',updatedID)
        #id = str(updated_ID)+'_'+str(id)
        
        id=CATTLE_LOCAL_ID
        STORED_IDS.append(id)
        STORED_MID_Y.append(mid_y)
        STORED_MID_Y1.append(y1)
        STORED_MID_Y2.append(y2)
        STORED_MISS.append(1)
    
    #print('returned id :',id)
    
    print(id)
    
    result = []
    result.append(str(id-1))
    
    #region remove stored id
    removed = 0
    for i in range(len(STORED_MID_Y)-1,0,-1):
        if(y1>STORED_MID_Y1[i] and y2>STORED_MID_Y2[i]):
             del STORED_MISS[i-removed]  
             del STORED_MID_Y[i-removed]
             del STORED_MID_Y1[i-removed]
             del STORED_MID_Y2[i-removed]
             del STORED_IDS[i-removed]
             removed+=1
                 
    return result


def calculate_most_cattle_id():
    global current_cow
    global excel_cow_count
    global final_result
    global fial_total
    global final_percentage
    global prev_id_record
    maxpos = excel_cow_count.index(max(excel_cow_count))
    #or i in range (len(current_cow)):
    #   print('cattle ',current_cow[i],' id is ',excel_cow_count[i] , ' count(s)')
    cattle_id = current_cow[maxpos]
    
    final_total.append(sum(excel_cow_count))
    final_percentage.append(max(excel_cow_count)/sum(excel_cow_count))
    final_result.append(cattle_id)
    excel_cow_count = [] #reset
    current_cow = [] #reset
    

def Generate_Cattle_Id_By_Apperance(csv_path,save_dir):
    print(csv_path, " is csv_path and ", save_dir , " is save_dir")

    data = pd.read_csv(csv_path)

    list_of_csv = [list(row) for row in data.values]
    global  final_result 
    global final_percentage
    global final_total
    global current_cow
    global excel_cow_count
    prev_id_record = [] 
    prev=None



    for i in range (len(list_of_csv)):
        #rint('from ',list_of_csv[i][1],' to ',list_of_csv[i][0])
        filtered_id = list_of_csv[i][0]
        actual_id = list_of_csv[i][1]
        if(prev!=filtered_id):
            if(prev is not None):
                calculate_most_cattle_id()
                prev_id_record.append(prev)
            prev = filtered_id

        try: 
            index = current_cow.index(actual_id)
            #print('I reached here')
            excel_cow_count[index]+=1
        except:
            current_cow.append(actual_id)
            excel_cow_count.append(1)


    df = pd.DataFrame(final_result, columns = ["ID"])
    try:
        final_percentage = torch.tensor(final_percentage, device = 'cpu')
        final_total = torch.tensor(final_total, device = 'cpu')
        final_prev = torch.tensor(prev_id_record, device = 'cpu')

        df["total"] = final_total
        df["percentage"] = final_percentage
        df["prev_id"]=prev_id_record
    except:
        df["total"] = final_total
        df["percentage"] = final_percentage
        df["prev_id"]=prev_id_record
    now=str(datetime.now().date())

    df.to_csv(save_dir+"/MaxCattleId_new "+now+'.csv', index= False)
    print("successfully saved")


#csv_path = "D:\\Python\\SULarbmon\\Python\\env\\yolov5\\runs\\detect_SVM_NV_demo_center\\exp_3_fps79\\1\\3849\\3849.csv"
def CALCULATE_MAX_CATTLE_ID(csv_path):
    print(csv_path, " is csv_path and ")

    data = pd.read_csv(csv_path)

    list_of_csv = [list(row) for row in data.values]
    
    prev_id_record = [] 
    prev=None

    current_cow = []
    excel_cow_count = []
    boxes = []
    file_locations = []

    for i in range (len(list_of_csv)):
        #rint('from ',list_of_csv[i][1],' to ',list_of_csv[i][0])
        filtered_id = list_of_csv[i][0]
        actual_id = list_of_csv[i][1]
        file_locations.append(list_of_csv[i][2])
        boxes.append([list_of_csv[i][3],list_of_csv[i][4],list_of_csv[i][5],list_of_csv[i][6]])
        #print(list_of_csv[i][2])
        try: 
            index = current_cow.index(actual_id)
            #print('I reached here')
            excel_cow_count[index]+=1
        except:
            current_cow.append(actual_id)
            excel_cow_count.append(1)
    
    maxpos = excel_cow_count.index(max(excel_cow_count))
    #or i in range (len(current_cow)):
    #   print('cattle ',current_cow[i],' id is ',excel_cow_count[i] , ' count(s)')
    cattle_id = current_cow[maxpos]
    #print(cattle_id)
    #print(current_cow)
    #print(excel_cow_count)
    return cattle_id,file_locations,boxes
default = "D:\\Python\\SULarbmon\\Python\\env\\yolov5\\runs\\detect_SVM_NV_demo_center\\exp_3_fps481\\9\\9"
def writeVideo(filePath):
    img_array = []
    size = (302,1080)
    names = ['cow']
    
    
    vid_name = os.path.basename(os.path.normpath(filePath))
    vid_path = str(Path(filePath + "/" + vid_name ).with_suffix('.mp4'))
    id,img_locations,*xyxys = CALCULATE_MAX_CATTLE_ID(filePath+"/"+vid_name+".csv")
    #print(xyxys)
    
    out = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 6, size)
    if len(img_locations)<10: #skip if less than 6 photos
        return -1
    
    for ind in range(len(img_locations)):
        #x,y,w,h = cv2.boundingRect(contour)
        #x1,y1,x2,y2 = xyxys[0][ind][1], xyxys[ind][1], xyxys[ind][2], xyxys[ind][3]
        #print(x1,y1,x2,y2)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 1)
        #image = cv2.rectangle(img_array[ind],(x1,y1),(x2,y2) , (36,255,12),1)
        #cv2.putText(image, str(id), (xyxys[0][0], xyxys[ind][1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        #print(xyxys[0][ind])
        
        
        img = cv2.imread(img_locations[ind])
        annotator = Annotator(img, line_width=8, example=str(names))
        #print('ind', ind)
        #print(xyxys[0][ind])
        try:
            annotator.box_label(xyxys[0][ind],str(id), color=(15, 0, 255))
            annotated_img =cv2.resize(annotator.result(),size) 
            #cv2.imshow('new cow',img)
            #if cv2.waitKey(1) == ord('a'):  # q to quit
            #    raise StopIteration
            out.write(annotated_img)
        except:
            continue
    out.release()
    img_array=[]
    print("done ", vid_name)
    cv2.destroyAllWindows()
    return id
#writeVideo(default)
#%%python --source "D:\Python\env\Lameness\Frames\Videos\20220201_145508_7108.mp4"  --yolo-weights weights_slm/best_6_23_gpu.pt --view-img --save-crop --device 0


# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU

"""

import gc

gc.collect()

torch.cuda.empty_cache()

FILE = Path("__file__").resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

X1=240
X2=400
Y1=94
Y2=590
SIZE =224


default=640
save_video=True

file_location="D:\\815_CowDataChecking\\20230130\\13\\20230130_135955_8267_ACCC8EEE85E1\\20230130_15\\" #\\20221228_055019_5DDC.mkv"#20221230_155051_3DBF.mkv"#20221228_055019_5DDC.mkv"
#file_location="D:\\815_CowDataChecking\\20230110\\20221228_M\\20221228_055019_5DDC.mkv" #\\20221228_055019_5DDC.mkv"#20221230_155051_3DBF.mkv"#20221228_055019_5DDC.mkv"

SKIP_VIDEOS=False # True False toggle here to have skip videos
NUMBER_SKIP_VIDEOS = 5
#file_location = "\\172.16.4.111\\Public\訓子府L5G_2020\\生データ_original_data\\360カメラ\\A\\20221115\\13\\20221115_135954_9B55_ACCC8EEE85E1\\20221115_15\\20221115_151030_8349.mkv"


#file_location="D:\\815_CowDataChecking\\20220906\\360\A\\20220906\\13\\20220906_135955_2249_ACCC8EEE85E1\\20220906_16\\20220906_161102_E42D.mkv"
#file_location = "C:\\Users\\thithilab\\Desktop\\Cow Data (22~28)\\20220722\\all\\20220722_152539_53E5.mkv"
#filename="20220705_135955_4D30"
#file_path = "D:\\815_CowDataChecking\\20220704\\13\\20220704_135955_D85D_ACCC8EEE85E1\\20220704_E_All\\"
#deep_test ='C:/Users/thithilab/Desktop/20220705/m_videos_5_7/DEEP1/DEEP2'
#multifile = 'D:/CheckFrame/14B8/20220704_145523_14B8.mkv,C:/Users/thithilab/Desktop/20220705/m_videos_5_7/20220705_053512_A9B0.mkv'
#Y1_NEW=110
#Y2_NEW=530
Y1_NEW=110 #120  #decrease here to extend, increase to shrink 
Y2_NEW=530  #500  # redyce here to extend , increase to do vice casa 460 previous

Y1_PRECISE=100
Y2_PRECISE=400  #where cow is most precise  August 7 2022
HAS_COW=False  # to save video when has cow

cow_order=[]
cow_count = []
cow_label=[]
frame_rate=3
has_seen_cattle = False 
prev_label_store=[None] * 5
prev_cow_position=[None] * 5


all_detected_cow=[]

local_id=1


#for max apperance cattle id 
final_result = []
final_percentage = []
final_total = []
current_cow = []
excel_cow_count=[]
#end

#demo video write
BATCH = 100
BATCH_COUNT = 1
PREV_BATCH = 0
LAST_SEEN = time.time()
FIRST_SEEN = True
demo_img_save_path = []

prevId_record =[]
MAX_prevId = [] 
MAX_xyxy1 = [] 
MAX_xyxy2 = [] 
MAX_xyxy3 = [] 
MAX_xyxy4 = [] 
MAX_orgId = []
IMAGE_STORED_LOCATION = []
#end

#region Cattle Tracking
STORED_IDS= []
STORED_MID_Y = []
STORED_MID_Y1 = []
STORED_MID_Y2 = []
STORED_MISS = []
PREVIOUS_ID = [] # keep the record of last seen ids and position
PREVIOUS_Y1 = [] 
PREVIOUS_Y2 = [] 
PREVIOUS_LOCAL_IDS = []
CATTLE_LOCAL_ID= 0
IS_FIRST_CATTLE = True
#end



def DoROI(image):
    h,w,c = image.shape
    img_arr = np.array(image)
    img_arr[0 : int(94*(h/default)), 0 : h] = (0, 0, 0)   #top
    img_arr[0 : h, 0 : int(240*(w/default))] = (0, 0, 0)   #left
    img_arr[0 : h, int(400*(w/default)) : w] = (0, 0, 0)   #right
    img_arr[int(590*(h/default)) : h,0 : w] = (0, 0, 0)   #bottom
    return img_arr

def Demo_DoROI(image):
    h,w,c = image.shape
    img_arr = np.array(image)
    #img_arr[0 : int(94*(h/default)), 0 : h] = (0, 0, 0)   #top
    img_arr[0 : h, 0 : int(230*(w/default))] = (0, 0, 0)   #left
    img_arr[0 : h, int(410*(w/default)) : w] = (0, 0, 0)   #right
    #img_arr[int(590*(h/default)) : h,0 : w] = (0, 0, 0)   #bottom
    return img_arr


def DoROI_640(image):
    img_arr = np.array(image)
    img_arr[0 : 94, 0 : 640] = (0, 0, 0)   #top
    img_arr[0 : 640, 0 : 240] = (0, 0, 0)   #left
    img_arr[0 : 640, 400 : 640] = (0, 0, 0)   #right
    img_arr[590 : 640,0 : 640] = (0, 0, 0)   #bottom
    return img_arr
  
def check_withinROI(x1,y1,x2,y2,h,w):
    if(x1<int(X1*(w/default)) or x2>int(X2*(w/default)) or y1<int(Y1*(h/default)) or y2>int(Y2*(h/default)) or x1>=int(X2*(w/default))):
      return False
    return True  

def check_withinROI_NEW(x1,y1,x2,y2,h,w):
    if(x1<int(X1*(w/default)) or x2>int(X2*(w/default)) or y1<int(Y1_NEW*(h/default)) or y2>int(Y2_NEW*(h/default)) or x1>=int(X2*(w/default))):
        return False
    if(y2 - y1>1300 or y2-y1<800):
        return False
    return True  

def check_withinROI_PRECISE(x1,y1,x2,y2,h,w):
    if(x1<int(X1*(w/default)) or x2>int(X2*(w/default)) or y1<int(Y1_PRECISE*(h/default)) or y2>int(Y2_PRECISE*(h/default)) or x1>=int(X2*(w/default))):
      return False
    return True  

def check_cow_Count(label):
    global cow_label
    global cow_count
    print("inserting label")
    if label in cow_label: #check exist
        cow_count[cow_label.index(label)]+=1  #start counting of the newly inserted cow
        
    else:
        cow_label.append(label)  # if not exist then add the cow label to array
        cow_count.append(1)  #start counting of the newly inserted cow


def determine_label(img):
    
    #if Isolation_Forest(img) != 1:
    #    res = ['unknown']
    #    check_cow_Count(res[0])
    #    return res
    global all_detected_cow
    label = Predict_SVM(img)
    HAS_COW=True
    check_cow_Count(label[0])
    all_detected_cow.append(label[0])
    return label

#label for cow label, y for y2 postion of cow, h for total height if image, position for 1st cow of the frame, 2nd cow of the frame etc,...
def take_first_appear_lable(label,y,h,nth_cows):
    
    global prev_label_store
    global prev_cow_position
    

    #prev_label_length = len(prev_label_store)
    #prev_position_length = len(prev_label_store)
    #print(prev_label_length)
    
    #first
    print("cow position :"+str(nth_cows))
    print("label "+label)
    if(prev_label_store[nth_cows]==None and prev_label_store[nth_cows+1]==None):
        prev_label_store[nth_cows]=label
        prev_cow_position[nth_cows]=y
        res = [label]
        return res
        
    if(prev_label_store[nth_cows]!=None):
        if(y<prev_cow_position[nth_cows]+35) : #check if prev_cow
            prev_cow_position[nth_cows]=y
            res = [prev_label_store[nth_cows] ]
            return res
        elif(prev_cow_position[nth_cows+1]!=None and y<prev_cow_position[nth_cows+1]+35) : #check if prev_cow second cow 
            #2nd one become 1st cow
            prev_cow_position[nth_cows]=None
            prev_label_store[nth_cows]=None
            
            prev_label_store = deque(prev_label_store)
            prev_label_store(1)
            prev_label_store = list(prev_label_store)
            
            
            prev_cow_position = deque(prev_cow_position)
            prev_cow_position(1)
            prev_cow_position = list(prev_cow_position)
            
            
            prev_cow_position[nth_cows]=y
            res = [prev_label_store[nth_cows]]
            return res  #move 2nd index to first index
        elif(prev_cow_position[nth_cows+1] == None) : #new cows in first place
            prev_cow_position[nth_cows]=y
            prev_label_store[nth_cows] = label
            res = [label]
            return res
           
    res = [label]
    return res
    
    
    
prev_labels=[]  #keep last records to compare y pixel value    
prev_y1s=[]    


def compare_with_prev_cow(label,y,h):
    prev_labels.append(label)
    prev_y1s.append(y)
    has_100_record = len(prev_labels)
    start = 0
    end = 0
    ceiling = h-int(h*(Y1_NEW/default))
    #print(ceiling )
    #print(h)
    #print(y)
    if(y+100>=ceiling) :   #checking if the image reach the top
        if has_100_record>=20:
            start=has_100_record - 20 - 1 #only check last 20 values
            end = has_100_record - 1
        cow_count_c=[]
        cow_label_c=[]
        prev_y_value=y
        total_frames=0
        global cow_order
        for i in range(end,start,-1):
            if(prev_y1s[i]>=h/2 +50 ):  #check only for half of screen
                #for l in range(len(label[i].split(',')):
                #split_label = label[i].split(',')[l]
                #if split_label in cow_label: #check exist
                if(prev_y1s[i]>prev_y_value):
                    prev_y_value=prev_y1s[i] #go with 30 pixel different
                    total_frames += 1
                    if prev_labels[i] in cow_label_c:
                        cow_count_c[cow_label_c.index(prev_labels[i])]+=1  #start counting of the newly inserted cow
        
                    else:
                        cow_label_c.append(prev_labels[i])  # if not exist then add the cow label to array
                        cow_count_c.append(1)  #start counting of the newly inserted cow
                #else:
                    
                
        #prediction_RF = np.argmax(prop)         
        #get max cow id
        if(len(cow_count_c)<1):
            return None
        max_count = max(cow_count_c)
        threshold_50_percent = math.floor(total_frames*0.5)
        if(max_count>threshold_50_percent + 1):
            index = np.argmax(cow_count_c)
            cow_order.append(cow_label_c[index])
            #print(" cow label "+str(cow_label_c[index]))
            return cow_label_c[index]
        else:
            #print(" cow label unknown")
            cow_order.append("unknown")
            return "unknown"
    
    if(len(prev_y1s) >700): # delete first 500 when greater than 800
        del prev_labels[:500]
        del prev_y1s[:500]    
         


@torch.no_grad()
def run(
        #weights=ROOT / 'Sept_no_alien_weight_v1/best.pt',  # model.pt path(s)  #july_weight
        #weights=ROOT / 'September_bounding_flip_800/best.pt',
        #weights=ROOT / 'weights/Dec_new_v1/best.pt',
        weights=ROOT / 'weights/Jan_2023_Weight_v3/best.pt',  #v3
        source=ROOT / file_location,  #file_location,  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.2,  # confidence threshold
        iou_thres=0.001,  #NS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3 #None
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect_SVM_NV_demo_center',  # save results to project/name
        name='exp_'+str(frame_rate)+'_fps',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=8,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=True,  # use FP16 half-precision inference #True
        dnn=False,  # use OpenCV DNN for ONNX inference
    
):
    
    global all_detected_cow
    global frame_rate
    #added
    sec=0
    global cow_lable
    global cow_count
    global cow_order
    global FIRST_SEEN
    global BATCH
    global BATCH_COUNT
    global PREV_BATCH
    global LAST_SEEN
    global demo_img_save_path
    global has_seen_cattle
    cow_id = []
    cow_id_original =[]
    cow_top = []
    cow_left = []
    cow_width = []
    cow_height = []
    cow_score = []
    cow_frame = []
    
    manual_summarize_ids = []
    manual_local_ids = []
    manual_id = 1
    
    read_after_frame = 1
    manual_cow_count = 1
    
    global prevId_record
    global MAX_prevId
    global MAX_xyxy1
    global MAX_xyxy2
    global MAX_xyxy3
    global MAX_xyxy4
    global MAX_orgId

    global SKIP_VIDEOS
    global NUMBER_SKIP_VIDEOS
    
    global IMAGE_STORED_LOCATION
    
    cf = 0  
    count=0
    
    source = str(source)
    #vid_path = []
    #vid_path.append("D:\\CheckFrame\\14B8")
    #vid_path.append(source)
    #source = vid_path
    #save_img = not nosave and not source.endswith('.txt')  # save inference images
    #added
    save_img=True
    
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #print(save_dir)
    csv_save_dir = str(save_dir)
    #print(csv_save_dir)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    
    
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    #print(names)
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam and False:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    demo_vid_path ,demo_vid_writer = [] ,[None]  * 12
    
    demo_vid_save_path = str(save_dir)

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    #frame_rate = 4    #frame rate here
    prev = 0
    prev_frame = 0
    current_vid_name = ''
    #print(classes)
    SKIPPING = False
    SKIPPED_COUNT = 0
    for path, im, im0s, vid_cap, s in dataset:
        if SKIP_VIDEOS :
            if current_vid_name!=path :  # check if moring
                SKIPPED_COUNT +=1
                current_vid_name = path
                #print('skipped ', SKIPPED_COUNT, ' video(s)')
            if SKIPPED_COUNT <= NUMBER_SKIP_VIDEOS :

                continue
        #cv2.waitKey(1000) #1 fps   1000/ value =fps
        #vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
        #vid_cap.set(cv2.CV_CAP_PROP_FPS, 1)
        #vid_cap.set(cv2.CAP_PROP_FPS, 1)
        
        
        
        HAS_COW=False
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        
        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        #time_elapsed = time.time() - prev
    

        #if time_elapsed > 1/frame_rate:
            #prev = time.time()
            #print("Greater")
            #print(prev)
        #else:
            #print("break")
            #
        
        if (read_after_frame - prev_frame == 0 and not has_seen_cattle):
            prev_frame=0
            continue
        prev_frame +=1
        for i, det in enumerate(pred):  # per image
           
            #det = det.sort(key=lambda row: (row[1]))
            #print(time_elapsed)
            
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            #added ROI    
            #h,w,c=im0.shape
            
            #resize
            #if(w>640 or h>640):
            #  im0=imutils.resize(im0, width = 640)
            
            
            
            #check containing frame here
            
            
            
            
            
            #end of checking containing frame
            
            

            #ROI
            im0=Demo_DoROI(im0)
            h,w,c=im0.shape
            cropped_img = im0.copy()[0 : h,int(230*(w/default)):int(410*(w/default))]
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                #det.sort(key=lambda row: (row[1][0]))
                #print(det)
                det, b = torch.sort(det, dim=0)
                #print('sorted',det)
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                
                
                # Write results
                cow_position = 0
                counter = 0
                has_seen_cattle = False
                for *xyxy, conf , cls in (det):#reversed(det):
                    if(check_withinROI_NEW(xyxy[0],xyxy[1],xyxy[2],xyxy[3],h,w)):
                      #print(cls)
                      count+=1
                      has_seen_cattle = True
                        
                      
                      box_left = xyxy[0]
                      box_top = xyxy[1]
                      box_w = xyxy[2] - xyxy[0]
                      box_h = xyxy[3] - xyxy[1]
                      
                      #cow_left.append(box_left)
                      #cow_top.append(box_top)
                      #cow_width.append(box_w)
                      #cow_height.append(box_h)
                      #cow_score.append(conf)
                      #cow_frame.append(seen)
                     
                      #feed on cnn and get label
                      #save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}_{count}.jpg', BGR=True)
                      
                      #crop
                      BGR=False
                      #print("step-3-y")
                      crop = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                      #frame_crop = im0[0 : h,int(200*(w/default)):int(540*(w/default))] 
                      

                      #cropped = torch.tensor(crop, device = 'cpu')
                      #image = Image.fromarray(crop)
                      #img = image.resize((128, 128), Image.ANTIALIAS)
                      
                    
                      #do some process like testing data in cnn
                      img = cv2.resize(crop, (SIZE, SIZE))
                      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                      #cv2.imshow('detected cow',img)
                      if cv2.waitKey(1) == ord('a'):  # q to quit
                          raise StopIteration
                      #crop=imutils.resize(crop, width = 224)
                      img=img / 255.0
                      label = Predict_SVM(img)
                      #label = determine_label(img)
                      #isknown = isKnownCattle(img) #for unknown
                      #print(isknown)
                      HAS_COW=True
                      #prev_id = Take_Prev_Label(int(xyxy[3]),label,cow_position)
                      prev_id = Take_Prev_Label(box_top,box_h,label,cow_position)
                      if(prev_id==-1): #skip cattle when prev_id // filter id is -1
                          if(count==1):
                            has_seen_cattle=False
                          count-=1
                          continue
                      print(prev_id)
                      cow_position+=1
                      #label = Predict_SVM_test_pro(img)
                      cow_id.append(prev_id[0])
                      cow_id_original.append(int(label[0]))
                      #check cow count here
                      h,w,c=im0.shape  
                      BATCH_COUNT = prev_id[0] # skip batch count here  
                      #BATCH calculator
                      if(FIRST_SEEN):
                        LAST_SEEN = time.time() #first seen time
                        FIRST_SEEN=False
                
                      if(time.time()-LAST_SEEN>=300): # 3 mins different
                        #write excel for each cattle
                        # print(len(prevId_record), ' previd_record', prevId_record)
                        for csv_index in range(len(prevId_record)):
                            df = pd.DataFrame(MAX_prevId[csv_index], columns = ['ID'])
                            try:
                                org_ids = torch.tensor(MAX_orgId[csv_index], device = 'cpu')
                                df["Original"] = org_ids
                            except:
                                 df["Original"] = MAX_orgId[csv_index]
                                    
                                    
                            try:
                                stored_locations = torch.tensor(IMAGE_STORED_LOCATION[csv_index],device = 'cpu')
                                df["location"] = stored_locations
                            except:
                                df["location"]=IMAGE_STORED_LOCATION[csv_index]
                                
                            df["xyxy1"] = MAX_xyxy1[csv_index]
                            df["xyxy2"] = MAX_xyxy2[csv_index]
                            df["xyxy3"] = MAX_xyxy3[csv_index]
                            df["xyxy4"] = MAX_xyxy4[csv_index]
                            
                            

                            now=str(datetime.now().date())
                            
                            save_csv_each_path = str(Path(save_dir / str(prevId_record[csv_index]) / str(prevId_record[csv_index]) / f'{str(prevId_record[csv_index])}.csv'))
                            #print(save_csv_each_path)
                            
                            df.to_csv(save_csv_each_path, index= False) ##
                            MAX_xyxy1[csv_index]=[]
                            MAX_xyxy2[csv_index]=[]
                            MAX_xyxy3[csv_index]=[]
                            MAX_xyxy4[csv_index]=[]
                            MAX_orgId[csv_index]=[]
                            MAX_prevId[csv_index]=[]
                            IMAGE_STORED_LOCATION[csv_index]=[]
                            
                        #print("new batch")
                        prevId_record = []
                        MAX_prevId = []
                        MAX_xyxy1 = [] 
                        MAX_xyxy2 = [] 
                        MAX_xyxy3 = [] 
                        MAX_xyxy4 = [] 
                        MAX_orgId = [] 
                        IMAGE_STORED_LOCATION = []
                  
                        
                        cattle_ids = []
                        #print(len(det))
                        
                  
                        #release video write and reset vid_path
                        
                        #for index in range(len(demo_vid_path)):
                            
                        #    if isinstance(demo_vid_writer[index], cv2.VideoWriter):
                        #        demo_vid_writer[index].release()  # release previous video writer
                        #        print('removed video write ', demo_vid_path[index])
                        
                        #demo_vid_path = []
                        #demo_img_save_path = []
                        #end
                        
                      LAST_SEEN = time.time()
                      
                      #final_label = compare_with_prev_cow(label[0],int(xyxy[3]),h)
                      #label = take_first_appear_lable(label[0],int(xyxy[3]),h,cow_position) #remove
                      #print(im0.shape)
                      #if(isknown[0] == -1): #open when doing unknonw
                      #  label = ['unknown']
                      #print(label)
                      #if final_label != None: print("final label "+ final_label) 
                      annotator.box_label(xyxy,prev_id[0], color=(15, 0, 255))#color=colors(c, True))  # change back to prev_id 
                      #if save_txt:  # Write to file
                      #    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                      #    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                      #    with open(f'{txt_path}.txt', 'a') as f:
                      #        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                      #if save_img or save_crop or view_img:  # Add bbox to image
                      #    c = int(cls)  # integer class
                      #    #label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  #original
                      #    annotator.box_label(xyxy, label, color=colors(c, True))
                      
                      if save_crop:
                           save_one_box(xyxy, imc, file=save_dir /  str(BATCH_COUNT)  / prev_id[0]  / 'cropped' / f'{p.stem}.jpg', BGR=True)
                      # change by cattle id here
                      #demo_vid_index= 0
                      #demo_path = str(Path(str(save_dir)+"/"+str(BATCH_COUNT)+"/"+prev_id[0]).with_suffix('.mp4'))
                      
                      #save_one_box(xyxy, im0, file=save_dir / str(BATCH_COUNT) / prev_id[0]  / 'cropped' / f'{p.stem}.jpg', BGR=True)
                      #annotated_img = annotator.result()
                      
                      #fps, fw, fh = 6, annotated_img.shape[1], annotated_img.shape[0] 
                      #print('width ',fw,' height ',fh)
                      base_path = str(Path(save_dir / str(BATCH_COUNT) / prev_id[0]))
                      demo_annotated_img_save_path = Path(base_path+ '/' + f'{p.stem}_{str(manual_cow_count).zfill(4)}.jpg')
                      #print(demo_annotated_img_save_path)
                      #save_one_box(xyxy, imc, file = base_path / prev_id[0]  / f'{p.stem}.jpg', BGR=True)
                      cv2.imwrite(demo_annotated_img_save_path, cropped_img)
                      no_tensor_xyxy =[int(xyxy[0]*230/(2*default)),int(xyxy[1]),int(xyxy[2]*410/(2*default)),int(xyxy[3])] #to get the crop size
                      #print(no_tensor_xyxy)
                      try:
                        index_prevId = prevId_record.index(int(prev_id[0]))
                        #print(index_prevId)
                        MAX_prevId[index_prevId].append(int(prev_id[0]))#,int(label[0]),xyxy)
                        MAX_xyxy1[index_prevId].append(int(no_tensor_xyxy[0]))
                        MAX_xyxy2[index_prevId].append(int(no_tensor_xyxy[1]))
                        MAX_xyxy3[index_prevId].append(int(no_tensor_xyxy[2]))
                        MAX_xyxy4[index_prevId].append(int(no_tensor_xyxy[3]))
                        MAX_orgId[index_prevId].append(int(label[0]))
                        
                        IMAGE_STORED_LOCATION[index_prevId].append(demo_annotated_img_save_path)
                        
                      except :
                        prevId_record.append(int(prev_id[0]))
                        # print(len(prevId_record)-1, 'prevID_record ', len(MAX_prevId) , 'max_previd' )

                        #MAX_prevId[len(prevId_record)-1].append(int(prev_id[0]))#,int(label[0]),xyxy)
                        #MAX_xyxy[len(MAX_prevId)-1].append(xyxy)
                        #MAX_orgId[len(MAX_prevId)-1].append(int(label[0]))
                        MAX_prevId.append([int(prev_id[0])])#,int(label[0]),xyxy)
                        
                        MAX_xyxy1.append([int(no_tensor_xyxy[0])])
                        MAX_xyxy2.append([int(no_tensor_xyxy[1])])
                        MAX_xyxy3.append([int(no_tensor_xyxy[2])])
                        MAX_xyxy4.append([int(no_tensor_xyxy[3])])
                        MAX_orgId.append([int(label[0])])
                        IMAGE_STORED_LOCATION.append([demo_annotated_img_save_path])
                    
                      try:
                        #demo_vid_index = demo_vid_path.index(demo_path)
                        demo_vid_index = demo_img_save_path.index(base_path)
                        
                        #print("path exist")
                      except:
                        manual_summarize_ids.append(int(prev_id[0]))
                        manual_local_ids.append(manual_id)
                        manual_id +=1
                        
                        
                        #demo_vid_path.append(demo_path)
                        #print(base_path)
                        #print('vid path is new ')
                        demo_img_save_path.append(base_path)
                        #demo_vid_index = len(demo_vid_path) -1
                        #demo_vid_writer[demo_vid_index]=(cv2.VideoWriter(demo_vid_path[demo_vid_index], cv2.VideoWriter_fourcc(*'mp4v'),6, (fw, fh)))
                        
                     
                      write_demo_vide=False
                      if write_demo_vide :  
                        
                        
                        #print('vid index is ', demo_vid_index, ' location is ', demo_vid_path[demo_vid_index])
                        #print(annotated_img.shape)
                        if isinstance(demo_vid_writer[demo_vid_index], cv2.VideoWriter):
                            demo_vid_writer[demo_vid_index].write(annotated_img)
                      
                      manual_cow_count +=1


            # Stream results
            im0 = cv2.resize(annotator.result(), (1080, 1080))
            if view_img or True:
                
                if(w>1080 or h>1080):
                    cv2.imshow('detected cows', imutils.resize(im0, width = 1080,height=720))
                else:
                    cv2.imshow('detected cows',im0)
                if cv2.waitKey(1) == ord('a'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if (save_img or save_video) and HAS_COW:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else :  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = 1080#int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = 1080#int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            #fps=frame_rate * 2
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            #fps=frame_rate
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        all_detected_cow.append('xxxxxxxxxxxxx')
                        all_detected_cow.append('xxxxxxxxxxxxx')
                        all_detected_cow.append(save_path)
                        
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    cv2.destroyAllWindows()  
    
    #region release remaining video write
    
    #for index in range(len(demo_vid_path)):
                        
    #    if isinstance(demo_vid_writer[index], cv2.VideoWriter):
    #        demo_vid_writer[index].release()  # release previous video writer
    #        print('removed video write ', demo_vid_path[index])
            
    #demo_vid_path = []
    
    ##cmtbyslm

    
        
    df = pd.DataFrame(cow_id, columns = ["ID"])
    try:
        original_ids = torch.tensor(cow_id_original, device = 'cpu')
        df["Original"] = original_ids
    except:
         df["Original"] = cow_id_original
    
    
    
    now=str(datetime.now().date())
    try: #ohh is it me?
        print(all_detected_cow)
        #all_detected_cow = torch.tensor(all_detected_cow,device="cpu")
        detected_cow_df = pd.DataFrame(all_detected_cow, columns = ['ID'])
        detected_cow_df.to_csv('csv/all_detected_cow_'+str(frame_rate)+'_fps_'+now+'.csv')  
        print('result saved to all_detected_cow_'+str(frame_rate)+'_fps_'+now+'.csv')
    except :
        print ("couldn't save all_detected_cow")
    
    path_to_csv = csv_save_dir+'/detected_cow_vggSVM_'+now+'.csv'
    df.to_csv(path_to_csv, index= False) 
    
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    
    Generate_Cattle_Id_By_Apperance(path_to_csv,csv_save_dir)
    
    ########### region save csv for each cattle

    for csv_index in range(len(prevId_record)):
        df = pd.DataFrame(MAX_prevId[csv_index] , columns = ['ID'])
        try:
            org_ids = torch.tensor(MAX_orgId[csv_index], device = 'cpu')
            df["Original"] = org_ids
        except:
            df["Original"] = MAX_orgId[csv_index]
        
        try:
            stored_locations = torch.tensor(IMAGE_STORED_LOCATION[csv_index],device = 'cpu')
            df["location"] = stored_locations
        except:
            df["location"]=IMAGE_STORED_LOCATION[csv_index]
        df["xyxy1"] = MAX_xyxy1[csv_index]
        df["xyxy2"] = MAX_xyxy2[csv_index]
        df["xyxy3"] = MAX_xyxy3[csv_index]
        df["xyxy4"] = MAX_xyxy4[csv_index]


        now=str(datetime.now().date())
                            
        save_csv_each_path = str(Path(save_dir / str(prevId_record[csv_index]) / str(prevId_record[csv_index]) / f'{str(prevId_record[csv_index])}.csv'))
        df.to_csv(save_csv_each_path, index= False)##asdfasdf
    prevId_record = []
    MAX_prevId = []
    MAX_xyxy1 = [] 
    MAX_xyxy2 = [] 
    MAX_xyxy3 = [] 
    MAX_xyxy4 = [] 
    MAX_orgId = [] 
    IMAGE_STORED_LOCATION = []
                                      
    cattle_ids = []
    #################################\
    manual_summarize_ids = []
    manual_local_ids = []
    #### write video after saving csv
    final_cattle_count = 1
    for loc in range(len(demo_img_save_path)):
        print(demo_img_save_path[loc])
        final_cattle_id = writeVideo(demo_img_save_path[loc])
        if(final_cattle_id != -1):
            manual_local_ids.append(final_cattle_count)
            manual_summarize_ids.append(final_cattle_id)
            final_cattle_count+=1 
    
    summarize_id_csv = pd.DataFrame(manual_local_ids, columns = ["Local Id"])
    try:
        manual_summarize_ids = torch.tensor(manual_summarize_ids, device = 'cpu')
        summarize_id_csv["Cow Id"] = manual_summarize_ids
    except:
         summarize_id_csv["Cow Id"] = manual_summarize_ids
            
    summarize_id_csv.to_csv(csv_save_dir+'/summarize_id_'+now+'.csv', index= False) 
    
    
    df.to_csv(csv_save_dir+'/detected_cow_vggSVM_'+now+'.csv', index= False) 
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
        
def parse_opt():
    class Args:
        #weights='September_bounding_flip_800\best.pt' # model.pt path(s) where is weight?
        weights=ROOT / 'weights/Jan_2023_Weight_v3/best.pt' #v3
        #source= "C:\\Users\\thithilab\\Desktop\\file\\New Data\\14\\first32\\20220310_152525_E1E0.mkv" # file/dir/URL/glob, 0 for webcam  //change your video path here
        source= file_location # file/dir/URL/glob, 0 for webcam  //change your video path here
        data='data/coco128.yaml'  # dataset.yaml path
        imgsz=(640, 640)  # inference size (height, width)
        conf_thres=0.2 # confidence threshold
        iou_thres=0.005  # NMS IOU threshold 0.45
        max_det=4 # maximum detections per image # prev 1000
        device='0'  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True  # show results
        save_txt=False  # save results to *.txt
        save_conf=False  # save confidences in --save-txt labels
        save_crop=True  # save cropped prediction boxes
        nosave=False  # do not save images/videos
        classes=None  # filter by class: --class 0, or --class 0 2 3 #None is prev value
        agnostic_nms=False  # class-agnostic NMS
        augment=False  # augmented inference
        visualize=False  # visualize features
        update=False  # update all models
        project='runs/detect_SVM_NV_demo_center'  # save results to project/name
        name='exp'  # save results to project/name
        exist_ok=False  # existing project/name ok, do not increment
        line_thickness=8  # bounding box thickness (pixels)
        hide_labels=False  # hide labels
        hide_conf=False  # hide confidences
        half=True  # use FP16 half-precision inference #False
        dnn=False  # use OpenCV DNN for ONNX inference

    return Args()
     
   #parser here


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))
    #run()

#__name__=="__main__"
if __name__ == "__main__":

    frame_rate=3
    opt = parse_opt()
    t = Timer()
    t.start() # timer start 
    main(opt)
    t.stop()  # A few seconds later=
