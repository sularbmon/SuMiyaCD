import torch
import gc

from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import os
import time
from ultralytics.yolo.utils.files import increment_path
import torch
import gc
import pickle
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
#from pathlib import Path
gc.collect()
torch.cuda.empty_cache()

from ultralytics.yolo.utils.plotting import Annotator

from ultralytics import YOLO


gc.collect()

torch.cuda.empty_cache()


################################

#NAS_path = sys.argv[]
#DATE = sys.argv[1]
SAVE_PATH = sys.argv[2]



## check DATE input
#if len(DATE) == 10:
#    save_DATE = str(DATE)
#    read_DATE = f"{DATE[:4]}{DATE[5:7]}{DATE[8:]}"
#elif len(DATE) == 8:
#    save_DATE = f"{DATE[:4]}_{DATE[4:6]}_{DATE[6:]}"
#    read_DATE = str(DATE)

#SAVE_CSV_PATH = Path(SAVE_PATH+"/Honkawa/360/"+save_DATE+"/")
#SAVE_VID_PATH = Path(SAVE_PATH+"/Honkawa/360_bb/"+save_DATE+"/")

#(SAVE_CSV_PATH).mkdir(parents=True, exist_ok=True)
#(SAVE_VID_PATH).mkdir(parents=True, exist_ok=True)

path_to_process= []
video_path2 = ''

dataset = sys.argv[1]
print(dataset)
#path_to_process.append(dataset) 



#load model
def get_bundle_dir():
    if getattr(sys, 'frozen', False):
        bundle_dir = sys._MEIPASS
    else:
        bundle_dir = os.path.dirname(os.path.abspath(__file__))
    return bundle_dir

bundle_dir = get_bundle_dir()
predictor_filename = os.path.join(bundle_dir, 'Honkawa_models', 'VGG_SVM_HONKAWA_1103_v1.pkl')
le_filename = os.path.join(bundle_dir, 'Honkawa_models', 'GENERAL_LE_HONKAWA_1103.le')
model = os.path.join(bundle_dir, 'Honkawa_models', 'HONKAWA_WEIGHT.pt')
#data_file_location = os.path.join(bundle_dir, 'Honkawa_models', 'coco128.yaml')

#clf = pickle.load(open(svm_filename, 'rb'))
#le = pickle.load(open(le_filename, 'rb'))
#VGG_model =  tf.keras.models.load_model(vgg_filename, compile = False)
##############




###################################
predictor = pickle.load(open(predictor_filename, 'rb'))
lable_encoder = pickle.load(open(le_filename, 'rb'))
###################################

####################################
SIZE=112

vgg = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in vgg.layers:
	layer.trainable = False
    
#vgg.summary()  #Trainable parameters will be 0
######################################

####################################
def Predict_SVM(image):  ## new with vgg
    
    input_img = np.expand_dims(image, axis=0) #Expand dims so the input is (num images, x, y, c)
    input_img_feature=vgg.predict(input_img)
    input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
    predicted_result = predictor.predict(input_img_features)[0] 
    predicted_result = lable_encoder.inverse_transform([predicted_result])  #Reverse the label encoder to original name
    
    return predicted_result
print("defined predictor")

############################################

#######################################

###### load dataset and model


def check_withinROI_NEW(x1,y1,x2,y2,h,w):
    #print(x1, '  ',y1, '  ',x2, '  ',y2, '  ',h, '  ',w)
    #if(x1<int(X1*(w/default)) or x2>int(X2*(w/default)) or y1<int(Y1_NEW*(h/default)) or y2>int(Y2_NEW*(h/default)) or x1>=int(X2*(w/default))):
    #    return False
    
    if(x1<150 or x2>1750):
        return False
    if(y2-y1<600 or x2-x1 <250): #1400 to 700 Before
        return False
    return True  

def check_withinROI_Resize(x1,y1,x2,y2,h,w):
    #print(x1, '  ',y1, '  ',x2, '  ',y2, '  ',h, '  ',w)
    resize_x1=850#*(w/2992)
    resize_x2=1050#*(w/2992)
    resize_y1=Y1_NEW#Y1_NEW*(w/2992)
    resize_y2=Y1_NEW#Y2_NEW*(w/2992)
    #print(resize_x1, '  ',resize_y1, '  ',resize_x2, '  ',resize_y2)
    if(x1<int(resize_x1) or x2>int(resize_x2) or x1>=int(resize_x2)):
        return False
    if(y2 - y1>1400 or y2-y1<700): #1400 to 700 Before
        return False
    return True  
###########################################



###XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX##

def Is_Duplicate_Id(y1,y2,id):
    global PREVIOUS_ID
    global PREVIOUS_Y1
    global PREVIOUS_Y2
    global PREVIOUS_LOCAL_IDS
    global CATTLE_LOCAL_ID
    
    try: 
        index = PREVIOUS_ID.index(id)
        print('I reached here')
        if(PREVIOUS_Y1[index]+321<=y1 and PREVIOUS_Y2[index]+371<y2): #duplicate from bottom
         #   if(id in PREVIOUS_LOCAL_IDS):
         #   #print('id: ',id,' LOCAL_ID: ',CATTLE_LOCAL_ID)
         #       return PREVIOUS_LOCAL_IDS[id][0]
            
            #print('except')
            PREVIOUS_ID.append(CATTLE_LOCAL_ID)
            PREVIOUS_Y1.append(y1)
            PREVIOUS_Y2.append(y2)
            #print('New Cattle Id')
            CATTLE_LOCAL_ID+=1
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
    
###XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#######

####XXXXXXXXXXXXXXXXXXXXXXXXXXX################


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
        last_y1 = STORED_MID_Y1[len(STORED_MID_Y1)-1]#max(STORED_MID_Y1)
        last_y2 = STORED_MID_Y2[len(STORED_MID_Y2)-1]#max(STORED_MID_Y2)
        MISSED_LEN = len(STORED_MISS)
        #if(IS_NEW):
        
        #    MISSED_LEN -=1
        removed = 0
        for i in range(MISSED_LEN):
            #print(i, ' missed index checking' )
            missed = STORED_MISS[i-removed]
            #print('checking ',i-removed, 'to remove')
            if(missed>70): #if missed 35 frames
    
                del STORED_MISS[i-removed]  
                del STORED_MID_Y[i-removed]
                del STORED_MID_Y1[i-removed]
                del STORED_MID_Y2[i-removed]
                del STORED_IDS[i-removed]
                removed+=1
                #print('removed')
                
    #clear misses
   
    
    threshold_1 = 100 #300
    threshold_2 = 100  #230
    Distance = 5
     
    #if mid_y <= 1300 or mid_y >= 700:
    #    threshold_1 = 320 #350
    #    threshold_2 = 370 #280
    for i in range(1,len(STORED_MID_Y)+1):
        #print(STORED_IDS[-i-1],STORED_MID_Y[-i-1],' ',i)
        
        
        #if(STORED_MID_Y[-i]+threshold_2>=mid_y and STORED_MID_Y[-i]-threshold_1<=mid_y): # and IS_NEW): #previous 150 #200
        if(STORED_MID_Y1[-i]-threshold_1<=y1 and STORED_MID_Y1[-i]+threshold_1>=y1) or (STORED_MID_Y2[-i]-threshold_2<=y2 and STORED_MID_Y2[-i]+threshold_2>=y2): # and IS_NEW): #previous 150 #200
            if(IS_NEW):
              
                
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
            elif Distance >5:
                STORED_MISS[-i]+=1
            #else:
            #    STORED_MISS[-i]-=1 #reset count to 2 when not moving
        #elif(STORED_MID_Y1[-i]<=y1 and STORED_MID_Y2[-i]>=y2):
        #        STORED_MISS[-i]=30
        else:
            STORED_MISS[-i]+=1    
     
    if(IS_NEW == False):
        if(y1>last_y1+20 and y2>last_y2+20):
            IS_NEW = True
        #elif (y1<last_y1+10 and y2<last_y2+10):
        #    print('Skipped first here')
        #    return -1
    
        #elif(cow_srno==1):
    if(IS_NEW):
        CATTLE_LOCAL_ID+=1
        id=CATTLE_LOCAL_ID
        if(y1<last_y1+70 and y2<last_y2+70):
            CATTLE_LOCAL_ID-=1
            #print('skipped second here')
            for i in range(len(STORED_MID_Y)):
                STORED_MISS[i]=5
            return -1
            
        STORED_IDS.append(id)
        STORED_MID_Y.append(mid_y)
        STORED_MID_Y1.append(y1)
        STORED_MID_Y2.append(y2)
        STORED_MISS.append(1)
    #print(STORED_IDS,' IDS ',STORED_MID_Y,' SMY ',mid_y,' mid_y')
    if(IS_NEW) and False:
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
    
    #print(id)
          
    result = []
    result.append(str(id-1))
    
    #region remove stored id
    removed = 0
    #print(STORED_MID_Y1,'  <===== y1, y2 =====>  ',STORED_MID_Y2,'    result =====>',result)
    #for i in range(len(STORED_MID_Y)-1,0,-1):
    #    if(y1>STORED_MID_Y1[i] and y2>STORED_MID_Y2[i]):
    #        del STORED_MISS[i-removed]  
    #        del STORED_MID_Y[i-removed]
    #        del STORED_MID_Y1[i-removed]
    #        del STORED_MID_Y2[i-removed]
    #        del STORED_IDS[i-removed]
    #        removed+=1
                 
    return result

#####XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#####

######################XXXXXXXXXXX#########
def CALCULATE_MAX_CATTLE_ID(csv_path):
    print(csv_path, " is csv_path and ")

    data = pd.read_csv(csv_path)

    list_of_csv = [list(row) for row in data.values]

    prev_id_record = [] 
    prev=None

    current_cow = []
    excel_cow_count = []
    #boxes = []
    #file_locations = []

    for i in range (len(list_of_csv)):
        filtered_id = list_of_csv[i][0]
        actual_id = list_of_csv[i][1]
        #file_locations.append(list_of_csv[i][2])
        #boxes.append([list_of_csv[i][3],list_of_csv[i][4],list_of_csv[i][5],list_of_csv[i][6]])
        try: 
            index = current_cow.index(actual_id)
            excel_cow_count[index]+=1
        except:
            current_cow.append(actual_id)
            excel_cow_count.append(1)

    maxpos = excel_cow_count.index(max(excel_cow_count))
    cattle_id = current_cow[maxpos]
    return cattle_id#,file_locations,boxes
#########XXXXXXXXXXXXXXXXXXXXXX############

#####XXXXXXXXXXX####

def get_final_cattle_id(save_dir,total_cattle):
    final_id = []
    for i in range(1,total_cattle+1):
        csv_path = save_dir + "/" + str(i) + "/" + str(i) + ".csv"
        final_id.append(CALCULATE_MAX_CATTLE_ID(csv_path))
    return final_id

###############XXXXXXXXXXXXXXXXXX##############

#############XXXXXXXXXXXXXXXXXX###############

import glob

def writeVideo(cap,filePath,csv_name,cattle_ids,total_frame):
    img_array = []
    size = (1920,1080)
    names = ['cow']
    main_csv_index = 0
    data = pd.read_csv(filePath+"/"+csv_name)

    list_of_csv = [list(row) for row in data.values]
    vid_name = os.path.basename(os.path.normpath(filePath))
    total_count = len(list_of_csv)
    prev_image_id = 1
    #vid_path = str(Path(filePath + "/" + vid_name ).with_suffix('.mp4'))
    #out = cv2.VideoWriter(vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 6, size)
    #if len(img_locations)<10: 
    #    return -1
    for filename in glob.glob(filePath+"/all_images/"+'/*.jpg'):
        img = cv2.imread(filename)
        
        #print(os.path.isfile(filename))
        #print(filename)
        #print(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
    image_location = filePath+"/all_images/";
    img_index = 0
    for ind in range(1,total_frame):
        image_path = image_location + str(ind) +".jpg"
        #print(image_path)
        img = cv2.imread(image_path)
        
        #print(os.path.isfile(filename))
        #print(filename)
        #print(filename)
        #height, width, layers = img.shape
        #size = (width,height)
        #img = img_array[img_index]
        #img_index +=1
        #img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
        #print(img.Shape)
        #cv2.imshow('Cattle Images ', img)
        #if cv2.waitKey(1) & 0xFF == ord(' '):
        #    break
        annotator = Annotator(img, line_width=3, example=str(names)) #font here
        from_index = main_csv_index
        print(from_index, ' from index')
        is_draw = False
        for i in range(from_index,total_count-1):
            
            main_csv_index+=1
            print(ind, "  <--x-->  ", list_of_csv[i][0])
            if(ind!=list_of_csv[i][0]):
                main_csv_index-=1
                break
            print('SAVING!!!!!')
            tracking_id  = list_of_csv[i][1] -1
            xyxy = [list_of_csv[i][2],list_of_csv[i][3],list_of_csv[i][4],list_of_csv[i][5]]


            annotator.box_label(xyxy,str(cattle_ids[tracking_id]), color=(15, 0, 255))
            is_draw = True
        if is_draw:
            annotated_img =cv2.resize(annotator.result(),size) 
            cap.write(annotated_img)
            #except:
            #    print('did not write')
            #    continue
    #ut.release()
    img_array=[]
    print("done ", vid_name)
    #v2.destroyAllWindows()
    return id

################XXXXXXXXXXX#################

################XXXXXXXXXXXXXXXXX##############

# Function to draw a bounding box and annotate the image
def draw_bounding_box(image, box, label):
    # Extract the coordinates from the box
    x1, y1, x2, y2 = box
    #print(x1,' ',y1,x2,y2)

    # Draw the bounding box rectangle on the image
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Define the text properties
    text = f'{label}'
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale =1
    thickness = 2

    # Calculate the size of the text
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the position for placing the text
    text_x = x1
    text_y = y1 - 10 if y1 >= 20 else y1 + 10 + text_height

    # Draw the text background rectangle
    cv2.rectangle(image, (text_x, text_y - text_height - 5), (text_x + text_width, text_y), (0, 255, 0), -1)

    # Put the label text on the image
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

# Example usage

#################XXXXXXXXXXXXX##################

##########XXXXXXXXXXXXXXX#################

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

###################XXXXXXXXXXXXXXXXXXXXX####################

#######XXXXXXXXXXXXXXXXXXX##################

def GET_LEFT_TO_RIGHT_ORDER(x1s):
    orders = []
    size = len(x1s)
    #print(' x1s     ',x1s)
    clones = x1s[:]
    for i in range(size):
        #print(clones)
        id = (x1s.index(min(clones)))
        #print(id)
        orders.append(id)
        clones.pop(clones.index(min(clones)))
    #print(' sorted order :',orders)
    return orders

#############XXXXXXXXXXXXXXXXX#################

################XXXXXXXXXXXXXXXXXX##############

def add_lines_to_excel(existing_file, data_to_add):
    """
    Add lines (rows) to an existing Excel file using a DataFrame.

    Parameters:
        existing_file (str): Path to the existing Excel file.
        data_to_add (dict): Dictionary containing data to add. Keys are column names,
                           and values are lists of data for each column.

    Returns:
        None
    """
    try:
        df_existing = pd.read_csv(existing_file,dtype={
     'ImageId': 'string',
    'LocalId': np.int64,
    'xyxy1': np.int64,
    'xyxy2': np.int64,
    'xyxy3': np.int64,
    'xyxy4': np.int64
})
    except FileNotFoundError:
        df_existing = pd.DataFrame()

    df_to_add = pd.DataFrame(data_to_add)
    df_combined = pd.concat([df_existing, df_to_add], ignore_index=True)
    df_combined.to_csv(existing_file, index=False)


######XXXXXXXXXXXXXXXXXX###############

##############XXXXXXXXXXXX##########

def create_default_csv(csv_file_path):
    default_columns = ['ImageId', 'LocalId', 'xyxy1', 'xyxy2', 'xyxy3', 'xyxy4']  # Add your desired column headers here

    # Create an empty DataFrame with the default columns
    empty_df = pd.DataFrame(columns=default_columns)

    # Save the empty DataFrame to a CSV file
    empty_df.to_csv(csv_file_path, index=False)
    
##############XXXXXXXXX############

############XXXXXXXXXXXXXXXXXXXXXX______________________---------------_____________XXXXXXXXXXXXXX###################



X1=200 #same as NEW_BLACK_X1
X2=400 #same as NEW_BLACK_X2 # incase of x2 out of bound
Y1=120
Y2=500

Y1_NEW=120 #125  #decrease here to extend, increase to shrink 
Y2_NEW=510  #500  # redyce here to extend , increase to do vice casa 460 previous
FRAME = 1

default=640

BATCH = 100
BATCH_COUNT = 1
PREV_BATCH = 0

LAST_SEEN = time.time()
FIRST_SEEN = True
demo_img_save_path = []

#end
prevId_record =[]
MAX_prevId = [] 
MAX_xyxy1 = [] 
MAX_xyxy2 = [] 
MAX_xyxy3 = [] 
MAX_xyxy4 = [] 
MAX_orgId = []
IMAGE_STORED_LOCATION = []
IMAGE_ID_LIST = []
LOCAL_ID_LIST = []
#end
#prevId_record =[]
TOTAL_CATTLE_COUNT = 0 


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
CATTLE_LOCAL_ID= 1
IS_FIRST_CATTLE = True

MAIN_DF_COLUMNS = ['ImageId',	'LocalId',	'xyxy1',	'xyxy2',	'xyxy3',	'xyxy4'] # using to store all cattle information in here instead of list
MAIN_DF = pd.DataFrame(columns = MAIN_DF_COLUMNS)

# Predict with the model
project = SAVE_PATH
name = 'identification'
#dataset = "D:\\815_CowDataChecking\\Honkawa\\2023-08-05\\03040506\\" 
#save_vid_name=  dataset.split("\\")[-1].replace('.mkv','_track')  #open this when running single video
save_vid_name = dataset.split("\\")[-1]+"_track" # open this when running multiple videos

print(save_vid_name)
results = model(dataset,imgsz=(960,640),save=False,retina_masks=False,show=False,stream=True,device='0',conf=0.3)
#results = model('D:\\815_CowDataChecking\\20221228\\20221228_E_cow\\20221228_151533_D474.mkv',imgsz=1088,save=False,retina_masks=False,show=False,stream=True,device='0',conf=0.2)
#save_dir = increment_path(Path(project) / name, exist_ok=True)  # increment run
save_dir = increment_path(Path(project) / name,mkdir=True)
csv_main_file_path = str(save_dir) + "\main_csv.csv"
#create_default_csv(csv_main_file_path)
#(save_dir / 'labels' if False else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
manual_cow_count = 0

#cap.set(4, 480)
save_vid_path = str(Path(os.path.join(save_dir, save_vid_name)).with_suffix('.mp4'))
print(save_vid_path)
cap = cv2.VideoWriter(save_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 30, (960,640))
skip_count = 2
skip_track = 1
image_count = 1

SAVING_THRESHOLD =200
SAVED_COUNT = 1

CATTLE_SAVING_THRESHOLD = 15
CATTLE_SAVED_COUNT = 1
for result in results:
    if(skip_track==2):
        skip_track = 1
        continue
    skip_track+=1
    
    #print(result.boxes)
    vid_path = result.path
    filename = vid_path.split("\\")[-1].replace(".mp4","")
    
    boxes = result.boxes.cpu().numpy()
    detections = boxes.xyxy.tolist()
    #print(detections)
    # Sort the detections based on the x1 coordinate (i.e., left-to-right)
    #detections.sort(key=lambda x: x[0])
    left_to_right = GET_LEFT_TO_RIGHT_ORDER([t[0] for t in detections])
    #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #print(left_to_right)
    #print(detections)
    
    ori_img = cv2.resize(result.orig_img, (640,384), interpolation = cv2.INTER_AREA)
    annotator = Annotator(ori_img)
    box_count = 0
    cow_position = 1
    
    h,w = result.orig_shape
    count = 1
    b_boxes = []
    masks = []
    ids = []
    has_cattle = False
    if  result.masks != None:
        
        #result_masks = result.masks.cpu().numpy().masks.astype(bool)
        #for m in result.masks.cpu().numpy().masks.astype(bool):
        result_masks = result.masks.cpu().numpy().masks.astype(bool)
        
        for LR_index in left_to_right:
            m = result_masks[LR_index]
        #for i in range(1,len(result_masks)+1):
        #for m in result.masks.masks.astype(bool):
            
            xyxy = boxes[LR_index].xyxy[0]
            #print('LR  INDEX  ',LR_index)
            #print(xyxy)
        #   m = result_masks[-1]
            #print(xyxy)
            #print(m.shape)
            box_count += 1
            x1= int(xyxy[0])
            y1= int(xyxy[1])
            x2= int(xyxy[2])
            y2= int(xyxy[3])
            #print(xyxy)
            #print(m.shape)
            
            ################## Validate  #####################
            if(check_withinROI_NEW(x1,y1,x2,y2,h,w)==False):
                #print("I was skipped")
                continue
            #print("not skipped")
            has_cattle = True
            #new_results.append(result)
            box_left = x1
            box_top = y1
            box_w = x2 - x1
            box_h = y2 - y1

            new = np.zeros_like(ori_img, dtype=np.uint8)
            new[m] = ori_img[m]
            
           
            x1= int(x1 * (640/1920))
            x2= int(x2 * (640/1920))
            y1= int(y1 * (384/1080))
            y2= int(y2 * (384/1080))

            crop = new[y1:y2, x1:x2]
            img = cv2.resize(crop, (SIZE, SIZE))
            img=img / 255.0
            
            ############# LABLE
            label = Predict_SVM(img)
            
            
            ###### LABEL
            #img = cv2.resize(crop, (SIZE, SIZE))
            #img=img / 255.0
            #print(box_left,'    xxxxxx     ' ,box_w)
            prev_id = Take_Prev_Label(box_left,box_w,label,cow_position) ## just passing x values instead of y
          #########################################  
            HAS_COW=True
            #has_cattle = True
            if(prev_id==-1): #skip cattle when prev_id // filter id is -1
                if(count==1):
                    has_seen_cattle=False
                    count-=1
                #print('skipped')
                continue
            has_cattle = True
            
            ids.append(prev_id[0])
            masks.append(m)
            b_boxes.append([x1,y1,x2,y2])
            #print(prev_id)
            cow_position+=1 
            BATCH_COUNT = prev_id[0] # skip batch count here  
            
           
            if CATTLE_SAVED_COUNT > CATTLE_SAVING_THRESHOLD:
                for i in range(7):
                    tracking_id = MAX_prevId[0][0]
                    df = pd.DataFrame(MAX_prevId[0], columns = ['ID'])
                    print("saving max_orgId csv :", len(MAX_orgId[0]), ' Tracking ID is :', tracking_id)
                    try:
                        org_ids = torch.tensor(MAX_orgId[0], device = 'cpu')
                        df["Original"] = org_ids
                    except:
                         df["Original"] = MAX_orgId[0]
                    
                    del MAX_orgId[0]
                    del MAX_prevId[0]
                    del prevId_record[0]
                    
                    save_csv_each_path = str(Path(save_dir / str(tracking_id) / f'{str(tracking_id)}.csv'))
                    df.to_csv(save_csv_each_path, index= False)##asdfasdf
                CATTLE_SAVED_COUNT = 8
            ###################### CREATE dir to save img
            #print(prev_id)
            base_path = str(Path(save_dir / prev_id[0]))
            if not os.path.exists(base_path):
                os.makedirs(base_path)


            #demo_annotated_img_save_path = Path(base_path+ '/' + f'{image_count}.jpg')
            CATTLE_DATA = {}
            try:
                
                index_prevId = prevId_record.index(int(prev_id[0]))
                #print(index_prevId)
                MAX_prevId[index_prevId].append(int(prev_id[0]))#,int(label[0]),xyxy)
                
                MAX_orgId[index_prevId].append(int(label[0]))

                #IMAGE_STORED_LOCATION[index_prevId].append(demo_annotated_img_save_path)

            except :
                CATTLE_SAVED_COUNT += 1
                TOTAL_CATTLE_COUNT +=1 
                print('new cattle ')
                prevId_record.append(int(prev_id[0]))
                # print(len(prevId_record)-1, 'prevID_record ', len(MAX_prevId) , 'max_previd' )

                #MAX_prevId[len(prevId_record)-1].append(int(prev_id[0]))#,int(label[0]),xyxy)
                #MAX_xyxy[len(MAX_prevId)-1].append(xyxy)
                #MAX_orgId[len(MAX_prevId)-1].append(int(label[0]))
                MAX_prevId.append([int(prev_id[0])])#,int(label[0]),xyxy)

            
                MAX_orgId.append([int(label[0])])
                #IMAGE_STORED_LOCATION.append([demo_annotated_img_save_path])
            IMAGE_ID_LIST.append(image_count)
            LOCAL_ID_LIST.append(int(prev_id[0]))
            MAX_xyxy1.append(x1)
            MAX_xyxy2.append(y1)
            MAX_xyxy3.append(x2)
            MAX_xyxy4.append(y2)
            
            #try:
                #demo_vid_index = demo_vid_path.index(demo_path)
            #    demo_vid_index = demo_img_save_path.index(base_path)

                #print("path exist")
            #except:
                #manual_summarize_ids.append(int(prev_id[0]))
                #manual_local_ids.append(manual_id)
                #manual_id +=1
            #    demo_img_save_path.append(base_path)
                
            try:
                ori_img = overlay(ori_img,m,(0,0,255),0.3)
                #cv2.imwrite(str(demo_annotated_img_save_path), ori_img) #save ori_img
                #print(demo_annotated_img_save_path)
            except:
                print('cannot save ',demo_annotated_img_save_path)
            #change cropped size here  #230 to 215 410 to 390

            manual_cow_count += 1

            manual_cow_count += 1

    #frame = annotator.result() 
   
    
    
    frame = cv2.resize(ori_img, (1920,1080), interpolation = cv2.INTER_AREA)
    
    if has_cattle:  
        data_to_add = {
        'ImageId': IMAGE_ID_LIST,
        'LocalId': LOCAL_ID_LIST,
        'xyxy1': MAX_xyxy1,
        'xyxy2': MAX_xyxy2,
        'xyxy3': MAX_xyxy3,
        'xyxy4': MAX_xyxy4
        # Add more columns as needed
        }
        df_to_add = pd.DataFrame(data_to_add)
        MAIN_DF = pd.concat([MAIN_DF,df_to_add], ignore_index=True)
        
        base_path = str(Path(save_dir / 'all_images'))
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        IMAGE_ID_LIST = []
        LOCAL_ID_LIST = []
        MAX_xyxy1 = []
        MAX_xyxy2 = []
        MAX_xyxy3 = []
        MAX_xyxy4 = []

        demo_annotated_img_save_path = Path(base_path+ '/' + f'{image_count}.jpg')
        cv2.imwrite(str(demo_annotated_img_save_path), ori_img) #save ori_img
        SAVED_COUNT += 1
        image_count += 1
        
        #ori_img = cv2.resize(ori_img,(1088,1088))
        for i in range(len(b_boxes)):
            box = b_boxes[i]
            mask = masks[i]
            #print("drawing")
            #print(ids)
            
            draw_bounding_box(ori_img,(box[0],box[1],box[2],box[3]),str(ids[i]))
            ori_img = overlay(ori_img,mask,(0,0,255),0.3)
        frame = cv2.resize(ori_img, (960,640), interpolation = cv2.INTER_AREA)
        cap.write(frame)
    
    
    #cv2.imshow('YOLO V8 Detection', frame)
    #if cv2.waitKey(1) & 0xFF == ord(' '):
    #    break
        
cap.release() #tracking video
cv2.destroyAllWindows()
image_count -=1 

###### write final csv
for csv_index in range(len(MAX_prevId)):
    df = pd.DataFrame(MAX_prevId[csv_index] , columns = ['ID'])
    tracked_id = MAX_prevId[csv_index][0]
    try:
        org_ids = torch.tensor(MAX_orgId[csv_index], device = 'cpu')
        df["Original"] = org_ids
    except:
        df["Original"] = MAX_orgId[csv_index]
    now=str(datetime.now().date())

    save_csv_each_path = str(Path(save_dir / str(tracked_id) / f'{str(tracked_id)}.csv'))
    df.to_csv(save_csv_each_path, index= False)##asdfasdf
#All records     


IMAGE_STORED_LOCATION = []

cattle_ids = []
#################################\
manual_summarize_ids = []
manual_local_ids = []

manual_summarize_ids = []
manual_local_ids = []
#### write video after saving csv
final_cattle_count = 1

# SAVING MAIN CSV
MAIN_DF.to_csv(csv_main_file_path, index=False)

MAIN_DF = None # CLEAR AFTER SAVING


save_vid_name=  dataset.split("\\")[-1].replace('.mp4','')+'_classification'

save_vid_path = str(Path(os.path.join(save_dir, save_vid_name)).with_suffix('.mp4'))
print(save_vid_path)
classification_vid = cv2.VideoWriter(save_vid_path,cv2.VideoWriter_fourcc(*'mp4v'), 15, (640,384))
print(save_vid_path)


#SAVING SUMMARIZE IDS
manual_summarize_ids = get_final_cattle_id(str(save_dir),TOTAL_CATTLE_COUNT)

summarize_id_csv = pd.DataFrame(manual_summarize_ids, columns = ["Cow Id"])

summarize_id_csv.to_csv(str(save_dir)+'/summarize_id_'+now+'.csv', index= False) 
#FINISH SAVING SUMMARIZE IDS

csv_name = 'main_csv.csv'
#region Write Video
try:
    writeVideo(classification_vid,str(save_dir),csv_name ,manual_summarize_ids,image_count)
except:
    print(" I am ugly")
#summarize_id_csv.to_csv(str(save_dir)+'/summarize_id_'+now+'.csv', index= False) 
classification_vid.release()
cv2.destroyAllWindows(

######XXXXXXXXXXXXXXXXX##########################

