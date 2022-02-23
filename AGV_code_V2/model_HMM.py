import glob
import os
import numpy as np
import copy as copy
from tqdm import tqdm

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

path = r'C:\Users\franc\Desktop\RT_Detection_AGV\data\train'

def class_extract(path):
    classes = next(os.walk( path) )
    classes=classes[1]
    return classes

def path_extraction(path,classes):
    train_dir=[]
    for num_classes in range(len(classes)):
        train_dir.append(glob.glob(path+"\\"+classes[num_classes]))
    return train_dir

