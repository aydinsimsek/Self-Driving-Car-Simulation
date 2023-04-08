import os
import pandas as pd 
import ntpath
import random
import numpy as np
import cv2  
from sklearn.utils import shuffle
from imgaug import augmenters as iaa

def csv_reader(data_dir):
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']    
    data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names = columns)
    data['left'] = [ntpath.split(p)[1] for p in data['left']] 
    data['center'] = [ntpath.split(p)[1] for p in data['center']] 
    data['right'] = [ntpath.split(p)[1] for p in data['right']] 
    return data

def truncate(data, bins, num_of_bins, sample_threshold):
    """
    Truncates the data frame to have more uniformly distributed steering angles    
    """
    remove_indices = []
    for i in range(num_of_bins):
        indices = []
        for j in range(len(data['steering'])):
            if data['steering'][j] >= bins[i] and data['steering'][j] <= bins[i+1]:
                indices.append(j) 
        indices = shuffle(indices) 
        indices = indices[sample_threshold:]
        remove_indices.extend(indices)
    data.drop(data.index[remove_indices], inplace = True)
    return data

def unpack(data, dir, correction):
    """
    Unpacks the data frame to overall image paths and steering angles 
    Corrects the steering angles for images taken from the left and right cameras
    """
    paths = []
    angles = []
    for i in range(len(data)):
        data_sample = data.iloc[i]  
        center_path, left_path, right_path = data_sample[0], data_sample[1], data_sample[2]
        paths.append(os.path.join(dir, center_path.strip()))
        angles.append(float(data_sample[3])) 
        paths.append(os.path.join(dir,left_path.strip()))
        angles.append(float(data_sample[3]) + correction)
        paths.append(os.path.join(dir,right_path.strip()))
        angles.append(float(data_sample[3]) - correction) 
    image_paths = np.asarray(paths)
    steering_angles = np.asarray(angles) 
    return image_paths, steering_angles

def zoom_in(img): 
    #scale_factor = np.random.uniform(low = 1, high = 1.3)
    #img = cv2.resize(img, None, fx = scale_factor, fy = scale_factor)
    #img = img[0:160, 0:320, :]
    aug = iaa.Affine(scale = (1, 1.3))  
    img = aug.augment_image(img)          
    return img

def translate(img):
    height, width = img.shape[:2]
    tx = np.random.uniform(low = -32, high = 32)
    ty = np.random.uniform(low = -16, high = 16)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, M, (width, height))
    return img

def adjust_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * np.random.uniform(low = 0.2, high = 1.2) 
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img

def horizontal_flip(img, angle):
    img = cv2.flip(img, 1)  
    angle = -angle
    return img, angle

def random_augment(path, angle):
    img = cv2.imread(path) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.random.rand() < 0.5:
        img = zoom_in(img)
    if np.random.rand() < 0.5:
        img = translate(img)
    if np.random.rand() < 0.5:
        img = adjust_brightness(img)
    if np.random.rand() < 0.5:
        img, angle = horizontal_flip(img, angle)
    return img, angle

def batch_generator(paths, angles, batch_size, isTraining):
    while True:
        batch_images = [] 
        batch_angles = []
        for i in range(batch_size):
            rand_idx = random.randint(0, len(paths) - 1) 
            if isTraining:
                img, steering = random_augment(paths[rand_idx], angles[rand_idx])
            else:
                img, steering = cv2.imread(paths[rand_idx]), angles[rand_idx]  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess(img) 
            batch_images.append(img)
            batch_angles.append(steering)
        yield (np.asarray(batch_images), np.asarray(batch_angles))    

def crop(img):
    """
    Leaves only the region of interest, crops the rest
    """
    return img[60:140, :, :]   

def rgb2yuv(img):
    """
    Changes the color model to increase the efficiency of the neural network architecture used 
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  

def blur(img):
    return cv2.GaussianBlur(img, (3, 3), 0) 

def resize(img):
    return cv2.resize(img, (200, 66)) 

def normalize(img):
    return img / 255

def preprocess(img):
    img = crop(img)
    img = rgb2yuv(img)
    img = blur(img) 
    img = resize(img)
    img = normalize(img) 
    return img
