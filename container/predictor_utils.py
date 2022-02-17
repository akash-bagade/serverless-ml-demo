import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

# Resize image to given output size
def RescaleT(image, output_size):
    img = transform.resize(image,(output_size, output_size), mode='constant')
    return img

# Convert input array to tensor
def ToTensorLab(image, flag):
    """Convert ndarrays in sample to Tensors."""
    # change the color space
    if flag == 2: # with rgb and Lab colors
        tmpImg = np.zeros((image.shape[0],image.shape[1],6))
        tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
        if image.shape[2]==1:
            tmpImgt[:,:,0] = image[:,:,0]
            tmpImgt[:,:,1] = image[:,:,0]
            tmpImgt[:,:,2] = image[:,:,0]
        else:
            tmpImgt = image

        tmpImgtl = color.rgb2lab(tmpImgt)

        # nomalize image to range [0,1]
        tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
        tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
        tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
        tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
        tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
        tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
        tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
        tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
        tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

    elif flag == 1: #with Lab color
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))

        if image.shape[2]==1:
            tmpImg[:,:,0] = image[:,:,0]
            tmpImg[:,:,1] = image[:,:,0]
            tmpImg[:,:,2] = image[:,:,0]
        else:
            tmpImg = image

        tmpImg = color.rgb2lab(tmpImg)

        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

        tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
        tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
        tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

    else: # with rgb color
        tmpImg = np.zeros((image.shape[0],image.shape[1],3))
        image = image/np.max(image)
        if image.shape[2]==1:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
        else:
            tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
            tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
            tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

    # change the r,g,b to b,r,g from [0,255] to [0,1]
    tmpImg = tmpImg.transpose((2, 0, 1))
    # convert (3,320,320) to (1,3,320,320)
    tmpImg =  np.expand_dims(tmpImg, 0)
    # convert numpy array to tensor
    tmpImg = torch.from_numpy(tmpImg)
    return tmpImg
    
# generate u2net mask for given input image
def predict_u2net_mask(img_np, model):  
    inputs_test = img_np
    inputs_test = inputs_test.type(torch.FloatTensor)

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1,d2,d3,d4,d5,d6,d7= model(inputs_test)

     # normalization
    pred = d1[:,0,:,:]
    pred = normPRED(pred)

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np = Image.fromarray(predict_np*255).convert('RGB')

    del d1,d2,d3,d4,d5,d6,d7
    return predict_np
