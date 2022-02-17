import os
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import cv2
import base64
import flask
import logging 
import json
from predictor_utils import *
from u2net_code import U2NET
from torchvision import transforms
from skimage import io, transform

logger = logging.getLogger() 

app = flask.Flask(__name__)

# Parameters
model_name='u2net'
model_path = '../ml/model/u2net.pth'


# Load the models
# Initialize the U2Net Model
if(model_name=='u2net'):
    unet_model = U2NET(3,1)

if torch.cuda.is_available():
    unet_model.load_state_dict(torch.load(model_path))
    unet_model.cuda()
    print("Cuda Available:",torch.cuda.is_available())
else:
    unet_model.load_state_dict(torch.load(model_path, map_location='cpu'))
unet_model.eval()


def detector(img_path, u2net_model):
    # read the input image
    img_np = io.imread(img_path)
    img_name = img_path.split('/')[-1][:-4]
    mask_name = 'mask_' + img_name + '.png'
    oimg = img_np.copy()
    
    # Generate u2net mask
    img_np = RescaleT(img_np, 320)
    img_np = ToTensorLab(img_np, 0)
    # get predicted mask
    u2net_mask = predict_u2net_mask(img_np, u2net_model)
    
    u2net_mask_new = u2net_mask.resize((oimg.shape[1], oimg.shape[0]),resample=Image.BILINEAR)
    u2net_mask_new = np.array(u2net_mask_new)
    img = cv2.cvtColor(u2net_mask_new, cv2.COLOR_RGB2BGR)
    img_str = cv2.imencode('.jpg', img)[1].tostring()
    return img_str


def health_check(model_dir):
    try:
        health=True
    except Exception as ex:
        logger.error(ex)
        health=False
    
    return health

@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully.
    
    """
    health = health_check(model_path)
    status = 200 if health else 404

    return flask.jsonify(response='\n', status=status)

@app.route('/invocations', methods=['POST'])
def transformation():

    if flask.request.content_type == 'image/jpeg':

        data = flask.request.data
        # convert binary string data to numpy and decoding to image
        nparr = np.fromstring(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_save_name = "../../tmp/" + "rifle1.jpg"
        cv2.imwrite(img_save_name, img)
        img_binary = detector(img_save_name, unet_model)
        img_data = base64.b64encode(img_binary)
        return flask.jsonify(response={"img":img_data.decode("utf-8")},
                             status="Success")
    else:
        return flask.jsonify(response='This predictor only supports jpg image',
                             status="Failure")

        
if __name__ == "__main__":
    app.run()