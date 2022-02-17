import gdown
import time


success = False
while not success:
    fileName = gdown.download('https://drive.google.com/u/0/uc?id=1VIvMfS_dRaPpfbUFlDm33TsTJ9f3sUdX&export=download&confirm=t',
    'u2net.pth',
    quiet=False)    
    if fileName is None:
        print("Failed to download model file, Retrying in 5 seconds")
        time.sleep(5)
    else:
        success = True
        
print("Model file downloaded successfully!")