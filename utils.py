###Show an image from cifar10###
import matplotlib.pyplot as plt
import torch
import numpy as np
from UNet import *
from DDPMs import *
from ResNets import *
import numpy as np
import torch
import random
from tqdm.notebook import tnrange



device = "cuda"
def show(img,index = None,detach=False):
    img1 = torch.tensor(np.copy(img))
    if detach:
        img = img.detach()
    new_img = img1.permute(1,2,0)
    plt.imshow(new_img)
    plt.show()
    if index:
        print("This image has the label: " + str(cifar10['train']['label'][index]))
        
        

###Set random seed###
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
#Load local variables

def load_resnet():
    device = "cuda"
    #Cifar10 resnet model
    resnet = resnet50(pretrained=True)

    def scale_x_imgs(x):
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2471, 0.2435, 0.2616]
        y = torch.clone(x)
        for i in range(y.shape[1]):
            y[:,i] = (x[:,i]-mean[i])/std[i]
        return y

    #Scaling layer
    class CifarScalingLayer(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return scale_x_imgs(x)

    #New model
    cifar_layer = CifarScalingLayer()
    resnet_scaled = nn.Sequential(
        cifar_layer,
        resnet
    )
    resnet = resnet.to(device)
    resnet_scaled = resnet_scaled.to(device)
    resnet.eval();
    resnet_scaled.eval();
    return resnet, resnet_scaled

#def load_diff_model():
#    device = "cuda"
#    unet = torch.load('/home/ubuntu/PSU_research/Notebooks/state_dicts/optimal_unet.pt',device)
#    unet.eval();
#    diff_model = DenoiseDiffusion(unet,5000,device)
#    return diff_model
def load_eps_model():
    device = "cuda"
    unet = torch.load('/home/ubuntu/PSU_research/Notebooks/state_dicts/optimal_unet.pt',device)
    unet.eval();
    return unet
#def load_oneshot_unet():
#    device = "cuda"
#    unet = torch.load('/home/ubuntu/PSU_research/Notebooks/state_dicts/oneshot_unet.pt',device)
#    unet.eval();
#    return unet

def load_accuracy():
    device = "cuda"
    from tqdm.notebook import tnrange
    from torchmetrics import Accuracy
    acc = Accuracy(num_classes=10).to(device)
    
def load_data():
    device = "cuda"
    import pickle
    with (open("/home/ubuntu/PSU_research/Cifar10/x_test.pickle", "rb")) as openfile:
        x_test = pickle.load(openfile)
    with (open("/home/ubuntu/PSU_research/Cifar10/y_test.pickle", "rb")) as openfile:
        y_test = pickle.load(openfile)
    with (open("/home/ubuntu/PSU_research/Cifar10/x_train.pickle", "rb")) as openfile:
        x_train = pickle.load(openfile)
    with (open("/home/ubuntu/PSU_research/Cifar10/y_train.pickle", "rb")) as openfile:
        y_train = pickle.load(openfile)
    #Send to device
    device = "cuda"
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    return x_train,y_train,x_test,y_test
    
def load_cert_denoiser():
    device = "cuda"
    certified_denoiser = torch.load('/home/ubuntu/PSU_research/Notebooks/state_dicts/certified_denoiser.pt',device)
    certified_denoiser.eval();
    return certified_denoiser


from torchmetrics import Accuracy
def get_accuracy_batch(model,x,y,batch_size):
    acc = Accuracy(num_classes = 10).to(device)
    accs = 0
    for i in tnrange(0,len(x),batch_size):
        accs += acc(model(x[i:i+batch_size]),y[i:i+batch_size])
    return accs/(i+1)


import pickle
import torch
def load_x_adv(directory,adv_type):
    device = "cuda"
    x_adv = torch.empty(size = (10000,3,32,32))
    for i in range(10):
        with (open("/home/ubuntu/PSU_research/"+str(directory)+"/x_adv_"+adv_type+"_"+str(i+1)+".pkl", "rb")) as openfile:
            x_curr = pickle.load(openfile)
        x_adv[i*1000:(i+1)*1000] = x_curr
    return x_adv.to(device)

