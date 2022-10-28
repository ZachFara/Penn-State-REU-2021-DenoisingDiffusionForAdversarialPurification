import random
import torch
device = "cuda"

from torchmetrics import Accuracy
import random
import numpy as np
from utils import show

class PurificationProcess:
    def __init__(self, model,x,y,num_classes, diffusion_model,t, x_adv):
        self.model = model
        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.x_adv = x_adv
        self.diffusion_model = diffusion_model
        self.t = torch.tensor([t]).to(device)

    def get_acc(self,model,x,y,num_classes):
        if model is None:
            model = self.model
        if num_classes is None:
            num_classes = self.num_classes
        acc = Accuracy(num_classes= num_classes).to(device)
        accuracy = acc(model(x),y)
        return accuracy

    def purify(self,x = None,diffusion_model = None,t = None,show = False):
        if x is None:
            x = self.x
        if diffusion_model is None:
            diffusion_model = self.diffusion_model
        if t is None:
            t = self.t
        noisy_x = diffusion_model.q_sample(x,t)[0]
        denoised_images = diffusion_model.denoise(noisy_x,t)
        if show:
            print("Image before forwards sampling")
            show(x[0].cpu())
            print("Image after forwards sampling")
            show(noisy_x[0].squeeze().cpu())
            print("Image after backwards sampling")
            show(denoised_images[0].squeeze().detach().cpu())
        return denoised_images
    def get_robust_acc(self):
        x_pure = self.purify(self.x_adv,self.diffusion_model,self.t)
        new_robust_acc = self.get_acc(self.model,x_pure,self.y,self.num_classes)
        return new_robust_acc
    def compare_acc(self,show = False):
        raw_acc = self.get_acc(self.model,self.x,self.y,self.num_classes)
        robust_acc = self.get_acc(self.model,self.x_adv,self.y,self.num_classes)
        x_pure = self.purify(self.x_adv,self.diffusion_model,self.t,show = show)
        new_robust_acc = self.get_acc(self.model,x_pure,self.y,self.num_classes)
        return new_robust_acc
    def bpda_eot_func(self,x = None,mode = None):
        if x is None:
            x = self.x
        if mode is None:
            mode = "purify"
        if mode == "purify":
            return self.purify(x)
        if mode == "classify":
            return self.model(x)
        
#One shot purification
        
class OneshotPurification:
    def __init__(self, diffusion_model,denoise_model, classifier,num_classes,t):
        self.diffusion_model = diffusion_model
        self.denoise_model = denoise_model
        self.classifier =classifier
        self.num_classes = num_classes
        self.t = torch.tensor([t]).to(device)

    def get_acc(self,x,y,classifier = None,num_classes = None):
        if classifier is None:
            classifier = self.classifier
        if num_classes is None:
            num_classes = self.num_classes
        acc = Accuracy(num_classes= num_classes).to(device)
        accuracy = acc(classifier(x),y)
        return accuracy
    def purify(self,x,denoise_model = None,diffusion_model = None,t = None,weight = None,show_ = False):
        if denoise_model is None:
            denoise_model = self.denoise_model
        if diffusion_model is None:
            diffusion_model = self.diffusion_model
        if weight is None:
            weight = 1
        if t is None:
            t = self.t
        noisy_x = diffusion_model.q_sample(x,t)[0]
        denoised_images = denoise_model(x,t)
        #Now we weight the denoised images vs the noisy ones
        weighted_avg_imgs = (noisy_x + weight * denoised_images) / (1+ weight)
        if show_:
            print("Image before forwards sampling")
            show(x[0].cpu())
            print("Image after forwards sampling")
            show(noisy_x[0].squeeze().cpu())
            print("Image after backwards sampling")
            show(denoised_images[0].squeeze().detach().cpu())
        return weighted_avg_imgs
    def get_robust_acc(self,x,y,weight):
        x_pure = self.purify(x,self.denoise_model,self.diffusion_model,self.t,weight)
        new_robust_acc = self.get_acc(x_pure,y)
        return new_robust_acc
    def bpda_eot_func(self,x = None,mode = None):
        if x is None:
            x = self.x
        if mode is None:
            mode = "purify"
        if mode == "purify":
            return self.purify(x)
        if mode == "classify":
            return self.classifier(x)
    
    