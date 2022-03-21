import torch
import torch.nn as nn
from torchvision.models import inception_v3
class block(nn.Module):
    def __init__(self,resblock=nn.Identity(),pic_size=2):
        super(block, self).__init__()
        self.flatten=nn.Flatten()
        self.block=nn.Sequential(
            nn.Linear(pic_size * pic_size, pic_size * pic_size),
            nn.ReLU(),
        )
        self.resblock=resblock
    def frozen(self,index):
        if index>0:
            for i in self.block.parameters():
                i.requires_grad = False
            if not isinstance(self.resblock, nn.Identity):
                self.resblock.frozen(index - 1)
    def sol(self,index):
        if index > 0:
            for i in self.block.parameters():
                i.requires_grad = True
            if not isinstance(self.resblock, nn.Identity):
                self.resblock.sol(index - 1)
    def forward(self, x,layer_lim):
        x = self.block(x)
        x_append=0
        if not isinstance(self.resblock,nn.Identity) and layer_lim>1:
            x_append=self.resblock(x,layer_lim-1)
        output=x+x_append
        return output
    def para(self,layer):
        para=[]
        if layer==1:
            for i in self.block.parameters():
                para.append(i)
        else:
            for i in self.resblock.para(layer-1):
                para.append(i)
        return para

class NeuralNetwork(nn.Module):
    def __init__(self,layer,pic_size=2,device="cpu"):
        super(NeuralNetwork, self).__init__()
        self.flatten=nn.Flatten()
        self.picsize = pic_size
        self.block=block(nn.Identity(),pic_size=self.picsize)
        self.device=device
        tem=self.block
        for i in range(layer-1):
            tem.resblock=block(nn.Identity())
            tem=tem.resblock
        self.fcs={ }
    def forward(self, x,layer):
        self.block.frozen(layer-1)
        if self.fcs.get(layer)==None:
            self.fcs[layer]=nn.Sequential( nn.Dropout(p=0.2), nn.Linear(self.picsize * self.picsize, 10),nn.ReLU()).to(self.device)
        x = self.flatten(x)
        x=self.block(x,layer)
        output= self.fcs[layer](x)
        return output,self.para(layer)
    def para(self,layer):
        return self.block.para(layer)
    def sol_all(self,layer):
        self.block.sol(layer)


if __name__=="__main__":
    model=NeuralNetwork(2,2)
    x=torch.randn(3,1,2,2)
    y=torch.randn(3)
    y_pred,need_grad=model(x,2)
    froze_para=model.para(2)
    for i in need_grad:
        print(i)
    for i in froze_para:
        print(i)
    model.sol_all(4)
    froze_para = model.para(1)
    for i in froze_para:
        print(i)

    print(y_pred.shape)
