import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import  torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
class block(nn.Module):
    def __init__(self,resblock=nn.Identity(),pic_size=28):
        super(block, self).__init__()
        self.flatten=nn.Flatten()
        self.block=nn.Sequential(
            BasicConv2d(64,64,kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        self.upsample=None
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
            if self.upsample==None:
                self.upsample=nn.Upsample(x.shape[2:], mode = 'bilinear', align_corners = False)
            x_append=self.upsample(x_append)
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
    def __init__(self,layer,pic_size=28,device="cuda"):
        super(NeuralNetwork, self).__init__()
        self.precess=BasicConv2d(1,64,kernel_size=3)
        self.flatten=nn.Flatten()
        self.picsize = pic_size
        self.block=block(nn.Identity(),pic_size=self.picsize)

        tem=self.block
        for i in range(layer-1):
            tem.resblock=block(nn.Identity())
            tem=tem.resblock
        self.fcs={ }
    def forward(self, x,layer):
        self.block.frozen(layer-1)
        x=self.precess(x)
        #x = self.flatten(x)
        x=self.block(x,layer)
        x_flatten=nn.Flatten()(x)
        if self.fcs.get(layer)==None:
            self.fcs[layer]=nn.Sequential( nn.Dropout(p=0.2), nn.Linear(x_flatten.shape[1], 10),nn.ReLU()).to(device)
        output= self.fcs[layer](x_flatten)
        return output
    def sol_all(self,layer):
        self.block.sol(layer)
def train(dataloader, model, loss_fn,layer):
    size = len(dataloader.dataset)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for i in range(layer):
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            if i == 1:
                pass
            # Compute prediction error
            pred = model(X, i + 1)
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"layer {i + 1} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        model.sol_all(layer)

def test(dataloader, model, loss_fn,layer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i in range(layer):
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred= model(X, i+1)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

layer=5
model = NeuralNetwork(layer,28).to(device)
loss_fn = nn.CrossEntropyLoss()

epochs = 10
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn,layer)
    test(test_dataloader, model, loss_fn,layer)
print("Done!")

