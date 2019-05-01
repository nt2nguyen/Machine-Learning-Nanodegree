import torch.nn as nn
import torch.nn.functional as F
import torch


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).cuda() * self.stddev)
        return din 
    
class BasicBlock(nn.Module):

    def __init__(self, inchannels, outchannels, stride=1, padding = 0, dilation = 1):
        super(BasicBlock, self).__init__()
        self.Conv1D_1 = nn.Conv1d(inchannels, outchannels, kernel_size = 3, stride=stride,
                                padding = padding, dilation = dilation)
        #self.noise1 = GaussianNoise(stddev=0.0001)
        self.bn1 = nn.BatchNorm1d(outchannels)
        self.relu1 = nn.ReLU(inplace=True)
        self.Conv1D_2 = nn.Conv1d(outchannels, outchannels, kernel_size = 3, stride=stride,
                                padding = padding, dilation = dilation)
        #self.noise2 = GaussianNoise(stddev=0.0001)
        self.bn2 = nn.BatchNorm1d(outchannels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.Conv1D_1(x)
        #out = self.noise1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.Conv1D_2(out)
        #out = self.noise2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out

class DeepBasicBlock(nn.Module): 
    def __init__(self, inchannels, outchannels, kernel_size = 3, stride=1, padding = 0, dilation = 1):
        super(DeepBasicBlock, self).__init__()

        self.Conv1D = nn.Conv1d(inchannels, outchannels, kernel_size = kernel_size, stride=stride,
                                padding = padding, dilation = dilation)
        self.bn = nn.BatchNorm1d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.Conv1D(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

    
class HyperSpectrumModelAvgPoolDeepCNNV2(nn.Module):
    def __init__(self, dropout_percent=0.0, kernel_size=2, stride=2):
        super(HyperSpectrumModelAvgPoolDeepCNNV2, self).__init__()
        self.noise = GaussianNoise(stddev=0.001)
        self.avgPool1 = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # 717x1 
        self.block1 = DeepBasicBlock(inchannels=1, outchannels=10, kernel_size=kernel_size, stride=stride, padding=1)#358
        self.avgPool2 = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # 717x1 
        self.block2_1 = DeepBasicBlock(10, 50, kernel_size, stride, 1) # 358x50
        self.avgPool3 = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # 717x1 
        self.block2_2 = DeepBasicBlock(50, 50, kernel_size, stride, 1) # 179x100
        self.avgPool4= torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # 717x1
        
        self.fc1 = nn.Linear(850, 350)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_percent)
        self.fc2 = nn.Linear(350, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_percent)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = x.view(-1, 1, 2151)
        x = self.noise(x)
        x = self.avgPool1(x)
        x = self.block1(x)
        x = self.avgPool2(x)
        x = self.block2_1(x)
        x = self.avgPool3(x)
        x = self.block2_2(x)
        x = self.avgPool4(x)
        x = x.view(-1, 850)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

    
class HyperSpectrumModel4Layers(nn.Module): # kernel_size=3, stride=3
    def __init__(self, dropout_percent=0.0, kernel_size=3, stride=3, stddev = 0.0):
        super(HyperSpectrumModel4Layers, self).__init__()
        self.noise = GaussianNoise(stddev=stddev)
        self.avgPool1 = torch.nn.AvgPool1d(kernel_size=5, stride=5, padding=0) # 430 
        self.block1 = DeepBasicBlock(inchannels=1, outchannels=50, kernel_size=5, stride=5, padding=1)#86x50
        self.block2_1 = DeepBasicBlock(50, 100, 3, 3, 0) # 28x50
        self.block2_2 = DeepBasicBlock(100, 200, 3, 3, 0) # 9x100
        self.block3_1 = DeepBasicBlock(200, 400, 3, 3, 0) #3x400
        
        self.fc1 = nn.Linear(1200, 800)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_percent)
        self.fc2 = nn.Linear(800, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_percent)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = x.view(-1, 1, 2151)
        x = self.noise(x)
        x = self.avgPool1(x)
        x = self.block1(x)
        x = self.block2_1(x)
        x = self.block2_2(x)
        x = self.block3_1(x)
       
        x = x.view(-1, 1200)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
     
class HyperSpectrumModelAvgPoolDeepCNN_ConvNet(nn.Module): # kernel_size=3, stride=3
    def __init__(self, dropout_percent=0.0, kernel_size=3, stride=3):
        super(HyperSpectrumModelAvgPoolDeepCNN_ConvNet, self).__init__()
        #self.noise1 = GaussianNoise(stddev=0.0001)
        self.avgPool1 = torch.nn.AvgPool1d(kernel_size=5, stride=5, padding=2) # 431x1 
        self.block1 = DeepBasicBlock(inchannels=1, outchannels=100, kernel_size=5, stride=1, padding=2)#431
        self.avgPool2 = torch.nn.AvgPool1d(kernel_size=3, stride=3, padding=0) # 143x1
        self.block2_1 = DeepBasicBlock(100, 200, 3, 1, 1) # 143x50
        self.avgPool3 = torch.nn.AvgPool1d(kernel_size=3, stride=3, padding=0) # 47x1
        self.block2_2 = DeepBasicBlock(200, 400, 3, 1, 1) # 47x200
        self.avgPool4 = torch.nn.AvgPool1d(kernel_size=3, stride=3, padding=0) #15x200
                       
        self.fc1 = nn.Linear(6000, 1000)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_percent)
        self.fc2 = nn.Linear(1000, 500)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_percent)
        self.fc3 = nn.Linear(500, 1)

    def forward(self, x):
        x = x.view(-1, 1, 2151)
        #x = self.noise1(x)
        x = self.avgPool1(x)
        x = self.block1(x)
        x = self.avgPool2(x)
        x = self.block2_1(x)
        x = self.avgPool3(x)
        x = self.block2_2(x)
        x = self.avgPool4(x)
        x = x.view(-1, 6000)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x   
class HyperSpectrumModelAvgPoolDeepCNN_New(nn.Module): # kernel_size=3, stride=3
    def __init__(self, dropout_percent=0.0, kernel_size=3, stride=3):
        super(HyperSpectrumModelAvgPoolDeepCNN_New, self).__init__()
        #self.noise1 = GaussianNoise(stddev=0.0001)
        self.avgPool1 = torch.nn.AvgPool1d(kernel_size=5, stride=5, padding=2) # 717x1 
        self.block1 = DeepBasicBlock(inchannels=1, outchannels=100, kernel_size=5, stride=5, padding=0)#358
        self.block2_1 = DeepBasicBlock(100, 200, 3, 3, 0) # 358x50
        self.block2_2 = DeepBasicBlock(200, 200, 3, 3, 0) # 9x200
                       
        self.fc1 = nn.Linear(1800, 1000)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_percent)
        self.fc2 = nn.Linear(1000, 500)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_percent)
        self.fc3 = nn.Linear(500, 1)

    def forward(self, x):
        x = x.view(-1, 1, 2151)
        #x = self.noise1(x)
        x = self.avgPool1(x)
        x = self.block1(x)
        #x = self.avgPool2(x)
        x = self.block2_1(x)
        x = self.block2_2(x)

        x = x.view(-1, 1800)
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
class HyperSpectrumModelAvgPoolDeepCNNV3(nn.Module):
    def __init__(self, dropout_percent=0.0, kernel_size=2, stride=2):
        super(HyperSpectrumModelAvgPoolDeepCNNV3, self).__init__()
        self.avgPool1 = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # 717x1 
        self.block1 = DeepBasicBlock(inchannels=1, outchannels=32, kernel_size=kernel_size, stride=stride, padding=1)#358       
        self.avgPool2 = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # 717x1 
        
        self.block2_1 = DeepBasicBlock(32, 32, kernel_size, stride, 1) # 358x50
        #self.avgPool3 = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # 717x1 
        
        self.block2_2 = DeepBasicBlock(32, 64, kernel_size, stride, 1) # 179x100
        self.avgPool4= torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # 717x1 
        
        self.block3_1 = DeepBasicBlock(64, 64, kernel_size, stride, 1)
        #self.avgPool5= torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # 717x1 
        
        self.block3_2 = DeepBasicBlock(64, 128, kernel_size, stride, 1)
        self.avgPool6 = torch.nn.AvgPool1d(kernel_size=2, stride=2, padding=0) # 717x1 
        
        #self.block3_3 = DeepBasicBlock(160, 320, kernel_size=kernel_size, 3, 0)
        

        self.fc1 = nn.Linear(640, 500)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_percent)
        self.fc2 = nn.Linear(500, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_percent)
        self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = x.view(-1, 1, 2151)
        x = self.avgPool1(x)
        x = self.block1(x)
        x = self.avgPool2(x)
        
        x = self.block2_1(x)
       # x = self.avgPool3(x)
        
        x = self.block2_2(x)
        x = self.avgPool4(x)
        
        x = self.block3_1(x) 
        #x = self.avgPool5(x)
        
        x = self.block3_2(x)
        x = self.avgPool6(x)
        #x = self.block3_3(x)
        #x = self.avgPool6(x)
        x = x.view(-1, 640)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    

class HyperSpectrumModel(nn.Module):
    def __init__(self, dropout_percent=0.0):
        super(HyperSpectrumModel, self).__init__()

        self.block1 = BasicBlock(1, 10, 2, 1) # 2151 -> 1076 -> 538
        self.fc1 = nn.Linear(538 * 10, 200)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(200,1)
        self.dropout = nn.Dropout(p=dropout_percent)

    def forward(self, x):
        x = x.view(-1, 1, 2151)
        x = self.block1(x)
        x = x.view(-1, 538 * 10)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class HyperSpectrumModelAvgPool(nn.Module):
    def __init__(self, dropout_percent=0.0):
        super(HyperSpectrumModelAvgPool, self).__init__()

        self.block1 = BasicBlock(1, 10, 2, 1) # 431 -> 216 -> 108
        self.avgPool = torch.nn.AvgPool1d(5, 5, 2)
        self.fc1 = nn.Linear(108 * 10, 200)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(200,100)
        self.dropout = nn.Dropout(p=dropout_percent)
        self.fc3 = nn.Linear(100,1)


    def forward(self, x):
        x = x.view(-1, 1, 2151)
        x = self.avgPool(x)
        x = self.block1(x)
        x = x.view(-1, 108 * 10)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


    
class HyperSpectrumModelAvgPoolDeepCNN(nn.Module):
    def __init__(self, dropout_percent=0.0):
        super(HyperSpectrumModelAvgPoolDeepCNN, self).__init__()

        self.block1 = BasicBlock(1, 24, 2, 1)
        self.avgPool1 = torch.nn.AvgPool1d(5, 5, 2)
        
        self.block2 = BasicBlock(24, 24, 2, 1)
        self.avgPool2 = torch.nn.AvgPool1d(5, 5, 2)
        
        self.fc1 = nn.Linear(144, 144)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_percent)
        
        self.fc2 = nn.Linear(144, 100)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=dropout_percent)

        self.fc3 = nn.Linear(100,1)

    def forward(self, x):
        x = x.view(-1, 1, 2151)
        x = self.block1(x)
        x = self.avgPool1(x)
        x = self.block2(x)
        x = self.avgPool2(x)
        x = x.view(-1, 144)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    