import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward

import High_Frequecy_network
from Low_frequency_network import low_frequency_network



class Down_wt(nn.Module):
    def __init__(self):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        return yL, y_HL + y_LH + y_HH

class FusionModule(nn.Module):
    def __init__(self):
        super(FusionModule, self).__init__()
        self.wfd = Down_wt()
        self.low_model = low_frequency_network(num_classes=8,include_top=True)
        self.high_model= High_Frequecy_network.High_Frequency_network()
        self.fc = nn.Linear(in_features=2560, out_features=8)




    def forward(self,x):
        x1,x2= self.wfd(x)
        x1 = self.low_model(x1)
        x2 = self.high_model(x2)
        out = torch.cat((x1,x2),1)
        out = self.fc(out)
        return out

def WaveNet_SF():
    return FusionModule()



if __name__ == '__main__':

    x = torch.ones(32, 3, 448, 448).to('cuda:0')
    model = WaveNet_SF().to('cuda:0')
    y = model(x)
    print(y)




