import torch
import torch.nn as nn
#from torchsummary import summary

class NvidiaModel(nn.Module):

    def __init__(self):
        super(NvidiaModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers_angle = nn.Sequential(
            nn.Linear(in_features=64 * 8 * 13, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

        self.linear_layers_throttle = nn.Sequential(
            nn.Linear(in_features=64 * 8 * 13, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )
        
    def forward(self, input):  
        input = input.view(input.size(0), 3, 160, 100)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output1 = self.linear_layers_angle(output)
        output2 = self.linear_layers_throttle(output)
        return output1, output2

def visualize_model(model):
    model = NvidiaModel()
    model.cuda()
    #return summary(model, (3, 120, 160))
# visualize_model(NvidiaModel)