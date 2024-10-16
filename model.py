import torch
import torch.nn as nn
import torchvision.transforms.functional as TF 

class DoubleConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # First Conv
            nn.Conv2d(input_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            # Second Conv
            nn.Conv2d(output_channels, output_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, features=[64, 128, 256, 512],):
        super(UNET, self).__init__()
        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part for UNET
        for feature in features:
            self.downs.append(DoubleConv(input_channels, feature))
            input_channels = feature
        
        # Up part for UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2, 
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
            
        self.bottleNeck = DoubleConv(features[-1], features[-1] * 2)
        self.finalConv = nn.Conv2d(features[0], output_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        x = self.bottleNeck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            # Resizing so that the skip connection match the size 
            # of the feature map when upsampling
            if x.shape != skip_connection.shape:
                # print(skip_connection.shape)
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            
        return self.finalConv(x)
            
def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNET(input_channels=1, output_channels=1)
    preds = model(x)
    
    print(preds.shape)
    print(x.shape)
    # assert preds.shape == x.shape 
    
if __name__ == "__main__":
    test()