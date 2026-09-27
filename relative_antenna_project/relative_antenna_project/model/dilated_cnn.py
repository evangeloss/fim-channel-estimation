from torch import nn


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation))
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.layers(x))


class SharedObservationCNN(nn.Module):
    """Shared processing of [reference, difference] for two subcarriers."""
    def __init__(self, feature_channels=64, dilations=(1,2,4), input_channels=8):
        super().__init__()
        if feature_channels < 1 or not dilations or any(d < 1 for d in dilations):
            raise ValueError('Positive channels and dilations required.')
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels, 3, padding=1), nn.GELU(),
            *[DilatedResidualBlock(feature_channels,d) for d in dilations])

    def forward(self, x):
        b,m,c,nb,nu = x.shape
        f = self.net(x.reshape(b*m,c,nb,nu))
        return f.reshape(b,m,-1,nb,nu)
