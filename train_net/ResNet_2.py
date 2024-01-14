import torch
import torch.nn as nn
# from pytorch_model_summary import summary
import torch.nn.functional as F


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, padding=padding),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0),
        )

    def forward(self, x):
        return F.relu(self.conv_block(x) + self.conv_skip(x))


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)



class ResUnet(nn.Module):
    # def __init__(self, in_ch, output_ch, filters=[128, 128, 256, 256, 512]):  
    def __init__(self, in_ch, output_ch, filters=[32, 64, 128, 256, 256]):  # 32, 64, 128, 256, 256 or 64, 128, 256, 512, 512
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_ch, filters[0], kernel_size=7, padding=3),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=7, padding=3),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_ch, filters[0], kernel_size=1, padding=0)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 7, 1, 3)
        self.pool_1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 3, 1, 1)
        self.pool_2 = nn.AvgPool2d(3, stride=2, padding=1)
        self.residual_conv_3 = ResidualConv(filters[2], filters[3], 3, 1, 1)
        self.pool_3 = nn.AvgPool2d(3, stride=2, padding=1)

        self.bridge = ResidualConv(filters[3], filters[4], 3, 1, 1)

        self.upsample_1 = Upsample(filters[4], filters[4], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[3], filters[3], 3, 1, 1)

        self.upsample_2 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[2], filters[2], 3, 1, 1)

        self.upsample_3 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[1], filters[1], 3, 1, 1)

        self.upsample_4 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv4 = ResidualConv(filters[1] + filters[0], filters[0], 7, 1, 3)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], output_ch, 1, 1),
        )


    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x1 = F.relu(x1)
        x2 = self.residual_conv_1(x1)
        x2 = self.pool_1(x2)
        x3 = self.residual_conv_2(x2)
        x3 = self.pool_2(x3)
        x4 = self.residual_conv_3(x3)
        x4 = self.pool_3(x4)
        # Bridge
        x5 = self.bridge(x4)
        # Decode
        # x5 = self.upsample_1(x5)
        x5 = torch.cat([x5, x4], dim=1)

        x5 = self.up_residual_conv1(x5)

        x5 = self.upsample_2(x5)
        x5 = torch.cat([x5, x3], dim=1)

        x5 = self.up_residual_conv2(x5)

        x5 = self.upsample_3(x5)
        x5 = torch.cat([x5, x2], dim=1)

        x5 = self.up_residual_conv3(x5)

        x5 = self.upsample_4(x5)
        x5 = torch.cat([x5, x1], dim=1)

        x5 = self.up_residual_conv4(x5)

        output = self.output_layer(x5)

        return output


class Discriminator(nn.Module):
    def __init__(self, in_ch, filters=[64, 128, 256, 128, 64, 32]):
        super(Discriminator, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_ch, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_ch, filters[0], kernel_size=3, padding=1)
        )


        self.output_layer = nn.Sequential(
            ResidualConv(filters[0], filters[1], 2, 1),
            ResidualConv(filters[1], filters[2], 2, 1),
            ResidualConv(filters[2], filters[3], 2, 1),
            ResidualConv(filters[3], filters[4], 2, 1),
            nn.Conv2d(filters[4], filters[5], 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        x = self.input_layer(x) + self.input_skip(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 256
    x = torch.Tensor(1, 29, image_size, image_size)
    # x = x.cuda()
    x = x.to(device)

    x2 = torch.Tensor(1, 18, image_size, image_size)
    # x2 = x2.cuda()
    x2 = x2.to(device)
    print("x size: {}".format(x.size()))
    
    # x = torch.Tensor(1, 100 )
    # x = x.cuda()

    model = ResUnet(in_ch = 29, output_ch = 29)
    # model.cuda()
    model.to(device)
    out = model(x)
    print("out size: {}".format(out.size()))
    # print(summary(ResUnet(in_ch = 29, output_ch = 29), torch.zeros((1, 29, image_size, image_size)), show_input=True))