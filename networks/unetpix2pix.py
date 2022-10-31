import torch.nn as nn
import torch.nn.functional as F
import torch
from networks.disc_net import Discriminator
import torchsummary
from config import DEFAULT_CONFIG

WIDTH,HEIGHT = DEFAULT_CONFIG["WIDTH"],DEFAULT_CONFIG["HEIGHT"]

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        super(DoubleConv,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )

    def forward(self,x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        super(Up,self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = DoubleConv(in_channels,out_channels,in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2)
            self.conv = DoubleConv(in_channels,out_channels)
        
    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])
        x = torch.cat([x2,x1],dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self,x):
        return self.tanh(self.conv(x))

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,gf=32,bilinear=False):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.gf = gf

        self.inc = DoubleConv(n_channels,gf)
        self.down1 = Down(gf,gf*2)
        self.down2 = Down(gf*2,gf*4)
        self.down3 = Down(gf*4,gf*8)
        self.down4 = Down(gf*8,gf*16)
        self.up1 = Up(gf*16,gf*8,bilinear)
        self.up2 = Up(gf*8,gf*4,bilinear)
        self.up3 = Up(gf*4,gf*2,bilinear)
        self.up4 = Up(gf*2,gf,bilinear)
        self.outc = OutConv(gf,n_classes)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5,x4)
        x = self.up2(x4,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        logits = self.outc(x)
        return logits

class UNetPix2Pix:
    def __init__(self,gen,disc):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = gen.to(self.device)
        self.disc = disc.to(self.device)

    def show_model_summary(self):
        torchsummary.summary(self.gen,(1,WIDTH,HEIGHT))
        torchsummary.summary(self.disc,(3,WIDTH,HEIGHT))

    def train_step_pix2pix(
            self,
            dataA,
            dataB,
            gen_opt,
            disc_opt,
            adv_loss,
            feat_loss,
            feat_weight,
            ):
        gen_opt.zero_grad()
        b_generated = self.gen(dataA)
        b_generated_score = self.disc(b_generated)
        real_labels = torch.ones(b_generated_score.size()).to(self.device)
        adversial_loss =  adv_loss(b_generated_score,real_labels)
        feature_loss = feat_loss(b_generated,dataB) * feat_weight
        gen_loss = adversial_loss + feature_loss
        gen_loss.backward()
        gen_opt.step()

        disc_opt.zero_grad()
        b_real = dataB
        b_real_score = self.disc(b_real)
        b_generated = self.gen(dataA)
        b_generated_score = self.disc(b_generated)
        fake_labels = torch.zeros(b_generated_score.size()).to(self.device)
        real_labels = torch.ones(b_real_score.size()).to(self.device)
        disc_loss =  adv_loss(b_generated_score,fake_labels) + adv_loss(b_real_score,real_labels)
        disc_loss.backward()
        disc_opt.step()

        return adversial_loss.item(), feature_loss.item(), disc_loss.item()