import torch 
import torch.nn as nn
from torchsummary import summary
from config import DEFAULT_CONFIG
from disc_net import Discriminator

WIDTH,HEIGHT = DEFAULT_CONFIG["WIDTH"],DEFAULT_CONFIG["HEIGHT"]
print_model = DEFAULT_CONFIG["PRINT_MODEL"]
       
class Resnet_Block(nn.Module):
    def __init__(self,n_filters):
        super(Resnet_Block, self).__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(n_filters)
        self.in2 = nn.InstanceNorm2d(n_filters)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        return y + x

class Generator(nn.Module):
    def __init__(self,input_nc,output_nc,ngf=64,n_resnet=9):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]
        mult = 2**n_downsampling
        for i in range(n_resnet):
            model += [Resnet_Block(ngf * mult)]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult,
                                         int(ngf * mult / 2),
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)

if print_model:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gen = Generator(3,3,n_resnet=9).to(device)
    print(summary(gen,(3,HEIGHT,WIDTH)))
    dis = Discriminator(3).to(device)
    print(summary(dis,(3,HEIGHT,WIDTH)))

class ResNetPix2Pix:
    def __init__(self,lr=0.0002,beta1=0.5,beta2=0.999):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(3,3,n_resnet=9).to(self.device)
        self.discriminator = Discriminator(3).to(self.device)

    def train_step_pix2pix(
            gen,
            disc,
            dataA,
            dataB,
            gen_opt,
            disc_opt,
            adv_loss,
            feat_loss,
            device,
            feat_weight,
            ):
        gen_opt.zero_grad()
        b_generated = gen(dataA)
        b_generated_score = disc(b_generated)
        real_labels = torch.ones(b_generated_score.size()).to(device)
        adversial_loss =  adv_loss(b_generated_score,real_labels)
        feature_loss = feat_loss(b_generated,dataB) * feat_weight
        gen_loss = adversial_loss + feature_loss
        gen_loss.backward()
        gen_opt.step()

        disc_opt.zero_grad()
        b_real = dataB
        b_real_score = disc(b_real)
        b_generated = gen(dataA)
        b_generated_score = disc(b_generated)
        fake_labels = torch.zeros(b_generated_score.size()).to(device)
        real_labels = torch.ones(b_real_score.size()).to(device)
        disc_loss =  adv_loss(b_generated_score,fake_labels) + adv_loss(b_real_score,real_labels)
        disc_loss.backward()
        disc_opt.step()

        return adversial_loss.item(), feature_loss.item(), disc_loss.item()



