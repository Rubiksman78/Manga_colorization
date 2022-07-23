import torch 
import torch.nn as nn
from torchsummary import summary

class Discriminator(nn.Module):
    def __init__(self,input_nc,ndf=64,n_layers=3,norm_layer=nn.BatchNorm2d,use_sigmoid = False):
        super(Discriminator, self).__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)
        
        if use_sigmoid:
            self.model.add_module('sigmoid', nn.Sigmoid())
            
    def forward(self, input):
        return self.model(input)
       
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
    def __init__(self,input_nc,output_nc,ngf=64,n_resnet=6):
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
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gen = Generator(3,3,n_resnet=8).to(device)
print(summary(gen,(3,128,204)))
dis = Discriminator(3).to(device)
print(summary(dis,(3,128,204)))

def train_step(batch_size,gen1,gen2,disc1,disc2,data1,data2,gen1_optim,disc1_optim,gen2_optim,disc2_optim,cycle_loss,identity_loss,adversarial_loss,device):
    #Gen1 = genB2A, Gen2 = genA2B
    #data1 = A, data2 = B
    #Update gen1 and gen2
    gen1_optim.zero_grad()
    gen2_optim.zero_grad()

    #Identity loss
    identity_image_A = gen1(data1)
    loss_identity_A = identity_loss(identity_image_A,data1) 
    identity_image_B = gen2(data2)
    loss_identity_B = identity_loss(identity_image_B,data2) 

    #Gan loss
    fake_image_A = gen1(data2)
    fake_output_A = disc1(fake_image_A)
    
    #Labels like disc output
    real_label = torch.ones_like(fake_output_A).to(device)
    fake_label = torch.zeros_like(fake_output_A).to(device)
    
    loss_gan_1 = adversarial_loss(fake_output_A,real_label)
    fake_image_B = gen2(data1)
    fake_output_B = disc2(fake_image_B)
    loss_gan_2 = adversarial_loss(fake_output_B,real_label)

    #Cycle loss
    recovered_image_A = gen1(fake_image_B)
    loss_cycle_A = cycle_loss(recovered_image_A,data1) 
    recovered_image_B = gen2(fake_image_A)
    loss_cycle_B = cycle_loss(recovered_image_B,data2) 

    errG = loss_identity_A + loss_identity_B + loss_gan_1 + loss_gan_2 + loss_cycle_A + loss_cycle_B
    errG.backward()
    gen1_optim.step()
    gen2_optim.step()

    #Update disc1
    disc1_optim.zero_grad()
    real_output_A = disc1(data1)
    loss_real_A = adversarial_loss(real_output_A,real_label)
    fake_image_A = gen1(data2)
    fake_output_A = disc1(fake_image_A)
    loss_fake_A = adversarial_loss(fake_output_A,fake_label)
    errD_A = (loss_real_A + loss_fake_A)/2
    errD_A.backward()
    disc1_optim.step()

    #Update disc2
    disc2_optim.zero_grad()
    real_output_B = disc2(data2)
    loss_real_B = adversarial_loss(real_output_B,real_label)
    fake_image_B = gen2(data1)
    fake_output_B = disc2(fake_image_B)
    loss_fake_B = adversarial_loss(fake_output_B,fake_label)
    errD_B = (loss_real_B + loss_fake_B)/2
    errD_B.backward()
    disc2_optim.step()
    return {"Loss D":errD_A.item() + errD_B.item(),"Loss G":errG.item(),"Loss Identity":loss_identity_A.item() + loss_identity_B.item(),"Loss Gan":loss_gan_1.item() + loss_gan_2.item(),"Loss Cycle":loss_cycle_A.item() + loss_cycle_B.item()}
