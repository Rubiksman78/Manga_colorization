import torch 
import torch.nn as nn
from torchsummary import summary

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        self.in1 = nn.InstanceNorm2d(64)
        self.in2 = nn.InstanceNorm2d(128)
        self.in3 = nn.InstanceNorm2d(256)
        self.in4 = nn.InstanceNorm2d(512)
        self.relu = nn.ReLU()
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)
        self.conv6 = nn.Conv2d(512,512,kernel_size=3,stride=2,padding=1)
        self.in5 = nn.InstanceNorm2d(512)
    
    def forward(self, x):
        in_size = x.shape[-1]
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        x = self.relu(self.in4(self.conv4(x)))
        if in_size == 128:
            x = self.relu(self.in5(self.conv6(x)))
        x = self.conv5(x)
        x = torch.flatten(x,1)
        return x

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
    def __init__(self,n_resnet):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(64)
        self.in2 = nn.InstanceNorm2d(128)
        self.in3 = nn.InstanceNorm2d(256)
        self.resnet = Resnet_Block(256)
        self.conv4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.in4 = nn.InstanceNorm2d(128)
        self.in5 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.n_resnet = n_resnet

    def forward(self, x):
        x = self.relu(self.in1(self.conv1(x)))
        x = self.relu(self.in2(self.conv2(x)))
        x = self.relu(self.in3(self.conv3(x)))
        for _ in range(self.n_resnet):
            x = self.resnet(x)
        x = self.relu(self.in4(self.conv4(x)))
        x = self.relu(self.in5(self.conv5(x)))
        x = self.tanh(self.conv6(x))
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#gen = Generator(n_resnet=8).to(device)
#print(summary(gen,(3,128,int(1.58*128))))
#dis = Discriminator().to(device)
#print(summary(dis,(3,128,int(1.58*128))))

def train_step(batch_size,gen1,gen2,disc1,disc2,data1,data2,gen1_optim,disc1_optim,gen2_optim,disc2_optim,cycle_loss,identity_loss,adversarial_loss,device):
    #Gen1 = genB2A, Gen2 = genA2B
    #data1 = A, data2 = B
    #Update gen1 and gen2
    gen1_optim.zero_grad()
    gen2_optim.zero_grad()

    #Identity loss
    identity_image_A = gen1(data1)
    loss_identity_A = identity_loss(identity_image_A,data1) * 5.0
    identity_image_B = gen2(data2)
    loss_identity_B = identity_loss(identity_image_B,data2) * 5.0

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
    loss_cycle_A = cycle_loss(recovered_image_A,data1) * 10.0
    recovered_image_B = gen2(fake_image_A)
    loss_cycle_B = cycle_loss(recovered_image_B,data2) * 10.0

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
