from dataset import ImageDataset
from model import *
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import ImageFile
from config import DEFAULT_CONFIG
from utils import *
import wandb 

wandb.init(project='Manga_color',config=DEFAULT_CONFIG,name='test1',mode='disabled')
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
WIDTH,HEIGHT,BATCH_SIZE,LR,DATASET,EPOCHS = DEFAULT_CONFIG["WIDTH"],DEFAULT_CONFIG["HEIGHT"],DEFAULT_CONFIG["BATCH_SIZE"],DEFAULT_CONFIG["LR"],DEFAULT_CONFIG["DATASET"],DEFAULT_CONFIG["EPOCHS"]

dataset = ImageDataset(
    DATASET, 
    transform=transforms.Compose([
        transforms.Resize((WIDTH,HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
        unaligned=False,)

my_collate = lambda x: my_collate(x,dataset)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,drop_last=True)
gen1 = Generator(3,3,8).to(device)
gen2 = Generator(3,3,8).to(device)
disc1 = Discriminator(3).to(device)
disc2 = Discriminator(3).to(device)

cycle_crit = nn.L1Loss().to(device)
identity_crit = nn.L1Loss().to(device)
adversarial_crit = nn.MSELoss().to(device)

optG1 = torch.optim.Adam(gen1.parameters(),lr=LR,betas=(0.5,0.999))
optG2 = torch.optim.Adam(gen2.parameters(),lr=LR,betas=(0.5,0.999))
optD1 = torch.optim.Adam(disc1.parameters(),lr=LR,betas=(0.5,0.999))
optD2 = torch.optim.Adam(disc2.parameters(),lr=LR,betas=(0.5,0.999))

def train(epochs):
    identity_losses = []
    gan_losses = []
    cycle_losses = []
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(dataloader),total=len(dataloader))
        for i,data in progress_bar:
            data1 = data["A"].to(device)
            data2 = data["B"].to(device)
            losses = train_step(BATCH_SIZE,gen1,gen2,disc1,disc2,data1,data2,optG1,optG2,optD1,optD2,cycle_crit,identity_crit,adversarial_crit,device)
            disc_loss,gen_loss,identity_loss,gan_loss,cycle_loss = losses["Loss D"],losses["Loss G"],losses["Loss Identity"],losses["Loss Gan"],losses["Loss Cycle"]
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
            dict = {"Discriminator_Loss":disc_loss,"Generator_Loss":gen_loss,"Identity_Loss":identity_loss,"Gan_Loss":gan_loss,"Cycle_Loss":cycle_loss}
            progress_bar.set_postfix({"Discriminator Loss":f"{disc_loss:.4f}","Generator Loss":f"{gen_loss:.4f}","Identity Loss":f"{identity_loss:.4f}","Gan Loss":f"{gan_loss:.4f}","Cycle Loss":f"{cycle_loss:.4f}"})
            wandb.log(dict)
            identity_losses.append(identity_loss)
            gan_losses.append(gan_loss)
            cycle_losses.append(cycle_loss)
        plot_test(gen1,gen2,data1,data2,epoch)
        torch.save(gen1.state_dict(),f"weights/gen1_{epoch+1}.pt")
        torch.save(gen2.state_dict(),f"weights/gen2_{epoch+1}.pt")
        torch.save(disc1.state_dict(),f"weights/disc1_{epoch+1}.pt")
        torch.save(disc2.state_dict(),f"weights/disc2_{epoch+1}.pt")

train(EPOCHS)