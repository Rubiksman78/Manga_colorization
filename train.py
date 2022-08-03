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
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ]),
        unaligned=False,)

my_collate = lambda x: my_collate(x,dataset)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,drop_last=True)
genB2A = Generator(3,3,9).to(device)
genA2B = Generator(3,3,9).to(device)
disc1 = Discriminator(3).to(device)
disc2 = Discriminator(3).to(device)

cycle_crit = nn.L1Loss().to(device)
identity_crit = nn.L1Loss().to(device)
adversarial_crit = nn.BCEWithLogitsLoss().to(device)

optG1 = torch.optim.Adam(genB2A.parameters(),lr=LR,betas=(0.5,0.999))
optG2 = torch.optim.Adam(genA2B.parameters(),lr=LR,betas=(0.5,0.999))
optD1 = torch.optim.Adam(disc1.parameters(),lr=LR,betas=(0.5,0.999))
optD2 = torch.optim.Adam(disc2.parameters(),lr=LR,betas=(0.5,0.999))

def train(epochs):
    for epoch in range(epochs):
        progress_bar = tqdm(enumerate(dataloader),total=len(dataloader))
        for i,data in progress_bar:
            data1 = data["B"].to(device)
            data2 = data["A"].to(device)
            #print(data1.detach().cpu().numpy().max())
            #print(data1.detach().cpu().numpy().min())
            #print(data2.detach().cpu().numpy().max())
            #print(data2.detach().cpu().numpy().min())
            losses = train_step(
                BATCH_SIZE,
                genB2A,
                genA2B,
                disc1,
                disc2,
                data1,
                data2,
                optG1,
                optG2,
                optD1,
                optD2,
                cycle_crit,
                identity_crit,
                adversarial_crit,
                device)
            (disc_loss_B2A,
             disc_loss_A2B,
             gen_loss_B2A,
             gen_loss_A2B,
             cycle_loss_B2A,
             cycle_loss_A2B,
             identity_loss_B2A,
             identity_loss_A2B,
             gan_loss_B2A,
             gan_loss_A2B) = (losses["Loss D_B2A"],
                                losses["Loss D_A2B"],
                                losses["Loss G_B2A"],
                                losses["Loss G_A2B"],
                                losses["Loss Cycle_B2A"],
                                losses["Loss Cycle_A2B"],
                                losses["Loss Identity_B2A"],
                                losses["Loss Identity_A2B"],
                                losses["Loss Gan_B2A"],
                                losses["Loss Gan_A2B"])
            #disc_loss,gen_loss,identity_loss,gan_loss,cycle_loss = losses["Loss D"],losses["Loss G"],losses["Loss Identity"],losses["Loss Gan"],losses["Loss Cycle"]
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
            #dict = {"Discriminator_Loss":disc_loss,"Generator_Loss":gen_loss,"Identity_Loss":identity_loss,"Gan_Loss":gan_loss,"Cycle_Loss":cycle_loss}
            dict = {
                "Discriminator_Loss_B2A":disc_loss_B2A,
                "Discriminator_Loss_A2B":disc_loss_A2B,
                "Generator_Loss_B2A":gen_loss_B2A,
                "Generator_Loss_A2B":gen_loss_A2B,
                "Cycle_Loss_B2A":cycle_loss_B2A,
                "Cycle_Loss_A2B":cycle_loss_A2B,
                "Identity_Loss_B2A":identity_loss_B2A,
                "Identity_Loss_A2B":identity_loss_A2B,
                "Gan_Loss_B2A":gan_loss_B2A,
                "Gan_Loss_A2B":gan_loss_A2B}
            progress_bar.set_postfix({k:f"{v:.4f}" for k,v in dict.items()})
            wandb.log(dict)
        plot_test(genB2A,genA2B,data1,data2,epoch)
        torch.save(genB2A.state_dict(),f"weights/gen1_{epoch+1}.pt")
        torch.save(genA2B.state_dict(),f"weights/gen2_{epoch+1}.pt")
        torch.save(disc1.state_dict(),f"weights/disc1_{epoch+1}.pt")
        torch.save(disc2.state_dict(),f"weights/disc2_{epoch+1}.pt")

train(EPOCHS)