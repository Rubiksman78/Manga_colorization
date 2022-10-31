from scripts.dataset import ImageDataset
#from networks.cyclegan import *
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import ImageFile
from config import DEFAULT_CONFIG
from scripts.utils import *
import wandb 
from scripts.train import train_cycle_gan,infer, train_pixpix
from networks.perceptual_loss import VGGPerceptualLoss
from networks.pix2pix import ResNetPix2Pix
from networks.unetpix2pix import UNet,UNetPix2Pix
from networks.disc_net import Discriminator

if __name__ == "__main__":
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    (WIDTH,
     HEIGHT,
     BATCH_SIZE,
     LR,
     DATASET,
     EPOCHS,
     SAVE_INTERVALL,
     N_RESNET,
     CYCLE_WEIGHT,
     ID_WEIGHT,
     ID) = (DEFAULT_CONFIG["WIDTH"],
                    DEFAULT_CONFIG["HEIGHT"],
                    DEFAULT_CONFIG["BATCH_SIZE"],
                    DEFAULT_CONFIG["LR"],
                    DEFAULT_CONFIG["DATASET"],
                    DEFAULT_CONFIG["EPOCHS"],
                    DEFAULT_CONFIG["SAVE_INTERVALL"],
                    DEFAULT_CONFIG["N_RESNET"],
                    DEFAULT_CONFIG["CYCLE_WEIGHT"],
                    DEFAULT_CONFIG["ID_WEIGHT"],
                    DEFAULT_CONFIG["ID"])
     
    wandb.init(project='Manga_color',config=DEFAULT_CONFIG,name=f"test{ID}",mode='disabled')
    use_cyclegan = False

    if use_cyclegan:
        print("Cycle GAN is training")
        dataset = ImageDataset(
        DATASET, 
        transform=transforms.Compose([
            transforms.Resize((WIDTH,HEIGHT)),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ]),
            unaligned=False,)

        create_folders_id(f"weights/{ID}")
        create_folders_id(f"results/{ID}")
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,drop_last=True)
        genB2A = Generator(3,3,N_RESNET).to(device)
        genA2B = Generator(3,3,N_RESNET).to(device)
        disc1 = Discriminator(3).to(device)
        disc2 = Discriminator(3).to(device)

        cycle_crit = nn.L1Loss().to(device)
        identity_crit = nn.L1Loss().to(device)
        adversarial_crit = nn.BCEWithLogitsLoss().to(device)

        optG1 = torch.optim.Adam(genB2A.parameters(),lr=LR,betas=(0.5,0.999))
        optG2 = torch.optim.Adam(genA2B.parameters(),lr=LR,betas=(0.5,0.999))
        optD1 = torch.optim.Adam(disc1.parameters(),lr=LR,betas=(0.5,0.999))
        optD2 = torch.optim.Adam(disc2.parameters(),lr=LR,betas=(0.5,0.999))
        schedulers = [torch.optim.lr_scheduler.ExponentialLR(optG1,gamma=0.9),
                    torch.optim.lr_scheduler.ExponentialLR(optG2,gamma=0.9),
                    torch.optim.lr_scheduler.ExponentialLR(optD1,gamma=0.9),
                    torch.optim.lr_scheduler.ExponentialLR(optD2,gamma=0.9)]
        train_cycle_gan(
            EPOCHS,
            dataloader,
            genB2A,
            genA2B,
            disc1,
            disc2,
            cycle_crit,
            identity_crit,
            adversarial_crit,
            optG1,
            optG2,
            optD1,
            optD2,
            schedulers
            )
    else:
        print("Pix2Pix is training")
        #Data augmnentation
        transformations = transforms.Compose([
            transforms.Resize((WIDTH,HEIGHT)),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))
            ])

        dataset = ImageDataset(
                        DATASET, 
                        transform=transformations,
                        unaligned=False,)
        #dataset = torch.utils.data.Subset(dataset,range(100))
        #Train,test split
        train_size = int(0.9 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        #Create dataloaders
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,pin_memory=True,drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)
        create_folders_id(f"weights/pix2pix/{ID}")
        create_folders_id(f"results/pix2pix/{ID}")
        #Check shape of data
        for i in test_dataloader:
            print(i["A"].shape,i["B"].shape)
            break
        #gen = Generator(3,3,N_RESNET).to(device)
        gen = UNet(1,3).to(device)
        disc = Discriminator(3).to(device)
        
        gen.load_state_dict(torch.load("2/gen1_11.pt"))
        disc.load_state_dict(torch.load("2/disc1_11.pt"))

        pix2pixmodel = UNetPix2Pix(gen,disc)
        #pix2pixmodel.show_model_summary()
        adversarial_crit = nn.BCEWithLogitsLoss().to(device)
        feature_loss = VGGPerceptualLoss().to(device)

        optG = torch.optim.Adam(gen.parameters(),lr=LR,betas=(0.5,0.999))
        optD = torch.optim.Adam(disc.parameters(),lr=LR,betas=(0.5,0.999))
        schedulers = [torch.optim.lr_scheduler.ExponentialLR(optG,gamma=0.9),
                    torch.optim.lr_scheduler.ExponentialLR(optD,gamma=0.9)]
        
        testing_dataset = ImageDataset(
                        "image_data",
                        transform=transformations,
                        unaligned=False,)
        dataloader = torch.utils.data.DataLoader(testing_dataset,batch_size=8,shuffle=True)
        data1 = next(iter(dataloader))["A"].to(device)
        data2 = next(iter(dataloader))["B"].to(device)
        plot_test_pix2pix(gen,data2,data1,epoch=10000,n_gen=4,save=False)
        """
        train_pixpix(
            pix2pixmodel,
            EPOCHS,
            train_dataloader,
            gen,
            disc,
            adversarial_crit,
            feature_loss,
            optG,
            optD,
            schedulers,
            test_dataloader
        )
        """
        