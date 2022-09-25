from networks.cyclegan import *
import torch
from tqdm import tqdm
from scripts.utils import *
import wandb 
from networks.pix2pix import *

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
            
def train_cycle_gan(
        epochs,
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
        ):
    for epoch in range(epochs):
        noise_var = 0.1
        progress_bar = tqdm(enumerate(dataloader),total=len(dataloader))
        for _,data in progress_bar:
            data1 = data["A"].to(device)
            data2 = data["B"].to(device)
            data1 = data1 + torch.randn_like(data1)*noise_var
            data2 = data2 + torch.randn_like(data2)*noise_var
            noise_var = noise_var*0.99
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
                device,
                cycle_weight=CYCLE_WEIGHT,
                id_weight=ID_WEIGHT)
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
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
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
        plot_test(genB2A,genA2B,data1,data2,epoch,n_gen=4,save=False)
        for scheduler in schedulers:
            scheduler.step()
        if epoch % SAVE_INTERVALL == 0:
            torch.save(genB2A.state_dict(),f"weights/{ID}/gen1_{epoch+1}.pt")
            torch.save(genA2B.state_dict(),f"weights/{ID}/gen2_{epoch+1}.pt")
            torch.save(disc1.state_dict(),f"weights/{ID}/disc1_{epoch+1}.pt")
            torch.save(disc2.state_dict(),f"weights/{ID}/disc2_{epoch+1}.pt")
            plot_test(genB2A,genA2B,data1,data2,epoch,n_gen=4,save=True)
            
def infer(data1,data2,n_gen,checkpoint=11):
    genB2A = Generator(3,3,N_RESNET)
    genA2B = Generator(3,3,N_RESNET)
    #genB2A.load_state_dict(torch.load(f"weights/{ID}/gen1_{checkpoint}.pt",map_location=torch.device("cpu")))
    genA2B.load_state_dict(torch.load(f"weights/{ID}/gen2_{checkpoint}.pt",map_location=torch.device("cpu")))
    plot_test(genB2A,genA2B,data1,data2,0,n_gen,save=False)
    
def train_pixpix(
        model,
        epochs,
        dataloader,
        gen,
        disc,
        adv_criterion,
        feat_criterion,
        optG,
        optD,
        schedulers,
        ):
    for epoch in range(epochs):
        noise_var = 0.1
        progress_bar = tqdm(enumerate(dataloader),total=len(dataloader))
        for _,data in progress_bar:
            data1 = data["B"].to(device)
            data2 = data["A"].to(device)
            data1 = data1 + torch.randn_like(data1)*noise_var
            data2 = data2 + torch.randn_like(data2)*noise_var
            noise_var = noise_var*0.99
            losses = model.train_step_pix2pix(
                data1,
                data2,
                optG,
                optD,
                adv_criterion,
                feat_criterion,
                feat_weight=10
            )
            feature_loss, adversial_loss,disc_loss = losses[0],losses[1],losses[2]
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
            dict = {
                "Feature_Loss":feature_loss,
                "Adversial_Loss":adversial_loss,
                "Discriminator_Loss":disc_loss
            }
            progress_bar.set_postfix({k:f"{v:.4f}" for k,v in dict.items()})
            wandb.log(dict)
        plot_test_pix2pix(gen,data1,data2,epoch,n_gen=4,save=True)
        for scheduler in schedulers:
            scheduler.step()
        if epoch % SAVE_INTERVALL == 0:
            torch.save(gen.state_dict(),f"weights/pix2pix/{ID}/gen1_{epoch+1}.pt")
            torch.save(disc.state_dict(),f"weights/pix2pix/{ID}/disc1_{epoch+1}.pt")
            plot_test_pix2pix(gen,data1,data2,epoch,n_gen=4,save=True)
