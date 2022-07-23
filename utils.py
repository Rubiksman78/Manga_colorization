import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb

def my_collate(batch,dataset):
    len_batch = len(batch) # original batch length
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    if len_batch > len(batch): # source all the required samples from the original dataset at random
        diff = len_batch - len(batch)
        for i in range(diff):
            batch.append(dataset[np.random.randint(0, len(dataset))])

    return torch.utils.data.dataloader.default_collate(batch)

#This function generates a batch of images from the generator 1 and 2 then plot them on a figure
def plot_test(gen1,gen2,data1,data2,epoch):
    n = min(len(data1),6)
    fig = plt.figure(figsize=(12,12))
    data1 = data1[:n]
    data2 = data2[:n]
    for i in range(len(data1)):
        im = data1[i]
        fake_1 = gen1(im)*0.5+0.5
        fake_1 = fake_1.detach().cpu().numpy()
        im = im.detach().cpu().numpy()*0.5+0.5
        plt.subplot(n,2,2*i+1)
        plt.imshow(im.transpose(1,2,0))
        plt.axis("off")
        plt.subplot(n,2,2*i+2)
        plt.imshow(fake_1.transpose(1,2,0))
        plt.axis("off")
    wandb.log({"Grey_images": plt})
    plt.savefig(f"results/{epoch+1}.png")
    plt.close()
    fig = plt.figure(figsize=(12,12))
    for i in range(len(data2)):
        im = data2[i]
        fake_2 = gen2(im)*0.5+0.5
        fake_2 = fake_2.detach().cpu().numpy()
        im = im.detach().cpu().numpy()*0.5+0.5
        plt.subplot(n,2,2*i+1)
        plt.imshow(im.transpose(1,2,0))
        plt.axis("off")
        plt.subplot(n,2,2*i+2)
        plt.imshow(fake_2.transpose(1,2,0))
        plt.axis("off")
    wandb.log({"Color_images": plt})
    plt.savefig(f"results/{epoch+1}_2.png")
    plt.close()
    