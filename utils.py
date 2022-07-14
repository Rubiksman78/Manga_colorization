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
    n = len(data1)
    fig = plt.figure(figsize=(12,12))
    for i,im in enumerate(data1):
        fake_1 = gen1(im)*0.5+0.5
        fake_1 = fake_1.detach().cpu().numpy()
        im = im.detach().cpu().numpy()*0.5+0.5
        plt.subplot(n,2,2*i+1)
        plt.imshow(im.transpose(1,2,0))
        plt.title("Real")
        plt.axis("off")
        plt.subplot(n,2,2*i+2)
        plt.imshow(fake_1.transpose(1,2,0))
        plt.title("Fake")
        plt.axis("off")
    plt.savefig(f"results/{epoch+1}.png")
    plt.close()
    fig = plt.figure(figsize=(12,12))
    for i,im in enumerate(data2):
        fake_2 = gen2(im)*0.5+0.5
        fake_2 = fake_2.detach().cpu().numpy()
        im = im.detach().cpu().numpy()*0.5+0.5
        plt.subplot(n,2,2*i+1)
        plt.imshow(im.transpose(1,2,0))
        plt.title("Real")
        plt.axis("off")
        plt.subplot(n,2,2*i+2)
        plt.imshow(fake_2.transpose(1,2,0))
        plt.title("Fake")
        plt.axis("off")
    wandb.log({"test_image": plt})
    plt.savefig(f"results/{epoch+1}_2.png")
    plt.close()
    