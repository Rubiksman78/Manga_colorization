from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#load_image and plot it
def load_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((252,400))
    img = np.array(img)
    img = img/255.0
    plt.imshow(img,cmap='gray')
    plt.show()
    
load_image('image_data/B/volume_1page_12.jpg')