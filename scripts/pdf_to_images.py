from pdf2image import convert_from_path
import os
from PIL import Image

def pdf_volume_to_images(pdf_path, images_path):
    pages = convert_from_path(pdf_path)
    for i in range(len(pages)):
        pages[i].save(images_path.split('.')[0] + 'page_' + str(i) + '.jpg','JPEG')
        
def all_volumes_to_images(pdf_path, images_path):
    for volume in os.listdir(pdf_path):
        print(pdf_path+volume)
        pdf_volume_to_images(pdf_path + volume, images_path + volume)

#open images from folder and save thel in another one
def open_images(images_path, images_path_2,volume):
    for page in os.listdir(images_path):
        im = Image.open(images_path + page)
        im.save(images_path_2 + 'hod_' + volume + '_' + page)
            
if __name__ == '__main__':
    pdf_path = '../highschoolofthedead/'
    images_path = 'image_data_OP/'
    #all_volumes_to_images(pdf_path, images_path)
    for id in range(1,8):
        open_images(f"../highschoolofthedead/vol{id}/",f"image_data/B/",f'volume_{id}')