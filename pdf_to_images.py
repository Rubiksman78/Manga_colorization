from pdf2image import convert_from_path
import os

def pdf_volume_to_images(pdf_path, images_path):
    pages = convert_from_path(pdf_path)
    for i in range(len(pages)):
        pages[i].save(images_path.split('.')[0] + 'page_' + str(i) + '.jpg','JPEG')
        
def all_volumes_to_images(pdf_path, images_path):
    for volume in os.listdir(pdf_path):
        pdf_volume_to_images(pdf_path + volume, images_path + volume)
        
if __name__ == '__main__':
    pdf_path = 'pdf_data/color/'
    images_path = 'image_data/color/'
    all_volumes_to_images(pdf_path, images_path)