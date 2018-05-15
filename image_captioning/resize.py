import argparse
import os
from PIL import Image, ImageDraw
import numpy as np
from PIL import ImageChops


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)

def resize_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)
        if i % 100 == 0:
            print ("[%d/%d] Resized the images and saved into '%s'."
                   %(i, num_images, output_dir))

def concatenate_images(image_dir, output_dir):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    images.sort(key=lambda f: int(filter(str.isdigit, f)))
    num_images = len(images)
    imgs = []
    for i, image in enumerate(images):
        if os.path.isdir(image_dir + image):
            continue
        # print(image)
        if i==0:
            continue
        if i%2==1:
            imgs.append(Image.open(image_dir + image))
            print(image)
        else:
            imgs.append(Image.open(image_dir + image))
            print(image)
            imgs_comb = np.hstack(( np.asarray(j) for j in imgs))
            imgs_comb = Image.fromarray(imgs_comb)
            imgs_comb.save(output_dir + 'Frogger_State_' + str(i/2) + '.jpg' )
            imgs = []
            print("next")

def subtract_and_concatenate_images(image_dir, output_dir):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    images.sort(key=lambda f: int(filter(str.isdigit, f)))
    num_images = len(images)
    imgs = []
    for i, image in enumerate(images):
        if os.path.isdir(image_dir + image):
            continue
        # print(image)
        if i==0:
            continue
        if i%2==1:
            imgs.append(Image.open(image_dir + image))
            print(image)
        else:
            imgs.append(Image.open(image_dir + image))
            print(image)
            diff = ImageChops.subtract(imgs[1],imgs[0])
            imgs[1] = diff
            imgs_comb = np.hstack(( np.asarray(j) for j in imgs))
            imgs_comb = Image.fromarray(imgs_comb)
            imgs_comb.save(output_dir + 'Frogger_State_' + str(i/2) + '.jpg' )
            imgs = []
            print("next")

def diff (a, b):
    return sum ( (a - b) ** 2 for a, b in zip(a, b) )

def main(args):
    im = Image.open('./png/Screenshot_1026.png')
    isize = im.size
    frog =  Image.open('./png/frog.png')
    fsize = frog.size
    x0, y0 = fsize [0] // 2, fsize [1] // 2
    # print(x0,y0)
    # print(fsize)
    pixel = frog.getpixel((x0 + 10, y0 + 10))[:-1]
    best = (100000, 0, 0)
    for x in range (isize[0]):
        for y in range (isize[1]):
            ipixel = im.getpixel ((x, y))
            d = diff (ipixel, pixel)
            if d < best[0]: best = (d, x, y)

    draw = ImageDraw.Draw(im)
    x, y = best [1:]
    rect = (150,150)
    im2 = im.crop((x - x0 + 10 - rect[0]/2, y - y0 + 5 - rect[1]/2, x + 2*x0 + rect[0]/2, y + 2*y0 + rect[1]/2))
    draw.rectangle ((x - x0 + 10 - rect[0]/2, y - y0 + 5 - rect[1]/2, x + 2*x0 + rect[0]/2, y + 2*y0 + rect[1]/2), outline = 'red')
    im.save ('out.png')
    im2_array = np.asarray(im2)
    im2_array.setflags(write=1)
    # im2_array[0][0][0] = 1
    # print(np.asarray(im).shape)
    # print(im2_array.shape)
    # np.place(im2_array,im2_array==0,255)
    print(im2_array.shape)
    print(im2_array[170][170][:])
    for i in range(im2_array.shape[0]):
        for j in range(im2_array.shape[1]):
            if im2_array[i][j][0]==0 and im2_array[i][j][1]==0 and im2_array[i][j][2]==0 and im2_array[i][j][3]==0:
                im2_array[i][j][0]=0
                im2_array[i][j][1]=0
                im2_array[i][j][2]=0
                im2_array[i][j][3]=255
    # print(im2_array)
    # exit(0)
    im2 = Image.fromarray(im2_array,'RGBA')
    im2.save ('cropped_output.png')
    exit(0)  
    splits = ['train', 'val']
    for split in splits:
        image_dir = args.frogger_image_dir
        output_dir = args.output_dir
        concatenated_images_dir = args.concatenated_images_dir
        subtracted_images_dir = args.subtracted_images_dir
        image_size = [args.image_size, args.image_size]
        # concatenate_images(image_dir, concatenated_images_dir)
        subtract_and_concatenate_images(image_dir, subtracted_images_dir)
        # exit(0)
        resize_images(concatenated_images_dir, output_dir, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='./data/train2014/',
                        help='directory for train images')
    parser.add_argument('--frogger_image_dir', type=str, default='./Sample_Training_Set/',
                        help='directory for train images')
    parser.add_argument('--output_dir', type=str, default='./data/FroggerDataset/',
                        help='directory for saving resized images')
    parser.add_argument('--concatenated_images_dir', type=str, default='./data/concatenated2014/',
                        help='directory for saving resized images')
    parser.add_argument('--subtracted_images_dir', type=str, default='./data/subtracted2014/',
                        help='directory for saving resized images')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size for image after processing')
    args = parser.parse_args()
    main(args)