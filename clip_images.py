import os

import cv2
from tqdm import tqdm


def main():
    target_dir = 'data/'
    input_dirs = ['data_bak/train/images', 'data_bak/train/masks', 'data_bak/test/images', 'data_bak/test/masks']

    for input_dir in input_dirs:
        images = os.listdir(input_dir)
        output_dir = os.path.join(target_dir, input_dir.split('/')[1], input_dir.split('/')[2])
        os.makedirs(output_dir, exist_ok=True)
        for image in tqdm(images):
            if '_pre_disaster.png' in image:
                image_name = image[:-17]
                img = cv2.imread(os.path.join(input_dir, image))
                img1 = img[:512, :512, :]
                img2 = img[:512, 512:, :]
                img3 = img[512:, :512, :]
                img4 = img[512:, 512:, :]

                cv2.imwrite(os.path.join(output_dir, image_name + '_1_pre_disaster.png'), img1)
                cv2.imwrite(os.path.join(output_dir, image_name + '_2_pre_disaster.png'), img2)
                cv2.imwrite(os.path.join(output_dir, image_name + '_3_pre_disaster.png'), img3)
                cv2.imwrite(os.path.join(output_dir, image_name + '_4_pre_disaster.png'), img4)

            elif '_post_disaster.png' in image:
                image_name = image[:-18]
                img = cv2.imread(os.path.join(input_dir, image))
                img1 = img[:512, :512, :]
                img2 = img[:512, 512:, :]
                img3 = img[512:, :512, :]
                img4 = img[512:, 512:, :]

                cv2.imwrite(os.path.join(output_dir, image_name + '_1_post_disaster.png'), img1)
                cv2.imwrite(os.path.join(output_dir, image_name + '_2_post_disaster.png'), img2)
                cv2.imwrite(os.path.join(output_dir, image_name + '_3_post_disaster.png'), img3)
                cv2.imwrite(os.path.join(output_dir, image_name + '_4_post_disaster.png'), img4)


if __name__ == "__main__":
    main()
