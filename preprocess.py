import os
import cv2
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=512)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    img_path_list = list(data_dir.glob('*'))

    cnt = 0
    for p in tqdm(img_path_list, total=len(img_path_list)):
        img = cv2.imread(str(p))
        h, w, c = img.shape
        if h != args.img_size or w != args.img_size:
            os.remove(p)
            cnt += 1
            print(f'Deleted {str(p)}')
    print(f'Deleted {cnt} images.')


if __name__ == '__main__':
    main()
