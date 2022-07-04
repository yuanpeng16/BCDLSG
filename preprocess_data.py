import argparse
import os
import scipy.io
import numpy as np
from PIL import Image


class AbstractDataPreprocessor(object):
    def __init__(self, args):
        self.args = args

    def load_an_image(self, fn):
        length = 64
        with Image.open(fn) as im:
            # Mode converting
            if im.mode != 'RGB':
                im = im.convert('RGB')

            # Resize
            mxlength = max(im.size[:2])
            size = (np.asarray(im.size[:2]) * length) // mxlength
            assert max(size) == length
            resized = im.resize(size, Image.ANTIALIAS)

            # Padding
            result = Image.new(resized.mode, (length, length), (0, 0, 0))
            position = (np.asarray(result.size[:2]) - size) // 2
            result.paste(resized, tuple(position))
        return np.array(result)

    def load_data(self, x_folder, fn_z, fn_f):
        names = self.get_image_files(fn_z)
        assert len(names) == len(set(names))
        matrix = []
        for i, name in enumerate(names):
            image = self.load_an_image(os.path.join(x_folder, name))
            matrix.append(image)
            if i % 100 == 0:
                print(i, len(names),
                      str(round((100 * i) / len(names), 2)) + '%')
        matrix = np.asarray(matrix)
        matrix.dump(fn_f)


class CUBDataPreprocessor(AbstractDataPreprocessor):
    def convert(self):
        dataset_dir = self.args.dataset_dir
        path = dataset_dir + 'cub/CUB2002011/CUB_200_2011/CUB_200_2011/'
        x_folder = path + 'images/'
        fn_z_train = path + 'images.txt'
        fn_f_train = path + 'feat.npy'
        return self.load_data(x_folder, fn_z_train, fn_f_train)

    def get_image_files(self, fn_z):
        with open(fn_z, 'r') as f:
            lines = f.readlines()
        names = [line.strip().split()[1] for line in lines]
        return names


class SUNDataPreprocessor(AbstractDataPreprocessor):
    def convert(self):
        dataset_dir = self.args.dataset_dir
        path = dataset_dir + 'sun/'
        x_folder = path + 'SUNAttributeDB_Images/images/'
        fn_z_train = path + 'SUNAttributeDB/images.mat'
        fn_f_train = path + 'feat.npy'
        return self.load_data(x_folder, fn_z_train, fn_f_train)

    def get_image_files(self, fn_z):
        mat = scipy.io.loadmat(fn_z)
        images = mat['images']
        images = np.reshape(images, [-1])
        names = [name[0] for name in images]
        return names


def main(args):
    if args.dataest == 'cub':
        proc = CUBDataPreprocessor(args)
    else:
        proc = SUNDataPreprocessor(args)
    proc.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataest', type=str, default='cub',
                        help='Dataset.')
    parser.add_argument('--dataset_dir', type=str,
                        default='../../data/zeroshot_datasets/',
                        help='Dataset directory.')
    main(parser.parse_args())
