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
            max_length = max(im.size[:2])
            size = (np.asarray(im.size[:2]) * length) // max_length
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

    def get_image_files(self, fn_z):
        raise NotImplementedError()


class CUBDataPreprocessor(AbstractDataPreprocessor):
    def convert(self):
        dataset_dir = self.args.dataset_dir
        path = os.path.join(dataset_dir,
                            'cub/CUB2002011/CUB_200_2011/CUB_200_2011')
        x_folder = os.path.join(path, 'images')
        fn_z_train = os.path.join(path, 'images.txt')
        fn_f_train = os.path.join(path, 'feat.npy')
        return self.load_data(x_folder, fn_z_train, fn_f_train)

    def get_image_files(self, fn_z):
        with open(fn_z, 'r') as f:
            lines = f.readlines()
        names = [line.strip().split()[1] for line in lines]
        return names


class SUNDataPreprocessor(AbstractDataPreprocessor):
    def convert(self):
        dataset_dir = self.args.dataset_dir
        path = os.path.join(dataset_dir, 'sun')
        x_folder = os.path.join(path, 'SUNAttributeDB_Images/images')
        fn_z_train = os.path.join(path, 'SUNAttributeDB/images.mat')
        fn_f_train = os.path.join(path, 'feat.npy')
        return self.load_data(x_folder, fn_z_train, fn_f_train)

    def get_image_files(self, fn_z):
        mat = scipy.io.loadmat(fn_z)
        images = mat['images']
        images = np.reshape(images, [-1])
        names = [name[0] for name in images]
        return names


class NICODataPreprocessor(AbstractDataPreprocessor):
    def get_categories(self):
        with open('category.txt', 'r') as f:
            lines = f.readlines()
        names = lines[0].split('\t')
        categories = [[] for _ in range(len(names) - 1)]
        for line in lines[1:]:
            line = line.strip()
            if len(line) == 0:
                continue
            terms = line.strip().split('\t')
            categories[len(terms) - 2].append(terms[0])
        return categories

    def convert(self):
        dataset_dir = '../../data/nico/track_1'
        background_names = ['autumn', 'dim', 'grass', 'rock', 'water']
        foreground_names = self.get_categories()
        return self.load_data(dataset_dir, background_names, foreground_names)

    def load_data(self, dataset_dir, background_names, foreground_names):
        matrix = [[] for _ in background_names]
        for i, background_name in enumerate(background_names):
            background_folder = os.path.join(dataset_dir, background_name)
            for j, foreground_name_list in enumerate(foreground_names):
                images = []
                for foreground_name in foreground_name_list:
                    path = os.path.join(background_folder, foreground_name)
                    files = os.listdir(path)
                    for file_name in files:
                        fn = os.path.join(path, file_name)
                        image = self.load_an_image(fn)
                        images.append(image)
                matrix[i].append(np.asarray(images))
                print(i, j)
        matrix = np.asarray(matrix, dtype=object)
        fn_f = os.path.join(dataset_dir, 'feat.npy')
        matrix.dump(fn_f)


def main(args):
    if args.dataset == 'cub':
        proc = CUBDataPreprocessor(args)
    elif args.dataset == 'sun':
        proc = SUNDataPreprocessor(args)
    elif args.dataset == 'nico':
        proc = NICODataPreprocessor(args)
    else:
        raise ValueError('{0} is not a valid dataset.'.format(args.dataset))
    proc.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nico', help='Dataset.')
    parser.add_argument('--dataset_dir', type=str,
                        default='../../data/zeroshot_datasets',
                        help='Dataset directory.')
    main(parser.parse_args())
