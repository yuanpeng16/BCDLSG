import os
import numpy as np
from PIL import Image


class CUBDataPreprocessor(object):
    def convert(self):
        dataset_dir = '../../data/zeroshot_datasets/'
        path = dataset_dir + 'cub/CUB2002011/CUB_200_2011/CUB_200_2011/'
        x_folder = path + 'images/'
        fn_z_train = path + 'images.txt'
        fn_f_train = path + 'feat.npy'
        return self.load_data(x_folder, fn_z_train, fn_f_train)

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
        with open(fn_z, 'r') as f:
            lines = f.readlines()
        names = [line.strip().split()[1] for line in lines]
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


if __name__ == '__main__':
    p = CUBDataPreprocessor()
    p.convert()
