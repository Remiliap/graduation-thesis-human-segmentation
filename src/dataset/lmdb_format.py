import pathlib
import lmdb
import io
import numpy as np

from dataset.dataset import Image_Dataset
from dataset.lmdb_env import open as open_lmdb


class Lmdb_dataset(Image_Dataset):
    def __init__(self, db_file: str, images_path: str, transform=None):
        env = open_lmdb(db_file, map_size=2**40)
        self.txn = env.begin(buffers=True)
        cursor = self.txn.cursor()
        if not cursor.set_range(images_path.encode()):
            raise RuntimeError("{} not found.".format(images_path))

        images_path = pathlib.Path(images_path)
        files = []
        for file in cursor.iternext(keys=True, values=False):
            file_path = pathlib.Path(file.tobytes().decode())
            if file_path.is_relative_to(images_path):
                files.append(file)

        super().__init__(files, transform)

    def read_image(self, path):
        buffer = io.BytesIO(self.txn.get(path))
        image = np.load(buffer, allow_pickle=False)
        transform_image = None
        if self.transform:
            transform_image = self.transform(image)
        return transform_image, image
