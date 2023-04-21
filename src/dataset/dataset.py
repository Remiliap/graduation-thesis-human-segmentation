import pathlib
import tarfile

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, Subset

from dataset.tarpath import Tar_path


class Image_Dataset(Dataset):
    def __init__(self,
                 files: list[pathlib.Path],
                 transform=None, cv_imread_flag=cv2.IMREAD_COLOR):
        super().__init__()

        self.transform = transform
        self.images_path = files
        self.cv_imread_flag_ = cv_imread_flag

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        path = self.get_image_path(idx)
        transform_images, _ = self.read_image(path)

        return (transform_images,)

    def get_image_path(self, idx: int):
        return self.images_path[idx]

    def read_image(self, path: pathlib.Path):
        """根据路径读取图片并转换
        返回:
        (转换图片,原始图片)
        """
        with path.open(mode="rb") as file:
            buf = file.read()
            buf = np.frombuffer(buf, dtype=np.uint8)
            image = cv2.imdecode(buf, self.cv_imread_flag_)

        transform_image = None
        if self.transform:
            transform_image = self.transform(image)
        return transform_image, image


class Zip_dataset(Dataset):
    def __init__(self, *datasets: Dataset) -> None:
        super().__init__()
        self.datasets = datasets
        if len(self.datasets) == 0:
            raise RuntimeError("Input datasets must more than 0.")
        elif len({len(d) for d in self.datasets}) > 1:
            raise RuntimeError("Input datasets must have same length")

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, index):
        zip_result = []
        rng_state = torch.get_rng_state()
        for dataset in self.datasets:
            torch.set_rng_state(rng_state)
            new_data = dataset[index]
            if isinstance(new_data, (tuple, list)):
                for d in new_data:
                    zip_result.append(d)
            else:
                zip_result.append(new_data)
        return zip_result


def split_dataset(
        dataset: Dataset,
        split: float = 0.85):
    """
    分割数据集
    split: 分割的训练集占比

    返回:
    (训练集,测试集)
    """

    data_size = len(dataset)
    indices = list(range(data_size))
    split = int(np.floor(split * data_size))

    train_idx, valid_idx = indices[:split], indices[split:]
    train_ds = Subset(dataset, train_idx)
    valid_ds = Subset(dataset, valid_idx)

    return (train_ds, valid_ds)


def get_data_files(images_path: pathlib.PurePath | str, work_dir="./"):
    if not isinstance(images_path, pathlib.PurePath):
        images_path = pathlib.PurePath(images_path)

    if not isinstance(work_dir, (pathlib.Path, Tar_path)):
        work_dir = pathlib.Path(work_dir)

    # 如果工作目录是tar文件
    if not isinstance(work_dir, Tar_path) and not work_dir.is_dir() and tarfile.is_tarfile(work_dir):
        work_dir = Tar_path.make_tar_root_path(work_dir)

    files = list((work_dir / images_path).iterdir())
    files.sort()
    return files
