import json
import pathlib
import zlib
import base64
import io
from itertools import filterfalse

import numpy as np
import cv2

from dataset.dataset import Image_Dataset, Zip_dataset


def base64_2_mask(s):
    z = zlib.decompress(base64.b64decode(s))
    n = np.fromstring(z, np.uint8)
    mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(bool)
    return mask


def mask_2_base64(mask):
    img_pil = Image.fromarray(np.array(mask, dtype=np.uint8))
    img_pil.putpalette([0, 0, 0, 255, 255, 255])
    bytes_io = io.BytesIO()
    img_pil.save(bytes_io, format='PNG', transparency=0, optimize=0)
    bytes = bytes_io.getvalue()
    return base64.b64encode(zlib.compress(bytes)).decode('utf-8')


def get_data_fname(data_dir: pathlib.Path):
    """从supervisely数据集文件夹中读取所有声明文件和图片文件"""
    for sub_dir in data_dir.iterdir():
        if sub_dir.is_dir():
            for ann in (sub_dir/"ann").iterdir():
                name = ann.stem
                yield ann, sub_dir/"img"/name


def is_bitmap_format(path: pathlib.Path):
    """判断文件是否含有bitmap格式的对象"""
    with open(path) as annf:
        ann = json.load(annf)
    for obj in ann["objects"]:
        if obj["geometryType"] == "bitmap":
            return True
    return False


def get_mask_img(path: pathlib.Path):
    """从supervisely格式声明文件中读取bitmap格式的图片"""
    with path.open() as annf:
        ann = json.load(annf)

    size = ann["size"]["height"], ann["size"]["width"]
    mask_img = np.zeros(size, dtype=np.bool8)

    have_bitmap = False
    for obj in ann["objects"]:
        if obj["geometryType"] == "bitmap":
            bitmap: str = obj["bitmap"]["data"]
            bitmap = base64_2_mask(bitmap)

            x = obj["bitmap"]["origin"][0]
            y = obj["bitmap"]["origin"][1]
            mask_img[y:y+bitmap.shape[0], x:x+bitmap.shape[1]] |= bitmap
            have_bitmap = True

    if not have_bitmap:
        raise RuntimeError("No Bitmap.")

    return mask_img.astype(np.uint8).reshape(mask_img.shape+(1,))


def get_super_dataset(path: pathlib.Path, img_trans, ann_trans):
    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    files = filterfalse(
        lambda f: not is_bitmap_format(f[0]), get_data_fname(path))
    ann_files, img_files = [], []
    for annf, imgf in files:
        ann_files.append(annf)
        img_files.append(imgf)

    img_dataset = Image_Dataset(img_files, img_trans)
    ann_dataset = Super_maskImg_dataset(ann_files, ann_trans)
    return Zip_dataset(img_dataset, ann_dataset)


class Super_maskImg_dataset(Image_Dataset):
    def __init__(self, files: list[pathlib.Path], transform=None):
        super().__init__(files, transform)

    def read_image(self, path: pathlib.Path):
        image = get_mask_img(path)
        transform_image = None
        if self.transform:
            transform_image = self.transform(image)
        return transform_image, image
