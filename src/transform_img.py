import cv2
import numpy as np
import torch
import torchvision

from collections.abc import Iterator

from util import ceil_2powN


def onehot_seq_np(data: np.ndarray, categories_num: int, dtype: np.dtype = np.uint8):
    """
    将标记图data转换为onehot编码。
    data是shape为(n,m)或(n,m,1)的标记图,使用不同颜色标记类别,该函数将其转换为(categories_num,n,m),相当于categories_num张图片,每一张表示一个类别的图像
    """
    if len(data.shape) == 2:
        shape = (categories_num, )+data.shape
    elif len(data.shape) == 3 and data.shape[2] == 1:
        shape = (categories_num, )+data.shape[0:2]
    else:
        raise RuntimeError("Can't resolve shape {}".format(data.shape))

    buf = np.zeros(shape, dtype=dtype)
    nmsk = np.arange(data.size, dtype=np.int64) + \
        (data.size * data.astype(np.int64).ravel())
    buf.ravel()[nmsk] = 1
    return buf


def onehot_seq_torch(data: torch.Tensor, categories_num: int, dtype: torch.dtype = torch.uint8):
    if data.dim() == 2:
        shape = (categories_num,)+data.shape
    elif data.dim() == 3 and data.shape[0] == 1:
        shape = (categories_num,)+data.shape[1:3]
    else:
        raise RuntimeError("Can't resolve shape {}".format(data.shape))

    try:
        data = data.to(torch.int64)
        onehot = torch.zeros(shape, dtype=dtype).scatter_(
            0, data, 1)
    except RuntimeError:
        print("Data bincount:")
        print(torch.bincount(data.flatten()))

    return onehot


def flatten_onehot(data: torch.Tensor | np.ndarray):
    return data.argmax(0)


def eq_proportion_resize(img: cv2.Mat, length: float, Interpolation_flag: int = cv2.INTER_LINEAR):
    height, width = img.shape[0:2]
    f = length / max(width, height)
    return cv2.resize(img, (0, 0), fx=f, fy=f, interpolation=Interpolation_flag)


class Diff_size_collect:
    @staticmethod
    def get_suitable_collect_size(imgs: Iterator[torch.Tensor], downsamp_multi: int = 1):
        """获得适合输入模型中的图片形状,首先得到所有图片中最大的长度和宽度,然后将其扩大到可被2的N次方整除,N对应模型有 downsamp_multi 次下采样"""
        max_height = 0
        max_width = 0
        for img in imgs:
            max_height = max(img.shape[1], max_height)
            max_width = max(img.shape[2], max_width)

        max_width = ceil_2powN(max_width, downsamp_multi)
        max_height = ceil_2powN(max_height, downsamp_multi)
        return max_height, max_width

    @staticmethod
    def collect(imgs: list[torch.Tensor],
                size: tuple[int, int], fill: torch.Tensor = None, random_place=True):
        """根据size,将图片整合成一个批次,不足的地方补fill"""

        channel = imgs[0].shape[0]
        dtype = imgs[0].dtype

        batched_imgs = torch.empty((len(imgs), channel, *size), dtype=dtype)

        if fill == None:
            fill = torch.zeros((batched_imgs.shape[1], 1, 1), dtype=dtype)

        for img, batched_img in zip(imgs, batched_imgs):
            if random_place:
                y_place = size[0]-img.shape[1]
                if y_place > 0:
                    y_place = torch.randint(y_place, (1,)).item()

                x_place = size[1]-img.shape[2]
                if x_place > 0:
                    x_place = torch.randint(x_place, (1,)).item()
            else:
                y_place = 0
                x_place = 0

            batched_img[:, y_place:y_place+img.shape[1],
                        x_place:x_place+img.shape[2]] = img

            batched_img[:, :y_place, :] = fill
            batched_img[:, y_place+img.shape[1]:, :] = fill

            batched_img[:, y_place:y_place+img.shape[1],
                        x_place+img.shape[2]:] = fill
            batched_img[:, y_place:y_place+img.shape[1], :x_place] = fill

        return batched_imgs

    @staticmethod
    def collect_fn(imgs: list[tuple[torch.Tensor]], downsamp_multi: int, un_collect_mask: set = None, black: dict = None):
        """un_collect_mask: 指示相应索引位置的数据不需要collect.
        black: 指示位置i的图片对应的填充颜色
        """

        if un_collect_mask == None:
            un_collect_mask = set()

        batched_imgs: list[torch.Tensor] = []

        rng_state = torch.get_rng_state()
        for i in range(len(imgs[0])):
            collect_imgs = [img[i] for img in imgs]

            if i in un_collect_mask:
                batched_imgs.append(collect_imgs)
                continue

            size = Diff_size_collect.get_suitable_collect_size(
                collect_imgs, downsamp_multi)

            torch.set_rng_state(rng_state)
            batched_imgs.append(Diff_size_collect.collect(
                collect_imgs, size, black.get(i)))

        return batched_imgs


color_means = [0.485, 0.456, 0.406]
color_stds = [0.229, 0.224, 0.225]
norm_black_color = [-m/s for m, s in zip(color_means, color_stds)]


def get_transform(img_size, output_channels):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(
            lambda img:eq_proportion_resize(img, float(img_size), cv2.INTER_CUBIC)),
        torchvision.transforms.Lambda(
            lambda img:cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomRotation((-10, 10)),
        torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.Normalize(
            mean=color_means,
            std=color_stds, inplace=True)
    ])
    transform_random_layer = {3, 4}

    target_transform = [
        torchvision.transforms.Lambda(
            lambda img:eq_proportion_resize(img, float(img_size), cv2.INTER_NEAREST)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomRotation((-10, 10)),
    ]

    if output_channels == 1:
        target_transform.append(torchvision.transforms.Lambda(
            lambda img: img*255))
    else:
        target_transform.append(torchvision.transforms.Lambda(
            lambda img: onehot_seq_torch(img*255, output_channels, torch.float32)))

    target_transform = torchvision.transforms.Compose(target_transform)

    target_transform_random_layer = {2}

    """opencv读取的图片形状为(Height,Width,Channel),转换后图片具有形状(Channel,Height,Width)"""

    transform_rm_rand_layer = torchvision.transforms.Compose(
        [tr for i, tr in enumerate(transform.transforms) if i not in transform_random_layer])

    target_transform_rm_rand_layer = torchvision.transforms.Compose(
        [tr for i, tr in enumerate(target_transform.transforms) if i not in target_transform_random_layer])

    return transform, target_transform, transform_rm_rand_layer, target_transform_rm_rand_layer


def get_pretreat_transform(output_channels):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomRotation((-10, 10)),
        torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2),
        torchvision.transforms.Normalize(
            mean=color_means,
            std=color_stds, inplace=True)
    ])
    transform_random_layer = {1, 2}

    target_transform = [
        torchvision.transforms.Lambda(
            lambda img:torch.tensor(np.expand_dims(img, 0), dtype=torch.float32)),
        torchvision.transforms.RandomRotation((-10, 10)),
    ]

    if output_channels != 1:
        target_transform.append(torchvision.transforms.Lambda(
            lambda img: onehot_seq_torch(img, output_channels, torch.float32)))

    target_transform = torchvision.transforms.Compose(target_transform)

    target_transform_random_layer = {1}

    """opencv读取的图片形状为(Height,Width,Channel),转换后图片具有形状(Channel,Height,Width)"""

    transform_rm_rand_layer = torchvision.transforms.Compose(
        [tr for i, tr in enumerate(transform.transforms) if i not in transform_random_layer])

    target_transform_rm_rand_layer = torchvision.transforms.Compose(
        [tr for i, tr in enumerate(target_transform.transforms) if i not in target_transform_random_layer])

    return transform, target_transform, transform_rm_rand_layer, target_transform_rm_rand_layer


def transform_label(img: torch.Tensor, label_map):
    size = img.size()
    size = (size[0], len(label_map)+1, *size[2:])
    new_img = torch.zeros(size, device=img.device, dtype=img.dtype)
    for v, l in label_map.items():
        l = torch.tensor(l, dtype=torch.int, device=img.device)
        new_img[:, v], _ = torch.max(img.index_select(1, l), dim=1)
    new_img[:, 0] = img[:, 0]

    return new_img
