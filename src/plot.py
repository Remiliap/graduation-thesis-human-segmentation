
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import torch
import torchvision

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from transform_img import flatten_onehot
import accuracy as acc


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.figure(figsize=(len(ave_grads)*0.2, 4.8))
    plt.bar(np.arange(len(max_grads)), max_grads, lw=1, color="y")
    plt.bar(np.arange(len(max_grads)), ave_grads, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers,
               rotation="vertical", size="small")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return plt.gcf()


def plot_kernels(tensor: torch.Tensor, num_cols=5, cmap="gray"):
    """Plotting the kernals and layers
    Args:
        Tensor :Input layer,
        n_iter : number of interation,
        num_cols : number of columbs required for figure
    Output:
        Gives the figure of the size decided with output layers activation map

    Default : Last layer will be taken into consideration
        """
    if not len(tensor.shape) == 4:
        raise Exception("assumes a 4D tensor")

    fig = plt.figure()
    i = 0
    t = tensor.data.numpy()
    b = 0
    a = 1

    for t1 in t:
        for t2 in t1:
            i += 1

            ax1 = fig.add_subplot(5, num_cols, i)
            ax1.imshow(t2, cmap=cmap)
            ax1.axis('off')
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

            if i == 1:
                a = 1
            if a == 10:
                break
            a += 1
        if i % a == 0:
            a = 0
        b += 1
        if b == 20:
            break


def onehot_gird(onehot_img: torch.Tensor, flatten: bool = True):
    """展示onethot图片的每一层,flatten为True,则额外展示压平的图片,返回make_grid制作的图片"""
    class_imgs = [t.unsqueeze(0).to(torch.float32) for t in onehot_img]
    if flatten:
        class_imgs.append(flatten_onehot(
            onehot_img).unsqueeze(0).to(torch.float32))
    return torchvision.utils.make_grid(
        class_imgs, normalize=True, scale_each=True, pad_value=0.5)


class Progress_writer:
    """使用matplot展示进度条"""

    def __init__(self, writer: SummaryWriter, tag: str) -> None:
        self.data = {}
        self.writer = writer
        self.tag = tag
        self.step = 0

    def plot(self, progress: int, size: int, title: str):
        self.data |= {title: (progress, size)}

        plt.figure(figsize=[6, len(self.data)*0.5])

        plt.barh(range(len(self.data)), [1]*len(self.data),
                 color="c")
        container = plt.barh(range(len(self.data)),
                             [float(p)/s for p, s in self.data.values()],
                             tick_label=[title for title in self.data.keys()],
                             color="y")
        plt.bar_label(container, ["{}/{}".format(p, s)
                                  for p, s in self.data.values()],
                      label_type="center")

        self.writer.add_figure(self.tag, plt.gcf(), self.step)
        self.step += 1


class LayerActivations:
    """Getting the hooks on each layer"""

    def __init__(self, layer: torch.nn.Module, device=None):
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.features: torch.Tensor = None
        self.device = device

    def hook_fn(self, module, input, output):
        self.features = output
        if self.device != None:
            self.features = self.features.to(self.device)

    def remove(self):
        self.hook.remove()


class Acc_record:
    def __init__(self, num_img: int, num_class) -> None:
        self.num_class = num_class
        self.num_img = num_img

        self.class_iou = np.ma.masked_array(
            np.empty((self.num_class, num_img)))
        self.mean_iou = np.empty((num_img,))
        self.front_iou = np.empty((num_img,))

        self.cpa = np.ma.masked_array(np.empty((self.num_class, num_img)))
        self.pa = np.empty((num_img,))

        self.step = 0

    def step_(self, step: int = None):
        if step == None:
            return self.step % self.num_img
        elif step < 0:
            raise RuntimeError("Step can't be negative.")
        else:
            self.step = step

    def calculate(self, prediction: torch.Tensor, y: torch.Tensor):
        if prediction.size(0) != y.size(0):
            raise RuntimeError("Channel doesn't match.")

        for pred_img, target_img in zip(prediction, y):
            confusion_matrix = acc.confusion_matrix(
                pred_img, target_img, self.num_class)

            step = self.step % self.num_img

            disappear = np.argwhere(np.sum(confusion_matrix, 1) == 0)
            # 类别iou
            self.class_iou[:, step] = acc.Iou(confusion_matrix)
            self.class_iou[disappear, step] = np.ma.masked
            # 类别平均iou
            self.mean_iou[step] = np.mean(self.class_iou[:, step])
            # 前景iou
            intersection = np.sum(confusion_matrix[1:, 1:])
            union = intersection + \
                np.sum(confusion_matrix[1:, 0]) + \
                np.sum(confusion_matrix[0, 1:])
            self.front_iou[step] = intersection / union
            # cpa
            self.cpa[:, step] = acc.class_accuracy(confusion_matrix)
            self.cpa[disappear, step] = np.ma.masked

            # pa
            self.pa[step] = acc.pixel_accuracy(confusion_matrix)

            self.step += 1


class Acc_writer:
    def __init__(self, writer: SummaryWriter, record: Acc_record, label: dict[int, str], topic: str = "Accuracy") -> None:
        self.writer = writer
        self.topic = topic
        self.label = label
        self.record = record

    def write_histogram(self, total_step: int):
        record_step = self.record.step_() - 1
        for lable_value, description in self.label.items():
            class_iou = self.record.class_iou[lable_value, :record_step]
            # 过滤被屏蔽值
            class_iou = class_iou[~class_iou.mask]
            if class_iou.size > 0:
                self.writer.add_histogram(
                    "{}/{}".format(self.topic, description),
                    class_iou, total_step)

        self.writer.add_histogram("{}/Mean".format(self.topic),
                                  self.record.mean_iou[:record_step], total_step)

        self.writer.add_histogram("{}/Front".format(self.topic),
                                  self.record.front_iou[:record_step], total_step)

    def write_scalas(self, epoch: int):
        mean_valid_iou = self.record.class_iou.mean(1)
        mean_valid_iou[mean_valid_iou.mask] = 0

        self.writer.add_scalars("{}/Iou".format(self.topic), {
            description: mean_valid_iou[lable_i]
            for lable_i, description in self.label.items()
        } | {
            "Mean": self.record.mean_iou.mean(),
            "Front": self.record.front_iou.mean()
        }, epoch)


class Loss_record:
    """用于记录loss"""

    def __init__(self, loss_fn: torch.nn.Module, step: int) -> None:
        self.loss_fn = loss_fn
        self.log = torch.empty((step,))

        self.step = 0

    def step_(self, step: int = None):
        if step == None:
            return self.step
        elif step < 0:
            raise RuntimeError("Step can't be negative.")
        else:
            self.step = step % self.log.size(0)

    def __call__(self, *args, **argv):
        loss = self.loss_fn(*args, **argv)
        self.log[self.step % self.log.size(0)] = loss.item()
        self.step += 1
        return loss


class Loss_writer:
    """将loss记录写到TensorBoard"""

    def __init__(self, *records: tuple[Loss_record, str], writer: SummaryWriter) -> None:
        self.records: list[tuple[Loss_record, str, str]] = []
        for record, topic in records:
            name = record.loss_fn.__class__.__name__
            histogram_tag = "{}/{}".format(topic, name)
            scalas_name = "{}_{}".format(topic, name)
            self.records.append((record, histogram_tag, scalas_name))

        self.writer = writer

    def write_histogram(self, total_step: int, topic: str):
        for record, r_topic, _ in self.records:
            if r_topic[:r_topic.find("/")] == topic:
                self.writer.add_histogram(
                    r_topic, record.log[:record.step_()], total_step)

    def write_scalas(self, topic: str, epoch: int):
        self.writer.add_scalars(topic, {
            r_topic: record.log.mean() for record, _, r_topic in self.records
        }, epoch)
