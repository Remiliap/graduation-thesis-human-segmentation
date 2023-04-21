import numpy as np
import torch


def _check_shape(prediction: np.ndarray | torch.Tensor,
                 ground_truth: np.ndarray | torch.Tensor):
    """"""
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            "Shape mismatch: input images must have the same shape.")


def confusion_matrix(prediction: torch.Tensor,
                     ground_truth: torch.Tensor, class_num: int):
    """计算混淆矩阵
    prediction: 任意形状,int类型
    ground_truth: 和prediction相同形状,int类型
    返回:
    (n_class,n_class)的矩阵,第i行j列表示第i类中预测为第j类的像素数
    """
    _check_shape(prediction, ground_truth)

    matrix = torch.bincount(
        class_num * ground_truth.flatten() + prediction.flatten(), minlength=class_num ** 2).reshape(class_num, class_num)
    return matrix.detach().cpu().numpy()


def pixel_accuracy(confusion_matrix: np.ndarray) -> np.ndarray:
    """像素正确率"""
    return np.diag(confusion_matrix).sum() / (confusion_matrix.sum() + 1e-10)


def class_accuracy(confusion_matrix: np.ndarray):
    """类别正确率"""
    class_accuracy: np.ndarray = np.diag(confusion_matrix) / \
        (confusion_matrix.sum(axis=1) + 1e-10)
    return class_accuracy


def Iou(confusion_matrix: np.ndarray):
    """类别交并比"""
    intersection = np.diag(confusion_matrix).astype(np.float32)
    union = confusion_matrix.sum(
        axis=1) + confusion_matrix.sum(axis=0) - intersection
    np.divide(intersection, union, out=intersection, where=union > 0)
    return intersection


def dice_coefficient(confusion_matrix: np.ndarray, true_label=1):
    """计算 dice系数,仅用于二分类"""
    if confusion_matrix.shape != (2, 2):
        raise ValueError("Shape must be 2*2")
    smooth = 1e-10

    intersection = confusion_matrix[true_label, true_label]

    sum = confusion_matrix[true_label, 1-true_label] + \
        confusion_matrix[1-true_label, true_label] + 2 * intersection

    return 2. * intersection / sum + smooth
