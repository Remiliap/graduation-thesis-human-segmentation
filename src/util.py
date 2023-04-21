from pathlib import Path
import os
import shutil
from datetime import datetime, timedelta
import torch
from torch import nn


def create_dir(path: Path):
    r"""尝试创建一个目录

    如果已存在但不是目录,删除并重新创建

    如果已存在,不做任何事

    如果未存在,创建它

    返回是否创建了目录"""
    if not path.exists():
        os.mkdir(path)
        print("Successfully created '{}' ".format(path))
    else:
        print("{} have been exists.".format(path))
        return False
    return True


def replace_dir(path: Path):
    r"""创建或替换一个目录

    如果已存在,删除并重新创建

    如果未存在,创建它"""
    if path.exists():
        print("{} have been exists,delete it.".format(path))
        if path.is_dir():
            shutil.rmtree(path)
        else:
            os.remove(path)

    os.mkdir(path)
    print("Successfully created '{}' ".format(path))


def compare_dir(dir1: Path, dir2: Path):
    """比较两个目录文件名是否全部一致"""

    names1 = [img.stem for img in dir1.iterdir()]
    names1.sort()

    names2 = [img.stem for img in dir2.iterdir()]
    names2.sort()

    for n1, n2 in zip(names1, names2):
        if n1 != n2:
            return False
    return True


def ceil_2powN(number: int, N: int):
    """将number增大到刚好可以被2的N次方整除"""
    N = 1 << N
    remainder = number & (N-1)
    number = number - remainder + (remainder != 0)*N
    return number


class Clock:

    def __init__(self) -> None:
        self.start = datetime.now()
        self.last_read = self.start
        self.timers = {}
        self.counter = 0

    def total_time(self):
        return datetime.now() - self.start

    def update(self):
        now_time = datetime.now()
        interval = now_time - self.last_read
        self.last_read = now_time
        for timer, (timer_duration, timer_clock) in self.timers.items():
            self.timers[timer] = (timer_duration, timer_clock - interval)
        return interval

    def set_timer(self, interval: timedelta):
        """"""
        self.counter += 1
        self.timers.setdefault(self.counter, (interval, interval))
        return self.counter

    def drop_timer(self, timer, **kargs):
        del self.timers[timer]

    def restart_timer(self, timer, **kargs):
        timer_duration = self.timers[timer][1]
        self.timers[timer] = (timer_duration, timer_duration)

    def continue_timer(self, timer, interval):
        timer_duration = self.timers[timer][0]
        self.timers[timer] = (timer_duration, timer_duration -
                              (interval % timer_duration))

    def is_timeout(self, timer, timeout_action=continue_timer):
        interval = self.update()
        if self.timers[timer][1] > timedelta():
            return False
        timeout_action(self, timer=timer, interval=interval)
        return True

def split_parameters(module:nn.Module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay