import os
import pathlib
import tarfile

# 管理已经打开的只读的tar文件
_opened_ro_tars = {}


def get_tar(path: pathlib.PurePath) -> tarfile.TarFile:
    """获得路径上的tar文件,如果已经打开,直接返回"""
    if not isinstance(path, pathlib.PurePath):
        path = pathlib.Path(path)
    path = path.resolve()
    if path in _opened_ro_tars:
        file_record = _opened_ro_tars[path]
        _opened_ro_tars[path] = (file_record[0], file_record[1]+1)
        return _opened_ro_tars[path][0]
    else:
        new_file = _opened_ro_tars.setdefault(
            path, (tarfile.open(path, 'r'), 1))
        return new_file[0]


def close_tar(file: tarfile.TarFile):
    """关闭tar文件"""
    path = pathlib.Path(file.name).resolve()
    if path in _opened_ro_tars:
        file_record = _opened_ro_tars[path]
        _opened_ro_tars[path] = (file_record[0], file_record[1]-1)
        if _opened_ro_tars[path][1] == 0:
            file.close()
            del _opened_ro_tars[path]


class Tar_path(pathlib.PurePath):
    """提供tar文件只读的接口,接口和Path保持一致"""
    def __new__(cls, *args):
        if cls is Tar_path:
            cls = WindowsTarPath if os.name == 'nt' else PosixTarPath
        self = cls._from_parts(args)
        return self

    def __init__(self, *args) -> None:
        self.file: tarfile.TarFile = None

    def __del__(self):
        if "file" in self.__dict__ and self.file != None:
            close_tar(self.file)

    def __truediv__(self, key):
        new_file = get_tar(self.file.name)
        new_path = Tar_path(super().__truediv__(key))
        new_path.file = new_file
        return new_path

    def __rtruediv__(self, key):
        new_file = get_tar(self.file.name)
        new_path = Tar_path(super().__rtruediv__(key))
        new_path.file = new_file
        return new_path

    def iterdir(self):
        """"""
        for name in self.file.getnames():
            name = pathlib.PurePath(name)
            if name.parent == self:
                yield self/name.name

    def open(self, *args, **argv):
        return self.file.extractfile(str(self))

    @staticmethod
    def make_tar_root_path(path: str):
        tar_path = Tar_path("./")
        tar_path.file = get_tar(path)
        return tar_path


class PosixTarPath(Tar_path, pathlib.PurePosixPath):
    __slots__ = ()


class WindowsTarPath(Tar_path, pathlib.PureWindowsPath):
    __slots__ = ()