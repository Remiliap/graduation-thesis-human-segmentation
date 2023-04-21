import lmdb
import cv2
import numpy as np
import io

from pathlib import Path

from transform_img import eq_proportion_resize


def transform(file: Path):
    file = file.resolve()
    buffer = io.BytesIO()
    if file.is_relative_to(
            "/mnt/fastest/Users/Public/code/graduate/lip_c5/training") or file.is_relative_to(
            "/mnt/fastest/Users/Public/code/graduate/lip_c5/validation"):
        img = cv2.imread(str(file))
        img = eq_proportion_resize(img, 512, cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np.save(buffer, img, allow_pickle=False)
    elif file.is_relative_to(
            "/mnt/fastest/Users/Public/code/graduate/lip_c5/training_seg") or file.is_relative_to(
            "/mnt/fastest/Users/Public/code/graduate/lip_c5/validation_seg"):
        img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        img = eq_proportion_resize(img, 512, cv2.INTER_NEAREST)
        np.save(buffer, img, allow_pickle=False)
    elif file.is_relative_to(
            "/mnt/fastest/Users/Public/code/graduate/lip_c5/test"):
        img = cv2.imread(str(file))
        np.save(buffer, img, allow_pickle=False)
    else:
        raise RuntimeError("")
    return buffer


def path2lmdb(path: Path, lmdb_file: Path, transform):

    def iter_dir_recur(path_to_iter: Path):
        """递归遍历文件夹，返回文件列表，不包含目录"""
        dirs = [path.iterdir()]
        while len(dirs) > 0:
            try:
                n = next(dirs[-1])
                if n.is_dir():
                    dirs.append(n.iterdir())
                elif n.is_file():
                    yield n
            except StopIteration:
                del dirs[-1]

    with lmdb.open(str(lmdb_file), readonly=False,map_size=2**40) as env:
        for f in iter_dir_recur(path):
            data: io.BytesIO = transform(f)
            data = data.getvalue()
            with env.begin(write=True) as txn:
                txn.put(str(f.relative_to(path)).encode(),
                        data, dupdata=False, overwrite=False)


if __name__ == "__main__":
    path2lmdb(Path("../../data/graduate/lip_c5"),
              Path("../../data/graduate/lip_c5_db"), transform)
