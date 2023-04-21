import lmdb

env = {}

def open(path: str, **argv):
    """如果env已经被打开，返回该env，否则使用参数创建"""
    if path not in env:
        env[path] = lmdb.open(path, **argv)
    return env[path]
