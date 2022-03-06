from itertools import repeat
import multiprocessing as mp
import os
import pprint

def f(d: dict) -> None:
    pid = os.getpid()
    d[pid] = "Hi, I was written by process %d" % pid

def main():
    with mp.Manager() as manager:
        d = manager.dict()
        with manager.Pool() as pool:
            pool.map(f, repeat(d, 10))
        # `d` is a DictProxy object that can be converted to dict
        pprint.pprint(dict(d))