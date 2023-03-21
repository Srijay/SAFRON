import os
import inspect
import time
import subprocess
from contextlib import contextmanager


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def int_tuple(s):
  return tuple(int(i) for i in s.split(','))


def float_tuple(s):
  return tuple(float(i) for i in s.split(','))


def str_tuple(s):
  return tuple(s.split(','))


