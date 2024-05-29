import numpy as np


def get_str(s, f, b):
    par = s.partition(f)
    return(par[2].partition(b))[0][:]
