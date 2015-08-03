from __future__ import division
from sys import stdin, stdout
import numpy as np
from utils import *



if __name__ == '__main__':

  k, t = ReadIntegers()
  text_list = ReadAllLines()
  motifs = BruteForceMedianString(text_list, k)
  PrintList(
      motifs,
      '\n'
  )

