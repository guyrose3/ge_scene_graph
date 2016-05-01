#!/usr/bin/env python

import sys

def count(file):
    with open(file) as f:
        a = sum(1 for _ in f)
    print a

if __name__ == '__main__':
    count(sys.argv[1])