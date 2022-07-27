import numpy as np


if __name__ == '__main__':
    with open("wiki.txt",encoding='utf-8') as f:
        l = f.read()
        min_= 99999
        data = l.split('\n')
        for l_ in data:
            print(len(l_))
            min_ = min(len(l_),min_)
        print(min_)