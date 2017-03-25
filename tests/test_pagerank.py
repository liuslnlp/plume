import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from mlearn.pagerank import page_rank

def main():
    a = np.array([[3/6, 2/6, 1/6], 
                  [10/15, 1/15, 4/15], 
                  [8/14, 1/14, 5/14]])
    print(page_rank(a))

if __name__ == '__main__':
    main()
    
