# Instructions
# ------------

# This file is to find "dirty data" in .libsvm file.
# Usage:
#   python check_nan.py filename.libsvm

# Written by Junbo Zhao, in Douban Inc.

##=================================================
import sys
import math
from svmutil import *

assert len(sys.argv) == 2
filename = sys.argv[1]
assert isinstance(filename, str)

data = svm_read_problem(filename)
y = data[0]
x = data[1]
assert isinstance(x, list) & isinstance(y, list)
assert len(x) == len(y)

# Checking the data
print "Bad data index:"
for xx in x:
    for check_ele in xx.values():
        if math.isnan(check_ele):
            print(x.index(xx))
            break
