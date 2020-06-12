import os
import random
import numpy as np
from collections import Counter


root_dir = '.'
max = 4.0
# path_train = []
files = os.listdir(root_dir)
for file in files:
    if 'oldload' in file:
        short_name = file.split('old')[1]
        with open(short_name, 'w') as new_writer:
            with open(file, 'r') as reader:
                lines = reader.readlines()
                nums = [float(y) for y in lines]
                for x in nums:
                    if x > max:
                        x = max - np.exp(-x/max)
                    new_writer.write(str(x) + '\n')