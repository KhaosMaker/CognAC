import numpy as np
from time import time
from random import random
from math import floor
from tqdm import tqdm
import csv

my_seeds = [5, 19, 31, 69, 75, 85, 97]
folder = 'misure2'

layers = 4
dim = [0] * layers
dist = [0] * layers
line_skip = 2
n = len(my_seeds)
letters = ["a",   "b",    "c",    "d",    "e",    "f",    'g',    'h', 'i',       'l']
for letter in letters:
    for my_seed in my_seeds:
        with open(folder+'/model_'+letter+'_info_'+str(my_seed)+'.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count >= line_skip:
                    dim[line_count-line_skip] += float(row[1])
                    dist[line_count-line_skip] += float(row[3])

                line_count += 1
    dim = [x/n for x in dim]
    dist = [x/n for x in dist]

    with open(folder+'/result_model_'+letter+'_all.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([letter, 'DIM']+dim)
        csv_writer.writerow([letter, 'DIST']+dist)