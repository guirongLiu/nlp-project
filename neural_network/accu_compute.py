import csv
import pandas as pd
import numpy as np
import sys

with open(sys.argv[1],"r") as csvfile:
    rows=csv.reader(csvfile)
    gtruth=[]
    for row in rows:
        if row[-1] == "is_duplicate":
            continue
        gtruth.append(int(row[-1]))

with open(sys.argv[2],"r") as res:
    rrows=csv.reader(res)
    lres=[]
    for row in rrows:
        if row[0] == "is_duplicate":
            continue
        if float(row[0][1:row[0].index(',')]) <= float(row[0][(row[0].index(',')+1):-1]):
            ilres = 1
        else:
            ilres = 0
        lres.append(int(ilres))

count=0

for ind in range(0,len(lres)):
    if (gtruth[ind] == lres[ind]):
        count += 1
print(float(count)/float(len(lres)))
