# written by Elias Vansteenkiste, May 22, 2016
import sys
import subprocess
import os
from random import randint
import operator
import argparse 




parser = argparse.ArgumentParser(description='Process log file.')
parser.add_argument('--filename', dest='filename', action='store',
                   default="test.log",
                   help='read and process the log from the filename given (default: test.log)')
args = parser.parse_args()



def geometric_mean(iterable):
    return (reduce(operator.mul, iterable)) ** (1.0/len(iterable))


rpt = open(args.filename,'r')
lines = rpt.readlines()

config_start_pattern = "Configuration "
config_end_pattern = "end Configuration"
inside_config = False
config_printed = False
for line in lines:
    if not config_printed and config_start_pattern in line:
        inside_config = True
        print "Configuration:"
    elif not config_printed and config_end_pattern in line:
        inside_config = False
        config_printed = True
    elif not config_printed and inside_config:
        print line,

found_parameters = False
no_parameters_pattern = "# Neural Network with "
for line in lines:
    if not found_parameters and no_parameters_pattern in line:
        words = line.split()
        print "No. learnable parameters:", words[4]
        found_parameters = True
print
acc = "Accuracy validation:"
auc = "auc"

for line in lines:
    if acc in line or auc in line:
        print line, 

