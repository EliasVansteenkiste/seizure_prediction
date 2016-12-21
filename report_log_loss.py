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
parser.add_argument('--detail', dest='details', action='store_true',
                   default=False,
                   help='give more details such as csv filenames (default: False)')
parser.add_argument('--show-cm', dest='show_cm', action='store_true',
                   default=False,
                   help='print the confusion matrix (default: False)')
parser.add_argument('--show-csv', dest='show_csv', action='store_true',
                   default=False,
                   help='print the name of the csv files for the prediction of the test set. (default: False)')
parser.add_argument('--show-fprdtpr', dest='show_fprdtpr', action='store_true',
                   default=False,
                   help='print the fpr/tpr of the results. (default: False)')

args = parser.parse_args()



def geometric_mean(iterable):
    return (reduce(operator.mul, iterable)) ** (1.0/len(iterable))

def median(lst):
    quotient, remainder = divmod(len(lst), 2)
    if remainder:
        return sorted(lst)[quotient]
    return sum(sorted(lst)[quotient - 1:quotient + 1]) / 2.


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

patterns = ["Error rate (%):"]
csv = ['.csv']
auc_patterns = ['geomean_roc_auc', 'geomean_roc_auc_hours']

tpr2dfpr = []
tprdfpr = []
aucs = []
haucs = []

pattern_found = 0
tp = -1
fn = -1
fp = -1
tpr = -1
fpr = -1

print 'tpr/(fpr+1e-6)\ttp \ttp*tpr/(fpr+1e-6)' 
for line in lines:
    if args.show_fprdtpr:
        if pattern_found == 1:
            if args.show_cm:
                print line,
            mu_line = filter(lambda ch: ch not in "[]", line)
            negatives = mu_line.split()
            tn = int(negatives[0])
            fp = int(negatives[1])
            fpr =  1.0*fp/(tn+fp)
            pattern_found = 2
        elif pattern_found == 2:
            if args.show_cm:
                print line,
            mu_line = filter(lambda ch: ch not in "[]", line)
            positives = mu_line.split()
            fn = int(positives[0])
            tp = int(positives[1])
            tpr = 1.0*tp/(fn+tp)
            print tpr/(fpr+1e-6), '\t', tp, '\t', tp*tpr/(fpr+1e-6)
            tpr2dfpr.append(tp*tpr/(fpr+1e-6))
            tprdfpr.append(tpr/(fpr+1e-6))
            pattern_found = 0
            tp = -1
            fn = -1
            fp = -1
            tpr = -1
            fpr = -1
        else:
            for p in patterns:
                if p in line:
                    pattern_found = 1
                break
    if args.show_csv:
        for p in csv:
            if p in line and 'Namespace' not in line:
                print line,
                break 
    if 'log_loss:' in line:
        aucs.append(float(line.split(':')[1].rsplit()[0]))
        print line,

print 'test:'
print len(aucs)
for i in range(0,len(aucs),5):
    print i/5, geometric_mean(aucs[i:i+5])

print 'geomean log_loss:', geometric_mean(aucs)
