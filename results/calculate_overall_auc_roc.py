file0 = "hours_patient0_4_net509-28-19-59-58csv"
file1 = "hours_patient1_7_net509-28-20-18-56csv"
file2 = "hours_patient2_4_net509-28-17-46-25csv"

files = [file0, file1, file2]

ytrue = []
ypred = []

for file in files:
	with open(file,'r') as fp:
		for line in fp:
			words = line.rstrip().split(',')
			ytrue.append(float(words[0]))
			ypred.append(float(words[1]))

print ytrue
print ypred
from sklearn.metrics import roc_auc_score,log_loss
print "roc_auc:", roc_auc_score(ytrue, ypred)
print "log_loss", log_loss(ytrue, ypred)
