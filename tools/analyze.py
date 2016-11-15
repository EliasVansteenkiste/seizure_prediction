import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hickle as hkl
from utils import calcFFT
from matplotlib.ticker import AutoMinorLocator

import scipy.io as sio
import pandas as pd


def mat_to_pandas(path):
  mat = sio.loadmat(path)
  names = mat['dataStruct'].dtype.names
  for n in names:
  	print n
  	print mat['dataStruct'][n][0, 0].shape
  ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
  return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0]) 

def mat_to_nparray(path):
  mat = sio.loadmat(path)
  names = mat['dataStruct'].dtype.names
  # print mat['dataStruct']['sequence']
  return mat['dataStruct']['data'][0, 0]

def analyze(basename, suffix, savedir, start, no_of_samples, fft_width=512, overlap=448, height=64):
	for i in range(start, no_of_samples):
		filename = basename+str(i)+suffix
		sample = mat_to_nparray(filename)
		print i, sample.shape
		no_channels = sample.shape[1]
		for ch in range(no_channels):
			magnitude = calcFFT(sample[:,ch],fft_width,overlap)[:,:height]
			fig = plt.figure()
			plt.imshow(np.flipud(magnitude.transpose()),vmin=20 , vmax=4000, aspect='auto', interpolation='none')
			fig.savefig( savedir+'/ch'+str(ch)+'/test'+str(i)+suffix+'.png')
		plt.close('all')
		# print "m0", np.amin(magnitude0), np.amax(magnitude0), np.mean(magnitude0), np.average(magnitude0)
		# print "m1", np.amin(magnitude1), np.amax(magnitude1), np.mean(magnitude1), np.average(magnitude1)
		# print "m2", np.amin(magnitude2), np.amax(magnitude2), np.mean(magnitude2), np.average(magnitude2)
		# print "m3", np.amin(magnitude3), np.amax(magnitude3), np.mean(magnitude3), np.average(magnitude3)
		# print "m4", np.amin(magnitude4), np.amax(magnitude4), np.mean(magnitude4), np.average(magnitude4)
		# print "m5", np.amin(magnitude5), np.amax(magnitude5), np.mean(magnitude5), np.average(magnitude5)
		# print "m6", np.amin(magnitude6), np.amax(magnitude6), np.mean(magnitude6), np.average(magnitude6)
		# print "m7", np.amin(magnitude7), np.amax(magnitude7), np.mean(magnitude7), np.average(magnitude7)

		# f, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7), (ax8, ax9, ax10, ax11), (ax12, ax13, ax14, ax15)) = plt.subplots(4, 4, sharex=False, sharey='row')
		# ax0.plot(ch0)
		# ax0.set_title('ch0')
		# ax0.set_ylim([-100,65536]);
		# ax1.plot(ch1)
		# ax1.set_title('ch1')
		# ax1.set_ylim([-100,65536]);
		# ax2.plot(ch2)
		# ax2.set_title('ch2')
		# ax2.set_ylim([-100,65536]);
		# ax3.plot(ch3)
		# ax3.set_title('ch3')
		# ax3.set_ylim([-100,65536]);
		# ax4.plot(ch4)
		# ax4.set_title('ch4')
		# ax4.set_ylim([-100,65536]);
		# ax5.plot(ch5)
		# ax5.set_title('ch5')
		# ax5.set_ylim([-100,65536]);
		# ax6.plot(ch6)
		# ax6.set_title('ch6')
		# ax6.set_ylim([-100,65536]);
		# ax7.plot(ch7)
		# ax7.set_title('ch7')
		# ax7.set_ylim([-100,65536]);

		# ax8.imshow(np.flipud(magnitude0.transpose()),vmin=20 , vmax=8000, aspect='auto', interpolation='none')
		# ax8.set_title('M ch0')
		# ax9.imshow(np.flipud(magnitude1.transpose()),vmin=20 , vmax=8000, aspect='auto', interpolation='none')
		# ax9.set_title('M ch1')
		# ax10.imshow(np.flipud(magnitude2.transpose()),vmin=20 , vmax=8000, aspect='auto', interpolation='none')
		# ax10.set_title('M ch2')
		# ax11.imshow(np.flipud(magnitude3.transpose()),vmin=20 , vmax=8000, aspect='auto', interpolation='none')
		# ax11.set_title('M ch3')
		# ax12.imshow(np.flipud(magnitude4.transpose()),vmin=20 , vmax=8000, aspect='auto', interpolation='none')
		# ax12.set_title('M ch4')
		# ax13.imshow(np.flipud(magnitude5.transpose()),vmin=20 , vmax=8000, aspect='auto', interpolation='none')
		# ax13.set_title('M ch5')
		# ax14.imshow(np.flipud(magnitude6.transpose()),vmin=20 , vmax=8000, aspect='auto', interpolation='none')
		# ax14.set_title('M ch6')
		# ax15.imshow(np.flipud(magnitude7.transpose()),vmin=20 , vmax=8000, aspect='auto', interpolation='none')
		# ax15.set_title('M ch7')
		# plt.pause(1e-9)

		# #Put figure window on top of all other windows
		# f.canvas.manager.window.attributes('-topmost', 1)
		# #After placing figure window on top, allow other windows to be on top of it later
		# f.canvas.manager.window.attributes('-topmost', 0)
		# plt.show()

		# # f, ((ax0, ax1, ax2, ax3)) = plt.subplots(1, 4, sharex='col', sharey='row')

		# # ax0.imshow(np.flipud(magnitude0.transpose()),vmin=4000 , vmax=144000, aspect='auto', interpolation='none')
		# # ax0.set_title('M ch0')
		# # ax0.xaxis.set_minor_locator(AutoMinorLocator(10))
		# # ax0.yaxis.set_minor_locator(AutoMinorLocator(10))
		# # ax1.imshow(np.flipud(magnitude1.transpose()),vmin=4000 , vmax=144000, aspect='auto', interpolation='none')
		# # ax1.set_title('M ch1')
		# # ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
		# # ax1.yaxis.set_minor_locator(AutoMinorLocator(10))
		# # ax2.imshow(np.flipud(magnitude2.transpose()),vmin=4000 , vmax=144000, aspect='auto', interpolation='none')
		# # ax2.set_title('M ch2')
		# # ax2.xaxis.set_minor_locator(AutoMinorLocator(10))
		# # ax2.yaxis.set_minor_locator(AutoMinorLocator(10))
		# # ax3.imshow(np.flipud(magnitude3.transpose()),vmin=4000 , vmax=144000, aspect='auto', interpolation='none')
		# # ax3.set_title('M ch3')
		# # ax3.xaxis.set_minor_locator(AutoMinorLocator(10))
		# # ax3.yaxis.set_minor_locator(AutoMinorLocator(10))
		# # plt.grid(which='minor')
		# # plt.pause(1e-9)

		# # #Put figure window on top of all other windows
		# # f.canvas.manager.window.attributes('-topmost', 1)
		# # #After placing figure window on top, allow other windows to be on top of it later
		# # f.canvas.manager.window.attributes('-topmost', 0)
		# # plt.show()

def read_data_1h(basename, suffix, start):
	sample = np.zeros((0,16))
	filename = basename+'1'+suffix
	test_slice = mat_to_nparray(filename)
	shape = test_slice.shape

	for k in range(6):
		filename = basename+str(start+k)+suffix
		slice = mat_to_nparray(filename)
		no_channels = slice.shape[1]
		sample = np.vstack((sample,slice))
	
	return sample

def read_data(basename, suffix, start):
	filename = basename+str(start)+suffix
	sample = mat_to_nparray(filename)
	return sample

def analyze_1h(basename, suffix, savedir, start, end, fft_width=512, overlap=265, height=64):
	session = 0
	for i in range(start, end, 6):
		print "session nr:",session
		sample = read_data_1h(basename,suffix, i)
		no_channels = sample.shape[1]
		for ch in range(no_channels):
			magnitude = calcFFT(sample[:,ch],fft_width,overlap)[:,:height]
                        print "five percentile", np.percentile(magnitude,5)
			fig = plt.figure()
			plt.imshow(np.flipud(magnitude.transpose()),vmin=20 , vmax=4000, aspect='auto', interpolation='none')
			fig_filename = 'figures/'+savedir+'/ch'+str(ch)+'/test_1h_'+str(i)+suffix+'.png'
			print fig_filename
			fig.savefig(fig_filename)
		session += 1



#analyze_1h('/home/eavsteen/seizure_detection/data/train_1/1_', '_0.mat','train_1', 1, 15)
#analyze_1h('/home/eavsteen/seizure_detection/data/train_1/1_', '_1.mat','train_1', 1, 15)

# analyze('/home/eavsteen/seizure_detection/data/train_1/1_', '_0.mat','train_1', 1, 20)
# analyze('/home/eavsteen/seizure_detection/data/train_1/1_', '_1.mat','train_1', 1, 20)
# analyze('/home/eavsteen/seizure_detection/data/train_2/2_', '_0.mat','train_2', 1, 20)
# analyze('/home/eavsteen/seizure_detection/data/train_2/2_', '_1.mat','train_2', 1, 20)
# analyze('/home/eavsteen/seizure_detection/data/train_3/3_', '_0.mat','train_3', 1, 20)
# analyze('/home/eavsteen/seizure_detection/data/train_3/3_', '_1.mat','train_3', 1, 20)


# analyze('/home/eavsteen/seizure_detection/data/test_1/1_', '.mat','test_1', 1, 10)
# analyze('/home/eavsteen/seizure_detection/data/test_2/2_', '.mat','test_2', 1, 10)
# analyze('/home/eavsteen/seizure_detection/data/test_3/3_', '.mat','test_3', 1, 10)

	
		
