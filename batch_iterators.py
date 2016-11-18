from nolearn.lasagne import BatchIterator
import numpy as np
import math
import scipy
import global_vars as g
import random

def normalize(x):
	return x/2000*255

def _sldict(arr, sl):
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]

class BI_train_bal_complete(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, seed=42):

		self.batch_size = batch_size
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.m_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']
		self.shuffle = False

	def __call__(self, X, y=None):
		self.y = y
		#print "BI_train_bal_complete called with len(X) =",len(X) 
		self.X = np.arange(len(X))
		return self


	def transform(self, Xb, yb):

		#print 'transforming batch'
		# Select and normalize:
		bs = Xb.shape[0]
		Xb_new = []
		yb_new = []

		for idx in range(bs):
			hour = Xb[idx]
			slice = None
			if yb[idx]:
				hour = hour % (g.ms_seizure_train.shape[0]+g.ms_xtra_seizure_train.shape[0])				#print 'seizure', hour
				if hour >= g.ms_seizure_train.shape[0]:
					#select extra samples
					hour = hour - g.ms_seizure_train.shape[0]
					r = range(g.ms_xtra_seizure_train.shape[2]-self.m_window)
					start = np.random.choice(r)
					slice = g.ms_xtra_seizure_train[hour, :, start:start+self.m_window]
					Xb_new.append(slice)
					yb_new.append(yb[idx])
				else:
					r = xrange(self.m_window)
					start = np.random.choice(r)
					for pos in range(start,g.ms_seizure_train.shape[2]-self.m_window+1,self.m_window):
						slice = g.ms_seizure_train[hour, :, pos:pos+self.m_window]
						Xb_new.append(slice)
						yb_new.append(yb[idx])


			else:
				hour = hour % g.ms_normal_train.shape[0]
				#print 'normal', hour
				# if not g.blacklist_normal_train[hour]:
				r = xrange(self.m_window)
				start = np.random.choice(r)
				for pos in range(start,g.ms_normal_train.shape[2]-self.m_window+1,self.m_window):
					slice = g.ms_normal_train[hour, :, pos:pos+self.m_window]
					Xb_new.append(slice)
					yb_new.append(yb[idx])
				# else:
				# 	print 'blacklist_normal_train blocked', hour

		Xb_new = np.stack(Xb_new)
		Xb_new = normalize(Xb_new)
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, np.stack(yb_new)

class BI_test_bal_complete(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, seed=42):

		self.batch_size = batch_size
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.m_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']
		self.shuffle = False

	def __call__(self, X, y=None):
		self.y = y
		#print "BI_test_bal_complete called with len(X) =",len(X) 
		self.X = np.arange(len(X))
		self.X_orig = X
		return self

	def __iter__(self):
		bs = self.batch_size
		for i in range((self.n_samples + bs - 1) // bs):
			sl = slice(i * bs, (i + 1) * bs)
			Xb = _sldict(self.X, sl)
			Xb_orig = _sldict(self.X_orig, sl) 
			if self.y is not None:
				yb = _sldict(self.y, sl)
			else:
				yb = None
			yield self.transform(Xb, yb, Xb_orig)

	def transform(self, Xb, yb, Xb_orig):

		# Select and normalize:
		bs = Xb.shape[0]
		Xb_new = []
		yb_new = []

		for idx in range(bs):
			label = None
			hour = None
			if yb != None:
				label = yb[idx]
				hour = Xb[idx] 
			else:
				label = Xb_orig[idx,1]
				hour = Xb[idx]
			if label:
				hour = hour % g.ms_seizure_val.shape[0]
				# if not g.blacklist_seizure_val[hour]:
				r = xrange(self.m_window)
				start = np.random.choice(r)
				for pos in range(start,g.ms_seizure_val.shape[2]-self.m_window+1,self.m_window):
					slice = g.ms_seizure_val[hour, :, pos:pos+self.m_window]
					Xb_new.append(slice)
					yb_new.append(yb[idx])
				# else:
				# 	print 'blacklist_seizure_val blocked', hour
			else:
				hour = hour % g.ms_normal_val.shape[0]
				# if not g.blacklist_normal_val[hour]:
				r = xrange(self.m_window)
				start = np.random.choice(r)
				for pos in range(start,g.ms_normal_val.shape[2]-self.m_window+1,self.m_window):
					slice = g.ms_normal_val[hour, :, pos:pos+self.m_window]
					Xb_new.append(slice)
					yb_new.append(yb[idx])
				# else:
				# 	print 'blacklist_normal_val blocked', hour

		Xb_new = normalize(np.stack(Xb_new))
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, np.stack(yb_new)


class BI_new(BatchIterator):

	def __init__(self, batch_size, seed=42):

		self.batch_size = batch_size
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.m_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']
		self.shuffle = False

	def __call__(self, X, y=None):
		if y == None:
			self.y = y 
		else:
			self.y = y[0]
		self.X = X
		return self

	def __iter__(self):
		bs = self.batch_size
		n_normal = len(self.X['normal'])
		n_seizure = len(self.X['seizure'])
		n_xtra_seizure = len(self.X['xtra_seizure'])

		n_max = max([n_normal, n_seizure, n_xtra_seizure])

		if bs > n_max:
			print 'Warning: batch size is larger than the no. normal, seizure and xtra_seizure files.\nSo it does not make sense to increase batch size further'

		for i in range((n_max + bs - 1) // bs):
			Xb = np.empty((3,bs),dtype=np.int32)
			yb = np.empty((3,bs),dtype=np.int32)
			for j in range(bs):
				idx = i*bs+j
				if idx >= n_normal:
					Xb[0,j] = np.random.choice(self.X['normal'])
				else:
					Xb[0,j] = self.X['normal'][idx]
				yb[0,j] = 0
				if idx >= n_seizure:
					Xb[1,j] = np.random.choice(self.X['seizure'])
				else:
					Xb[1,j] = self.X['seizure'][idx]
				yb[1,j] = 1
				if idx >= n_xtra_seizure:
					Xb[2,j] = np.random.choice(self.X['xtra_seizure'])
				else:
					Xb[2,j] = self.X['xtra_seizure'][idx]
				yb[2,j] = 1
			yield self.transform(Xb, yb)


	def transform(self, Xb, yb):

		bs = Xb.shape[0]
		Xb_new = []
		yb_new = []

		for idx in range(bs):
			# Normal samples
			r = xrange(self.m_window)
			start = np.random.choice(r)
			for pos in range(start,g.ms_normal.shape[2]-self.m_window+1,self.m_window):
				slice = g.ms_normal[Xb[0,idx], :, pos:pos+self.m_window]
				Xb_new.append(slice)
				yb_new.append(0)
			# Seizure samples
			r = xrange(self.m_window)
			start = np.random.choice(r)
			for pos in range(start,g.ms_seizure.shape[2]-self.m_window+1,self.m_window):
				slice = g.ms_seizure[Xb[1,idx], :, pos:pos+self.m_window]
				Xb_new.append(slice)
				yb_new.append(1)
			# Extra seizure samples
			r = xrange(g.ms_xtra_seizure.shape[2]-self.m_window)
			start = np.random.choice(r)
			slice = g.ms_xtra_seizure[Xb[2,idx], :, start:start+self.m_window]
			Xb_new.append(slice)
			yb_new.append(1)

		Xb_new = np.stack(Xb_new)
		yb_new = np.stack(yb_new)
		
		Xb_new = Xb_new.astype(np.float32)
		yb_new = yb_new.astype(np.int32)

		return Xb_new, yb_new
