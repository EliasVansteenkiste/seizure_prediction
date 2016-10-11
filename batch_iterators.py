from nolearn.lasagne import BatchIterator
import numpy as np
import math
import scipy
import global_vars as g

def normalize(x):
	return x/2000*255

def _sldict(arr, sl):
    if isinstance(arr, dict):
        return {k: v[sl] for k, v in arr.items()}
    else:
        return arr[sl]

class BI_skip_droput_train(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, shuffle=False, seed=42):

		self.batch_size = batch_size
		self.shuffle = shuffle
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']


	def transform(self, Xb, yb):

		Xb, yb = super(BI_skip_droput_train, self).transform(Xb, yb)

		# Select and normalize:
		bs = Xb.shape[0]
		new_shape = (bs,g.args.no_channels,self.magnitude_window,self.ceil-self.floor)
		Xb_new = np.empty(new_shape)

		for idx in range(bs):
			if yb[idx]:
				best = -1
				best_border=100
				hour = int(math.floor(np.random.random_sample() * g.magnitudes_seizure_train.shape[0]))
				for k in range(self.no_trials):
					r = xrange(g.magnitudes_seizure_train.shape[2]-self.magnitude_window)
					start = np.random.choice(r)
					slice = g.magnitudes_seizure_train[hour, :, start:start+self.magnitude_window]
					twop = np.percentile(slice,2)
					if twop>0.1:
						best = slice
						break
					else:
						border = scipy.stats.percentileofscore(slice,0.0000001)
						if border<best_border:
							best_border = border
							best = slice
				Xb_new[idx] = best
			else:
				best = -1
				best_border=100
				hour = int(math.floor(np.random.random_sample() * g.magnitudes_normal_train.shape[0]))
				for k in range(self.no_trials):
					r = xrange(g.magnitudes_normal_train.shape[2]-self.magnitude_window)
					start = np.random.choice(r)
					slice = g.magnitudes_normal_train[hour, :, start:start+self.magnitude_window]
					twop = np.percentile(slice,2)
					if twop>0.1:
						best = slice
						break
					else:
						border = scipy.stats.percentileofscore(slice,0.0000001)
						if border<best_border:
							best_border = border
							best = slice
				Xb_new[idx] = best

		Xb_new = normalize(Xb_new)
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, yb

class BI_skip_droput_test(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, shuffle=False, seed=42):

		self.batch_size = batch_size
		self.shuffle = shuffle
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']

	def transform(self, Xb, yb):

		Xb, yb = super(BI_skip_droput_test, self).transform(Xb, yb)

		# Select and normalize:
		bs = Xb.shape[0]
		new_shape = (bs,g.args.no_channels,self.magnitude_window,self.ceil-self.floor)
		Xb_new = np.empty(new_shape)

		for idx in range(bs):
			label = None
			x_rand = None
			if yb != None:
				label = yb[idx]
				x_rand = Xb[idx]
			else:
				label = Xb[idx,1]
				x_rand = Xb[idx,0]
			if label:
				best = -1
				best_border=100
				hour = int(math.floor(x_rand * g.magnitudes_seizure_val.shape[0]))
				for k in range(self.no_trials):
					r = xrange(g.magnitudes_seizure_val.shape[2]-self.magnitude_window)
					start = np.random.choice(r)
					slice = g.magnitudes_seizure_val[hour, :, start:start+self.magnitude_window]
					twop = np.percentile(slice,2)
					if twop>0.1:
						best = slice
						break
					else:
						border = scipy.stats.percentileofscore(slice,0.0000001)
						if border<best_border:
							best_border = border
							best = slice
				Xb_new[idx] = best
			else:
				best = -1
				best_border=100
				hour = int(math.floor(x_rand * g.magnitudes_normal_val.shape[0]))
				for k in range(self.no_trials):
					r = xrange(g.magnitudes_normal_val.shape[2]-self.magnitude_window)
					start = np.random.choice(r)
					slice = g.magnitudes_normal_val[hour, :, start:start+self.magnitude_window]
					twop = np.percentile(slice,2)
					if twop>0.1:
						best = slice
						break
					else:
						border = scipy.stats.percentileofscore(slice,0.0000001)
						if border<best_border:
							best_border = border
							best = slice
				Xb_new[idx] = best

		Xb_new = normalize(Xb_new)
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, yb


class BI_train(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, shuffle=False, seed=42):

		self.batch_size = batch_size
		self.shuffle = shuffle
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']

	def transform(self, Xb, yb):

		Xb, yb = super(BI_train, self).transform(Xb, yb)

		# Select and normalize:
		bs = Xb.shape[0]
		new_shape = (bs,g.args.no_channels,self.magnitude_window,self.ceil-self.floor)
		Xb_new = np.empty(new_shape)

		for idx in range(bs):
			if yb[idx]:
				hour = int(math.floor(np.random.random_sample() * g.magnitudes_seizure_train.shape[0]))
				r = xrange(g.magnitudes_seizure_train.shape[2]-self.magnitude_window)
				start = np.random.choice(r)
				slice = g.magnitudes_seizure_train[hour, :, start:start+self.magnitude_window]
				Xb_new[idx] = slice
			else:
				hour = int(math.floor(np.random.random_sample() * g.magnitudes_normal_train.shape[0]))
				r = xrange(g.magnitudes_normal_train.shape[2]-self.magnitude_window)
				start = np.random.choice(r)
				slice = g.magnitudes_normal_train[hour, :, start:start+self.magnitude_window]
				Xb_new[idx] = slice

		Xb_new = normalize(Xb_new)
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, yb

class BI_test(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, shuffle=False, seed=42):

		self.batch_size = batch_size
		self.shuffle = shuffle
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']

	def transform(self, Xb, yb):

		Xb, yb = super(BI_test, self).transform(Xb, yb)

		# Select and normalize:
		bs = Xb.shape[0]
		new_shape = (bs,g.args.no_channels,self.magnitude_window,self.ceil-self.floor)
		Xb_new = np.empty(new_shape)

		for idx in range(bs):
			label = None
			x_rand = None
			if yb != None:
				label = yb[idx]
				x_rand = Xb[idx]
			else:
				label = Xb[idx,1]
				x_rand = Xb[idx,0]
			if label:
				hour = int(math.floor(x_rand * g.magnitudes_seizure_val.shape[0]))
				r = xrange(g.magnitudes_seizure_val.shape[2]-self.magnitude_window)
				start = np.random.choice(r)
				slice = g.magnitudes_seizure_val[hour, :, start:start+self.magnitude_window]
				Xb_new[idx] = slice
			else:
				hour = int(math.floor(x_rand * g.magnitudes_normal_val.shape[0]))
				r = xrange(g.magnitudes_normal_val.shape[2]-self.magnitude_window)
				start = np.random.choice(r)
				slice = g.magnitudes_normal_val[hour, :, start:start+self.magnitude_window]
				Xb_new[idx] = slice

		Xb_new = normalize(Xb_new)
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, yb

class BI_train_balanced(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, seed=42):

		self.batch_size = batch_size
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']
		self.shuffle = False

	def __call__(self, X, y=None):
		self.y = y
		#print "BI_train_balanced called with len(X) =",len(X) 
		self.X = np.arange(len(X))
		return self


	def transform(self, Xb, yb):

		# Select and normalize:
		bs = Xb.shape[0]
		new_shape = (bs,g.args.no_channels,self.magnitude_window,self.ceil-self.floor)
		Xb_new = np.empty(new_shape)

		for idx in range(bs):
			hour = Xb[idx]
			slice = None
			if yb[idx]:
				hour = hour % g.magnitudes_seizure_train.shape[0]
				r = xrange(g.magnitudes_seizure_train.shape[2]-self.magnitude_window)
				start = np.random.choice(r)
				slice = g.magnitudes_seizure_train[hour, :, start:start+self.magnitude_window]
			else:
				hour = hour % g.magnitudes_normal_train.shape[0]
				r = xrange(g.magnitudes_normal_train.shape[2]-self.magnitude_window)
				start = np.random.choice(r)
				slice = g.magnitudes_normal_train[hour, :, start:start+self.magnitude_window]
			Xb_new[idx] = slice

		Xb_new = normalize(Xb_new)
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, yb

class BI_test_balanced(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, seed=42):

		self.batch_size = batch_size
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']
		self.shuffle = False

	def __call__(self, X, y=None):
		self.y = y
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
		new_shape = (bs,g.args.no_channels,self.magnitude_window,self.ceil-self.floor)
		Xb_new = np.empty(new_shape)

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
				hour = hour % g.magnitudes_seizure_val.shape[0]
				r = xrange(g.magnitudes_seizure_val.shape[2]-self.magnitude_window)
				start = np.random.choice(r)
				slice = g.magnitudes_seizure_val[hour, :, start:start+self.magnitude_window]
			else:
				hour = hour % g.magnitudes_normal_val.shape[0]
				r = xrange(g.magnitudes_normal_val.shape[2]-self.magnitude_window)
				start = np.random.choice(r)
				slice = g.magnitudes_normal_val[hour, :, start:start+self.magnitude_window]
			Xb_new[idx] = slice

		Xb_new = normalize(Xb_new)
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, yb

class BI_train_sch(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, seed=42):

		self.batch_size = batch_size
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']
		self.shuffle = False

	def __call__(self, X, y=None):
		self.y = y
		#print "BI_train_balanced called with len(X) =",len(X) 
		self.X = np.arange(len(X))
		return self


	def transform(self, Xb, yb):

		# Select and normalize:
		bs = Xb.shape[0]
		new_shape = (bs*g.args.no_channels,1,self.magnitude_window,self.ceil-self.floor)
		Xb_new = np.empty(new_shape)
		yb_new = np.empty((bs*g.args.no_channels),dtype=np.int32)

		for idx in range(bs):
			hour = Xb[idx]
			for ch in range(g.args.no_channels):
				slice = None
				if yb[idx]:
					hour = hour % g.magnitudes_seizure_train.shape[0]
					r = xrange(g.magnitudes_seizure_train.shape[2]-self.magnitude_window)
					start = np.random.choice(r)
					slice = g.magnitudes_seizure_train[hour,ch,start:start+self.magnitude_window]
				else:
					hour = hour % g.magnitudes_normal_train.shape[0]
					r = xrange(g.magnitudes_normal_train.shape[2]-self.magnitude_window)
					start = np.random.choice(r)
					slice = g.magnitudes_normal_train[hour,ch, start:start+self.magnitude_window]
				Xb_new[idx*g.args.no_channels+ch] = slice
				yb_new[idx*g.args.no_channels+ch] = yb[idx]

		Xb_new = normalize(Xb_new)
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, yb_new

class BI_test_sch(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, seed=42):

		self.batch_size = batch_size
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']
		self.shuffle = False

	def __call__(self, X, y=None):
		self.y = y
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
		new_shape = (bs*g.args.no_channels,1,self.magnitude_window,self.ceil-self.floor)
		Xb_new = np.empty(new_shape)
		yb_new = None
		if yb != None:
			yb_new = np.empty((bs*g.args.no_channels),dtype=np.int32) 
		else:
			yb_new = yb

		for idx in range(bs):
			label = None
			hour = None
			if yb != None:
				label = yb[idx]
				hour = Xb[idx] 
			else:
				label = Xb_orig[idx,1]
				hour = Xb[idx]
			for ch in range(g.args.no_channels):
				if label:
					hour = hour % g.magnitudes_seizure_val.shape[0]
					r = xrange(g.magnitudes_seizure_val.shape[2]-self.magnitude_window)
					start = np.random.choice(r)
					slice = g.magnitudes_seizure_val[hour, ch, start:start+self.magnitude_window]
				else:
					hour = hour % g.magnitudes_normal_val.shape[0]
					r = xrange(g.magnitudes_normal_val.shape[2]-self.magnitude_window)
					start = np.random.choice(r)
					slice = g.magnitudes_normal_val[hour, ch, start:start+self.magnitude_window]
				Xb_new[idx*g.args.no_channels+ch] = slice
				if yb != None:
					yb_new[idx*g.args.no_channels+ch] = yb[idx]

		Xb_new = normalize(Xb_new)
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, yb_new

class BI_test_sch_sch(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, channel, seed=42):

		self.batch_size = batch_size
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']
		self.shuffle = False
		self.ch = channel

	def __call__(self, X, y=None):
		self.y = y
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
		new_shape = (bs,1,self.magnitude_window,self.ceil-self.floor)
		Xb_new = np.empty(new_shape)

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
				hour = hour % g.magnitudes_seizure_val.shape[0]
				r = xrange(g.magnitudes_seizure_val.shape[2]-self.magnitude_window)
				start = np.random.choice(r)
				slice = g.magnitudes_seizure_val[hour, self.ch, start:start+self.magnitude_window]
			else:
				hour = hour % g.magnitudes_normal_val.shape[0]
				r = xrange(g.magnitudes_normal_val.shape[2]-self.magnitude_window)
				start = np.random.choice(r)
				slice = g.magnitudes_normal_val[hour, self.ch, start:start+self.magnitude_window]
			Xb_new[idx] = slice

		Xb_new = normalize(Xb_new)
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, yb

class BI_train_bal_complete(BatchIterator):

	no_trials = 20

	def __init__(self, batch_size, seed=42):

		self.batch_size = batch_size
		self.random = np.random.RandomState(seed)
		preprocess_params = g.cfg['preprocess']
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']
		self.shuffle = False

	def __call__(self, X, y=None):
		self.y = y
		print "BI_train_bal_complete called with len(X) =",len(X) 
		self.X = np.arange(len(X))
		return self


	def transform(self, Xb, yb):

		# Select and normalize:
		bs = Xb.shape[0]
		Xb_new = []
		yb_new = []

		for idx in range(bs):
			hour = Xb[idx]
			slice = None
			if yb[idx]:
				hour = hour % g.magnitudes_seizure_train.shape[0]
				r = xrange(self.magnitude_window)
				start = np.random.choice(r)
				for pos in range(start,g.magnitudes_seizure_train.shape[2]-self.magnitude_window+1,self.magnitude_window):
					slice = g.magnitudes_seizure_train[hour, :, pos:pos+self.magnitude_window]
					Xb_new.append(slice)
					yb_new.append(yb[idx])
			else:
				hour = hour % g.magnitudes_normal_train.shape[0]
				r = xrange(self.magnitude_window)
				start = np.random.choice(r)
				for pos in range(start,g.magnitudes_normal_train.shape[2]-self.magnitude_window+1,self.magnitude_window):
					slice = g.magnitudes_normal_train[hour, :, pos:pos+self.magnitude_window]
					Xb_new.append(slice)
					yb_new.append(yb[idx])
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
		self.magnitude_window = preprocess_params['magnitude_window']
		self.ceil = preprocess_params['ceil']
		self.floor = preprocess_params['floor']
		self.shuffle = False

	def __call__(self, X, y=None):
		self.y = y
		print "BI_test_bal_complete called with len(X) =",len(X) 
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
				hour = hour % g.magnitudes_seizure_val.shape[0]
				r = xrange(self.magnitude_window)
				start = np.random.choice(r)
				for pos in range(start,g.magnitudes_seizure_val.shape[2]-self.magnitude_window+1,self.magnitude_window):
					slice = g.magnitudes_seizure_val[hour, :, pos:pos+self.magnitude_window]
					Xb_new.append(slice)
					yb_new.append(yb[idx])
			else:
				hour = hour % g.magnitudes_normal_val.shape[0]
				r = xrange(self.magnitude_window)
				start = np.random.choice(r)
				for pos in range(start,g.magnitudes_normal_val.shape[2]-self.magnitude_window+1,self.magnitude_window):
					slice = g.magnitudes_normal_val[hour, :, pos:pos+self.magnitude_window]
					Xb_new.append(slice)
					yb_new.append(yb[idx])

		Xb_new = normalize(np.stack(Xb_new))
		Xb_new = Xb_new.astype(np.float32)
		return Xb_new, np.stack(yb_new)