# -*- coding: utf-8 -*-

class DataSet:
	def __init__(self, no_channels, base_name, set_name, user, trainset, no_files=0, no_seizure_files=0, no_normal_files=0, no_seizure=0, no_normal=0, blacklisted_samples=None):
		self.no_channels = no_channels
		self.no_seizure_files = no_seizure_files
		self.no_normal_files = no_normal_files
		self.no_files = no_files
		self.no_seizure = no_seizure
		self.no_normal = no_normal
		self.no_seizure_files_clean = -1
		self.no_normal_files_clean = -1
		self.no_seizure_clean = -1
		self.no_normal_clean = -1
		self.base_name = base_name
		self.trainset = trainset
		self.set_name = set_name
		self.user = user
		self.blacklisted_samples = blacklisted_samples
		self.whitelist = []
		if self.blacklisted_samples == None:
			self.blacklisted_samples = []
		self.enabled = True
		self.debug_sub_ratio = 1
	def __str__(self):
		return "base_name: %s, set_name: %s, user: %s, trainset: %d, files: %d"%(self.base_name, self.set_name, self.user, self.trainset, self.no_files)
	def noSamples(self):
		return len(self.fileIndices())
	def fileIndices(self):
		if not self.enabled:
			return []
		all_indices = xrange(int(self.no_files * self.debug_sub_ratio))
		filtered_indices = filter(lambda i: i not in self.blacklisted_samples, all_indices)
		return filtered_indices
	def file_indices_whitelist(self):
		all_indices = xrange(int(self.no_files * self.debug_sub_ratio))
		filtered_indices = filter(lambda i: i in self.whitelist, all_indices)
		return filtered_indices
	def fileName(self, index, channel):
		return '%s/%s%d_ch%d.raw'%(self.session_name, self.base_name, index, channel)
	__repr__ = __str__


patient1 = DataSet(no_channels=16,
				base_name="1_",
				set_name="train_1",
				user="patient1",
				trainset=True,
				no_seizure_files = 150,
				no_normal_files = 1152,
				no_seizure = 25,
				no_normal = 192)

patient2 = DataSet(no_channels=16,
				base_name="2_",
				set_name="train_2",
				user="patient2",
				trainset=True,
				no_seizure_files = 150,
				no_normal_files = 2196,
				no_seizure = 25,
				no_normal = 366)

patient3 = DataSet(no_channels=16,
				base_name="3_",
				set_name="train_3",
				user="patient3",
				trainset=True,
				no_seizure_files = 150,
				no_normal_files = 2244,
				no_seizure = 25,
				no_normal = 374)

patient1_extra = DataSet(no_channels=16,
				base_name="1_",
				set_name="test_1",
				user="patient1",
				trainset=True,
				no_files = 1584)

patient2_extra = DataSet(no_channels=16,
				base_name="2_",
				set_name="test_2",
				user="patient2",
				trainset=True,
				no_files = 2256)

patient3_extra = DataSet(no_channels=16,
				base_name="3_",
				set_name="test_3",
				user="patient3",
				trainset=True,
				no_files = 2289)


patient1_test = DataSet(no_channels=16,
				base_name="new_1_",
				set_name="test_1_new",
				user="patient1",
				trainset=False,
				no_files = 216)

patient2_test = DataSet(no_channels=16,
				base_name="new_2_",
				set_name="test_2_new",
				user="patient2",
				trainset=False,
				no_files = 1002)

patient3_test = DataSet(no_channels=16,
				base_name="new_3_",
				set_name="test_3_new",
				user="patient3",
				trainset=False,
				no_files = 690)

#TDOD add a noise session
new_datasets = [patient1_extra, patient2_extra, patient3_extra]

all = [patient1, patient2, patient3, patient1_test, patient2_test, patient3_test]
