import os
import random

import numpy as np
import torch


def set_random_seed(seed = 12):
	"""
	set random seed to a random number
	"""
	random.seed(seed)
	
	os.environ['PYTHONHASHSEED'] = str(seed)
	
	np.random.seed(seed)

	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
