
"""
This class is meant to combine both a policy format, an optimization method and a domain.
"""
class Trainer(object):
	def __init__(self):
		pass

	"""
	Start the training procedure.
	"""
	def train(self):
		raise NotImplementedError

	"""
	Returns the current policy.
	"""
	def get_policy(self):
		raise NotImplementedError

"""
This class extends the Trainer class to allow for iterative improvements of the policy
"""
class IncrementalTrainer(Trainer):
	def __init__(self):
		pass

	def train_iteration(self):
		raise NotImplementedError

"""
A policy interface.
"""
class Policy(object):
	def get_action(self, o_t):
		raise NotImplementedError

"""
This class extends the Trainer class to allow for retro-active changes to the data.
This changes is encoded by a new, similar domain.
"""
class ModifiableDataTrainer(Trainer):
	def __init__(self):
		pass

	def update_data(self, new_domain):
		raise NotImplementedError

