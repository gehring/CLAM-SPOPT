from interfaces import Policy

import numpy as np

from itertools import izip

"""
An episode listener interface. This class is meant to keep track of things happening during
an episode. An online agent or a episode recorder should implement this class.
"""
class EpisodeListenerInterface(object):
	def __init__(self):
		pass

	"""
	Start a new episode. If the previous episode was never ended, this method should assume
	the previous episode was ended on a none-terminal state.
	"""
	def start_episode(self, s_t, o_t):
		raise NotImplementedError

	"""
	Step the current episode forward
	"""
	def step_episode(self, a_t, r_t, s_tp1, o_tp1):
		raise NotImplementedError

	"""
	End the current episode. This assumes the episode encountered a terminal transition.
	"""
	def end_episode(self, a_t, r_t):
		raise NotImplementedError


"""
Class used to manage collection of episode listeners. This is usefull to simultaneously record
transitions as well as run learning agents. This class implements an add method for convenience.
More sophisticated listener collection classes should inherit from this class.
"""
class EpisodeListenerCollection(EpisodeListenerInterface):
	def __init__(self, listeners = None):
		if listeners is None:
			listeners = []
		self.listeners = listeners

	def __add__(self, listener):
		assert isinstance(listener, EpisodeListenerInterface), 'only adding episode listeners is implemented'
		if isinstance(listener, EpisodeListenerCollection):
			return EpisodeListenerCollection(self.listeners + listener.listeners)
		else:
			return EpisodeListenerCollection(self.listeners + [listener])

	def __iter__(self):
		return self.listeners.__iter__()


	"""
	Start a new episode. If the previous episode was never ended, this method should assume
	the previous episode was ended on a none-terminal state.
	"""
	def start_episode(self, s_t, o_t):
		for l in self:
			l.start_episode(s_t, o_t)

	"""
	Step the current episode forward
	"""
	def step_episode(self, a_t, r_t, s_tp1, o_tp1):
		for l in self:
			l.step_episode(a_t, r_t, s_tp1, o_tp1)

	"""
	End the current episode. This assumes the episode encountered a terminal transition.
	"""
	def end_episode(self, a_t, r_t):
		for l in self:
			l.end_episode(a_t, r_t)

"""
Simple augmentation to the episode listener interface to allow the convenient use of
the add override. All listeners should inherit from this class.
"""
class EpisodeListener(EpisodeListenerInterface):
	def __add__(self, listener):
		assert isinstance(listener, EpisodeListenerInterface), 'only adding episode listeners is implemented'
		if isinstance(listener, EpisodeListenerCollection):
			return EpisodeListenerCollection([self] + listener.listeners)
		else:
			return EpisodeListenerCollection([self, listener])



"""
An on-line, episodic agent interface using observations only. 
Sub-classes should implement the agent specific methods as well as the policy method,
get_action(self, o_t).
"""
class OnlineAgent(Policy, EpisodeListener):
	def __init__(self):
		pass

	def start_episode(self, s_t, o_t):
		self.start_agent(o_t)

	def step_episode(self, a_t, r_t, s_tp1, o_tp1):
		self.step_agent(a_t, r_t, o_tp1)

	def end_episode(self, a_t, r_t):
		self.end_agent(a_t, r_t)

	def start_agent(self, o_t):
		raise NotImplementedError

	def step_agent(self, a_t, r_t, o_tp1):
		raise NotImplementedError

	def end_agent(self, a_t, r_t):
		raise NotImplementedError

"""
Class responsible for recording seen transitions. Useful for replays.
"""
class RecordingListener(EpisodeListener):
	def __init__(self):
		# list of trajectories, each trajectory is a tuple.
		# In order, list of states, list of observations, list of actions, 
		# lsit of rewards, and bool with value (last transition is terminal). 
		self.trajectories = []

	def start_episode(self, s_t, o_t):
		self.trajectories.append([[s_t],[o_t],[],[],False])

	def step_episode(self, a_t, r_t, s_tp1, o_tp1):
		states, observations, actions, rewards, terminal = self.trajectories[-1]
		states.append(s_tp1)
		observations.append(o_tp1)
		actions.append(a_t)
		rewards.append(r_t)

	def end_episode(self, a_t, r_t):
		states, observations, actions, rewards, terminal = self.trajectories[-1]
		actions.append(a_t)
		rewards.append(r_t)
		self.trajectories[-1][-1] = True

	def __iter__(self):
		for states, observations, actions, rewards, terminal in self.trajectories:
			if len(actions) == 0:
				continue
			S_t = np.vstack(states)
			O_t = np.vstack(observations)
			A_t = np.vstack(actions)
			R_t = np.hstack(rewards)
			is_terminal = np.zeros_like(R_t)
			if terminal:
				S_tp1 = np.vstack((S_t[1:], np.zeros_like(S_t[0])))
				O_tp1 = np.vstack((O_t[1:], np.zeros_like(O_t[0])))
				is_terminal[-1] = 1.0
			else:
				S_tp1 = S_t[1:]
				S_t = S_t[:-1]
				O_tp1 = O_t[1:]
				O_t = O_t[:-1]

			yield S_t, O_t, A_t, R_t, S_tp1, O_tp1, is_terminal

	def num_traj(self):
		return len(self.trajectories)

	def __getitem__(self, val):
		r = RecordingListener()
		r.trajectories = self.trajectories[val]
		return r

	def __len__(self):
		return len(self.trajectories)


class NSamples:
	def __init__(self, trajectories):
		if isinstance(trajectories, RecordingListener):
			S_t, O_t, A_t, R_t, S_tp1, O_tp1, is_terminal = zip(*trajectories)
			S_t = np.concatenate(S_t, axis=0)
			O_t = np.concatenate(O_t, axis=0)
			A_t = np.concatenate(A_t, axis=0)
			S_tp1 = np.concatenate(S_tp1, axis=0)
			O_tp1 = np.concatenate(O_tp1, axis=0)
			R_t = np.concatenate(R_t, axis=0)
			is_terminal = np.concatenate(is_terminal, axis=0)
			self.data = (S_t, O_t, A_t, R_t, S_tp1, O_tp1, is_terminal)
		else:
			self.data = trajectories


	def __iter__(self):
		yield self.data

	def __getitem__(self, val):
		(S_t, O_t, A_t, R_t, S_tp1, O_tp1, is_terminal) = self.data
		samples = NSamples((S_t[val],
						O_t[val],
						A_t[val],
						R_t[val],
						S_tp1[val],
						O_tp1[val],
						is_terminal[val]))
		return samples

	def __len__(self):
		return self.data[0].shape[0]