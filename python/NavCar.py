import numpy as np

from scipy.integrate import odeint

from itertools import chain, izip

from domain import Domain

from shapely.geometry import LineString, LinearRing, Point, MultiPolygon, Polygon, box
import shapely
import shapely.affinity

import copy


class NavCar(Domain):
	_discrete_actions = np.array([[0,-np.pi/8],
		[0, np.pi/8],
		[1,-np.pi/8],
		[1, np.pi/8],
		[-1,-np.pi/8],
		[-1, np.pi/8],
		[1, 0],
		[-1, 0],
		[0,0]])

	"""
	obstacles: a list of list of vertices corresponding to polygons
	goal: a list of list of vertices corresponding to polygons representing the goal region
	bound_box: a 2D array with the mins for each state dim in the first row and the maxs of each 
				state dim in the second row.
	"""
	def __init__(self, goal, limits, obstacles = None, car_params = None):
		if car_params is None:
			self.car = CarDynamics()
		else:
			self.car = CarDynamics(**car_params)

		self.obstacles = obstacles		
		self.goal = goal
		self.limits = limits

	"""
	Compute the reward for one or more state-action pairs.
	"""
	def get_rewards(self, states, actions):
		next_states = self.car.sample_next_state(states, actions)
		rewards = []
		for s in next_states:
			if self._check_collision(s):
				r = self._get_collision_reward(s)
			elif self.goal.contains(*s[:2]):
				r = self._get_goal_reward(s)
			else:
				r = self._get_regular_reward(s)
			rewards.append(r)
		return np.array(rewards)

	def _get_collision_reward(self, state):
		return -1.0

	def _get_goal_reward(self, state):
		return 1.0

	def _get_regular_reward(self, state):
		return -0.01

	def _check_collision(self, state):
		collision = False
		if self.obstacles is not None:
			collision = self.obstacles.contains(*state[:2])
		return collision or not self.limits.contains(*state[:2])

	"""
	Get the current state of the domain. This is meant to be a state of the MDP and not the 
	state of the domain instance.
	"""
	def get_state(self):
		return self.car.state

	"""
	Set the current state of the domain. This is meant to be a state of the MDP and not the 
	state of the domain instance.
	"""
	def set_state(self, state):
		self.car.state = state

	"""
	Get the observation corresponding to some given states. If none are given, get observation
	for current state.
	"""
	def get_observation(self, states = None):
		if states is None:
			states = self.car.state
		return states

	"""
	Generate the next states after executing some actions. There should be as many states as
	there are actions.
	"""
	def sample_next_state(self, states, actions):
		next_states = self.car.sample_next_state(states, actions)
		return next_states
			


	"""
	Sample a start state.
	"""
	def sample_start_state(self):
		return np.zeros_like(self.car.state)

	"""
	Determines if some states are considered terminal. If none are given, return if the current
	state is terminal.
	"""
	def is_terminal(self, states = None):
		if state is not None:
			raise NotImplementedError()
		return self.goal.contains(*self.car.xy) or self._check_collision(self.get_state())

	"""
	Get the dimension of the observations.
	"""
	def get_observation_dim(self):
		return self.get_state_dim()

	"""
	Get the dimension of the states.
	"""
	def get_state_dim(self):
		return self.car.state_range.shape[1]

	"""
	Get the dimension of the actions.
	"""
	def get_action_dim(self):
		return self.car.action_range.shape[1]

	"""
	Get the range of the observation space. This should be an 2D array 
	with shape = (2, observation_dim). With the first row as the min 
	and the second row as the max.
	"""
	def get_observation_range(self):
		return self.get_state_range()

	"""
	Get the range of the state space. This should be an 2D array 
	with shape = (2, state_dim). With the first row as the min 
	and the second row as the max.
	"""
	def get_state_range(self):
		return self.car.state_range

	"""
	Get the range of the action space. This should be an 2D array 
	with shape = (2, action_dim). With the first row as the min 
	and the second row as the max.
	"""
	def get_action_range(self):
		return self.car.action_range

	def get_discrete_actions(self):
		return self._discrete_actions.copy()

class CarDynamics(object):
	action_range = np.array([[-1., -np.pi/6],
							[1., np.pi/6]])

	state_range = np.array([[-100, -100, 0.0, -1],
							[100, 100, np.pi*2, 1]])


	def __init__(self, L=1.0, damp = 1.0, dt = 1.0/20):
		self.L = L
		self.dt = dt
		self.damp = damp
		self._state = np.zeros(4)

	def step(self, action):
		u = np.clip(action, *self.action_range)

		next_state = odeint(self.state_dot, 
			y0 = self._state, 
			t = [0,self.dt], 
			args=(u,))[-1]
		next_state[2] = np.remainder(next_state[2], 2*np.pi)
		self._state[:] = self.sample_next_state(self._state.reshape((1,-1)), action.reshape((1,-1)))

	def sample_next_state(self, states, actions):
		dt = self.dt
		actions = np.clip(actions, *self.action_range)
		result_states = []
		if states.ndim == 1:
			states = states.reshape((1,-1))
		if actions.ndim == 1:
			actions = actions.reshape((1,-1))

		for s, u in zip(states, actions):
			next_state = odeint(self.state_dot, 
				y0 = s, 
				t = [0,dt], 
				args=(u,))[-1]
			next_state[2] = np.remainder(next_state[2], 2*np.pi)
			next_state = np.clip(next_state, *self.state_range)

			result_states.append(next_state)

		return np.vstack(result_states)

	def state_dot(self, q, t, u):
		q_dot = np.zeros_like(q)
		q_dot[:] = [ q[3]*np.cos(q[2]), 
					q[3]*np.sin(q[2]), 
					q[3]/self.L * np.tan(u[1]), 
					u[0] - self.damp*q[3]]
		return q_dot

	def get_velocity(self):
		return self._state[3]*np.cos(self._state[2]), self._state[3]*np.sin(self._state[2])

	def get_states_from_plan(self, plan):
		old_state = self.state
		encountered_states = []
		for p in plan:
			self.step(p)
			encountered_states.append(self.state)

		self._state = old_state
		return encountered_states

	@property
	def state(self):
		return self._state.copy()

	@state.setter
	def state(self, value):
		if value is None:
			raise ValueError()
		self._state[:] = value

	@property
	def xytheta(self):
		return self._state[:3]

	@property
	def xy(self):
		return self._state[:2]

	

class Polygons(object):
	def __init__(self, obstacles = None):
		if obstacles is None:
			obstacles = []
		self.obs = obstacles

	def add_point_obs(self,obs):
		self.obs.append(obs)

	def remove_last(self):
		self.obs = self.obs[:-1]

	def get_shapely_geom(self):
		if len(self.obs) > 0:
			geoms = [ Polygon(vert_list) for vert_list in self.obs]
			return MultiPolygon(geoms)
		else: 
			return None

	def contains(self, x, y):
		geom = self.get_shapely_geom()
		return  (geom is not None and geom.contains(Point(x,y)))

	def load(self, new_obstacles):
		self.obs_width = new_obstacles.obs_width
		self.obs = copy.deepcopy(new_obstacles.obs)



class Track(object):
	def __init__(self, width = 0.3):
		self.points = []
		self.width = 0.3
		self.closed = False

	def add_point(self, x, y):
		self.points.append((x,y))

	def remove_last_point(self):
		self.points = self.points[:-1]
		self.closed = False

	def close(self):
		self.closed = True

	def get_shapely_geom(self, with_width = True):
		track = None
		if len(self.points) >= 2:
			if self.closed and len(self.points) >= 3:
				line = LinearRing(self.points)
			else:
				line = LineString(self.points)
			if with_width:
				track = line.buffer(self.width)
			else:
				track = line
		return track


	def get_loops(self):
		track = self.get_shapely_geom()
		if track is None:
			loops = []
		else:
			loops = [track.exterior.coords] + [ i.coords for i in track.interiors]

		return loops

	def get_closest_line_segment(self, x,y):
		if len(self.points) > 1:
			p = Point(x,y)
			min_dist = np.inf
			closest_line = None
		
			for l in zip(self.points, self.points[1:]+[self.points[0]]):
				l = LineString(l)
				d = l.distance(p)
				if d < min_dist:
					closest_line = l.coords
					min_dist = d
			return list(closest_line)
		else:
			return None

	def contains(self, x,y):
		track = self.get_shapely_geom()
		is_contained = False
		if track is not None:
			is_contained = track.contains(Point(x,y))
		return is_contained


	def load(self, other_track):
		self.points = copy.deepcopy(other_track.points)
		self.width = other_track.width
		self.closed = other_track.closed

	def get_nearest_point(self, x, y):
		track = self.get_shapely_geom(with_width = False)
		if track is not None:
			d = track.project(Point(x,y))
			return track.interpolate(d).coords[0]
		else:
			return (x,y)

	

class Lidar(object):
	def __init__(self, car, track, number_sensors, max_range, angle_offsets = None):
		if angle_offsets is None:
			angle_offsets = np.linspace(0, np.pi*2, number_sensors)
		self.offsets = angle_offsets

		self.sensors = []
		for off in self.offsets:
			self.sensors.append(LineString([(0,0), (np.cos(off)*max_range, np.sin(off)*max_range)]))

		self.car = car
		self.track = track
		self.max_range = max_range

	def __call__(self):
		x,y,theta = self.car.xytheta
		track_geom = self.track.get_shapely_geom()
		obs = np.ones(len(self.sensors))*self.max_range
		if not track_geom is None:
			track_geom = track_geom.boundary
			for i,s in enumerate(self.sensors):
				s = shapely.affinity.rotate(s, theta, origin=(0,0), use_radians=True)
				s = shapely.affinity.translate(s, x, y)
				points = track_geom.intersection(s)
				if not points.is_empty:
					if isinstance(points, shapely.geometry.MultiPoint):
						points = list(points)
					else:
						points = [points]
					distances = [ np.linalg.norm((x-p.x, y-p.y)) for p in points]
					obs[i] = np.min(distances)

		return np.array(obs)

	def get_rel_speeds(self):
		# assuming stationary obstacles
		vel = self.car.get_velocity()
		points = self.get_rel_max_range_points()
		return points.dot(np.array(vel))/np.linalg.norm(points, axis=1)

	def get_rel_max_range_points(self):
		trans_sensors = []
		x,y,theta = self.car.xytheta
		for s in self.sensors:
			s = shapely.affinity.rotate(s, theta, origin=(0,0), use_radians=True)
			trans_sensors.append(s)
		points = np.array([line.coords[1] for line in trans_sensors])
		return points

	def get_collision_points(self):
		x,y,theta = self.car.xytheta
		dist = self()
		trans_sensors = []
		points = self.get_rel_max_range_points()
		points = points * (dist/np.linalg.norm(points, axis=1))[:,None]
		points = points + np.array((x,y))[None,:]
		return points

	@property
	def size(self):
		return len(self.sensors)


class Obstacles(object):
	def __init__(self, obs_width = 0.15):
		self.obs_width = obs_width
		self.obs = []

	def add_point_obs(self,x,y):
		self.obs.append((x,y))

	def remove_last(self):
		self.obs = self.obs[:-1]

	def get_shapely_geom(self):
		if len(self.obs) > 0:
			geoms = [ p.buffer(self.obs_width) for p in self.obs]
			return MultiPolygon(geoms)
		else: 
			return None

	def contains(self, x, y):
		geom = self.get_shapely_geom()
		return  (geom is not None and geom.contains(Point(x,y)))

	def get_loops(self):
		loops = [ p.buffer(self.obs_width).exterior.coords for p in self.obs]
		return loops

	def load(self, new_obstacles):
		self.obs_width = new_obstacles.obs_width
		self.obs = copy.deepcopy(new_obstacles.obs)



class LidarCarTrack(object):
	discrete_actions = np.array([[0,-np.pi/8],
		[0, np.pi/8],
		[1,-np.pi/8],
		[1, np.pi/8],
		[-1,-np.pi/8],
		[-1, np.pi/8],
		[1, 0],
		[-1, 0],
		[0,0]])

	def __init__(self, car, track, lidar):
		self.car = car
		self.track = track
		self.lidar = lidar

	def step(self, u):
		self.car.step(u)
		return self.reward()

	def get_obs(self):
		return np.hstack((self.lidar(), self.lidar.get_rel_speeds()))

	def reward(self, state = None):
		if state is None:
			x,y,theta = self.car.xytheta
		else:
			x,y,theta = state
		tx, ty = self.track.get_nearest_point(x,y)
		l = self.track.get_closest_line_segment(x,y)
		if l is None:
			direction = 0.0
		else:
			l = np.array(l)
			diff = (l[1] - l[0])
			direction = diff.dot((np.cos(theta), np.sin(theta)))*self.car._state[3]/np.linalg.norm(diff)

		d = self.lidar()
		pos_r = np.exp(-np.linalg.norm(np.array((x,y))-np.array((tx, ty)))) * np.max((self.car._state[3],0))
		neg_r = -.5*np.exp(-np.min(d))
		return direction #pos_r + neg_r

	def get_states_rewards_from_plan(self, plan):
		states = self.car.get_states_from_plan(plan)
		rewards = np.zeros(len(states))
		for i,s in enumerate(states):
			if not self.track.contains(*s[:2]):
				break
			rewards[i] = self.reward(s)
		return states, rewards



class LeftRightLidarCarTrack(Domain):
	discrete_actions = np.array([[0,-np.pi/8],
		[0, np.pi/8],
		[1,-np.pi/8],
		[1, np.pi/8],
		[-1,-np.pi/8],
		[-1, np.pi/8],
		[1, 0],
		[-1, 0],
		[0,0]])

	def __init__(self, car, track, lidar, right_reward = True, random_stream = None):
		self.car = car
		self.track = track
		self.lidar = lidar
		self.rnd_stream = random_stream
		self.right_reward = right_reward

	def reward(self, state = None):
		if state is None:
			x,y,theta,v = self.car._state
		else:
			x,y,theta,v = state
		l = self.track.get_closest_line_segment(x,y)
		if l is None:
			direction = 0.0
		else:
			l = np.array(l)
			diff = (l[1] - l[0])
			direction = diff.dot((np.cos(theta), np.sin(theta)))*v/np.linalg.norm(diff)

			p = np.array((x,y)) - l[0]
			side = p[0]*diff[1] -p[1]*diff[0]
			direction *= np.sign(side)
			if not self.right_reward:
				direction *= -1.

			direction *= min(np.abs(v), 1.0)
			if v < 0.0:
				direction = -np.abs(direction)
			else:
				direction = max(0, direction)
			# if (side > 0 and not self.right_reward) or (side <= 0 and self.right_reward):
			# 	direction = min(0, direction)
		return direction #pos_r + neg_r

	def get_states_rewards_from_plan(self, plan):
		states = self.car.get_states_from_plan(plan)
		rewards = np.zeros(len(states))
		for i,s in enumerate(states):
			if not self.track.contains(*s[:2]):
				break
			rewards[i] = self.reward(s)
		return states, rewards


	"""
	Compute the reward for one or more state-action pairs.
	"""
	def get_rewards(self, states, actions):
		if states.ndim > 1:
			return np.array([self.reward(s) for s in states]).reshape((-1,1))
		else:
			return self.reward(states)

	"""
	Get the current state of the domain. This is meant to be a state of the MDP and not the 
	state of the domain instance.
	"""
	def get_state(self):
		return self.car.state

	"""
	Set the current state of the domain. This is meant to be a state of the MDP and not the 
	state of the domain instance.
	"""
	def set_state(self, state):
		self.car.state = state

	"""
	Get the observation corresponding to some given states. If none are given, get observation
	for current state.
	"""
	def get_observation(self, states = None):
		if states is None:
			return np.hstack((self.lidar(), self.lidar.get_rel_speeds()))
		else:
			raise NotImplementedError

	"""
	Get the dimension of the observations.
	"""
	def get_observation_dim(self):
		return self.get_observation_range().shape[1]

	"""
	Get the dimension of the states.
	"""
	def get_state_dim(self):
		return self.get_state_range().shape[1]

	"""
	Get the dimension of the actions.
	"""
	def get_action_dim(self):
		return self.get_action_range().shape[1]

	"""
	Get the range of the observation space. This should be an 2D array 
	with shape = (2, observation_dim). With the first row as the min 
	and the second row as the max.
	"""
	def get_observation_range(self):
		state_range = self.get_state_range()
		min_obs = np.hstack((np.zeros((1, self.lidar.size)), np.ones((1,self.lidar.size))*state_range[0,3]))
		max_obs = np.hstack((np.ones((1, self.lidar.size))*self.lidar.max_range, np.ones((1,self.lidar.size))*state_range[1,3]))
		return np.vstack((min_obs, max_obs))

	"""
	Get the range of the state space. This should be an 2D array 
	with shape = (2, state_dim). With the first row as the min 
	and the second row as the max.
	"""
	def get_state_range(self):
		state_range = self.car.state_range.copy()
		geom = self.track.get_shapely_geom()
		if not geom is None:
			bounds = geom.bounds
			state_range[:2,:2] = np.array(bounds).reshape((2,2))
		return state_range

	"""
	Get the range of the action space. This should be an 2D array 
	with shape = (2, action_dim). With the first row as the min 
	and the second row as the max.
	"""
	def get_action_range(self):
		return self.car.action_range.copy()

	"""
	Sample a start state.
	"""
	def sample_start_state(self):
		state = np.zeros(4)
		if len(self.track.points) > 0:
			state[:2] = self.track.points[0]
		if len(self.track.points) > 1:
			direction = np.array(self.track.points[1]) - np.array(self.track.points[0])
			state[2] = np.arctan2(direction[1],direction[0])
		return state

	"""
	Generate the next states after executing some actions. There should be as many states as
	there are actions.
	"""
	def sample_next_state(self, states, actions):
		old_car_state = self.car.state
		if self.rnd_stream is not None:
			actions = actions + self.rnd_stream.normal(0, np.array([0.1, 0.15*np.pi/4]), size = actions.shape)
		if states.ndim > 1:
			next_states = []
			for s, a in izip(states, actions):
				self.car.state = s
				self.car.step(a)
				next_states.append(self.car.state)
			self.car.state = old_car_state
			return np.vstack(next_states)
		else:
			self.car.state = states
			self.car.step(actions)
			next_state = self.car.state
			self.car.state = old_car_state
			return next_state

	"""
	Determines if some states are considered terminal. If none are given, return if the current
	state is terminal.
	"""
	def is_terminal(self, states = None):
		if states is None:
			return not self.track.contains(*self.car.xytheta[:2])
		else:
			raise NotImplementedError

	def get_discrete_actions(self):
		return self.discrete_actions.copy()