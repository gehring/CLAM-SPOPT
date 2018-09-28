import numpy as np
import scipy.optimize
from interfaces import Policy
from mathtools import sampled_argmax, get_soft_max

import __builtin__

try:
    __builtin__.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    __builtin__.profile = profile

class Planner(Policy):
    """
    Method to update any internal state based on the given model
    """
    def update_model(self, model):
        raise NotImplementedError

class Egreedy(Policy):
	def __init__(self, greedy_policy, discrete_actions, epsilon, random_stream):
		self.eps = epsilon
		self.rnd = random_stream
		self.pi = greedy_policy
		self.actions = discrete_actions

	def get_action(self, o_t):
		if self.rnd.rand(1) < self.eps:
			return self.actions[self.rnd.randint(self.actions.shape[0])]
		else:
			return self.pi.get_action(o_t)

##########################################################################################
"""
Implementation of the low rank planning algorithm starts here
"""
def improve_timestep(t, vals, plan, temp = 0.01):
    if temp is not None:
        plan[t,:] = get_soft_max(t, vals, temp)
    else:
        plan[t,:] = 0.
        plan[t,vals.argmax()] = 1.0
 
@profile
def lr_improve_plan( plan, alphas, betas, lem, H, discount, start_temp = 1.0):
    diff = np.Infinity
    iter_number = 0
    old_val = 0

    ########## magic numbers ###########
    min_iter = 2
    max_iter = 10
    ####################################

    converged = False


    temp = start_temp

    lr_forward_pass(alphas, betas, plan, False, lem, H, discount, temp = temp)
    while iter_number<max_iter:
        old_plan = plan.copy()
        iter_number += 1

        temp *= 0.9

        lr_forward_pass(alphas, betas, plan, True, lem, H, discount, temp = temp)
        lr_backward_pass(alphas, betas, plan, True, lem, H, discount, temp = temp)
        new_val = np.einsum('ij,ij->i', alphas[0], betas[0]).dot(plan[0])
                    

        diff = new_val - old_val

        old_val = new_val

        converged = np.allclose(old_plan, plan, atol = 0.001) or (np.abs(diff) > np.abs(0.001*old_val))
        if converged:
            temp *= 0.1
        if (converged and iter_number >= min_iter and temp<=0.00001):
            break

    lr_forward_pass(alphas, betas, plan, False, lem, H, discount, temp = temp)
    plan_val = np.einsum('ij,ij->i', alphas[0], betas[0]).dot(plan[0])

    return plan, plan_val, alphas, betas, iter_number

def lr_compute_previous_alphas(t, alphas, plan, model, discount):
    Kab, Da, wa = model[:3]
    alphas[t-1,:,:] = (np.tensordot((discount * plan[t])[:,None] * alphas[t], Kab, ((1,0), (2,0)))+ wa)*Da


def lr_compute_next_betas(t, betas, plan, model):
    Kab, Da = model[:2]
    betas[t+1,:,:] = np.tensordot(Kab, plan[t][:,None] * (Da * betas[t]), ((3,1), (1,0)))
                
def lr_forward_pass(alphas, betas, plan, improve_plan, model, H, discount, temp):
    if improve_plan:
            vals = np.einsum('ij,ij->i', alphas[0], betas[0])
            improve_timestep(0, vals, plan)
    for t in xrange(1, H):
        lr_compute_next_betas(t-1, betas, plan, model)
        if improve_plan:
            vals = np.einsum('ij,ij->i', alphas[t], betas[t])
            improve_timestep(t, vals, plan, temp = temp)

def lr_backward_pass(alphas, betas, plan, improve_plan, model, H, discount, temp):
    if improve_plan:
            vals = np.einsum('ij,ij->i', alphas[-1], betas[-1])
            improve_timestep(H-1, vals, plan)
    for t in xrange(H-2, -1, -1):
        lr_compute_previous_alphas(t+1, alphas, plan, model, discount)
        if improve_plan:
            vals = np.einsum('ij,ij->i', alphas[t], betas[t])
            improve_timestep(t, vals, plan, temp = temp)

def lr_compute_all_vals(alphas, betas, plan, lem, H, discount):
    lr_forward_pass(alphas, betas, plan, False, lem, H, discount, temp = 0.0)
    lr_backward_pass(alphas, betas, plan, False, lem, H, discount, temp = 0.0)
    values = np.einsum('ijk,ijk->ij',alphas, betas)
    return values

# def lr_nodata_compute_previous_alpha(t, alphas, plan, model, discount):
#     Kab, Da, wab = model[:3]
#     alphas[t-1,:,:] = np.tensordot((discount * plan[t])[:,None] * alphas[t], Kab, ((1,0), (2,0)))
#     alphas[t-1,:,:] += wab*plan[t][:,None]

# def lr_nodata_backward_pass(alphas, betas, plan, model, H, discount, temp):
#     for t in xrange(H-2, 0, -1):
#         lr_nodata_compute_previous_alpha(t+1, alphas, plan, model, discount)


# def lr_nodata_compute_all_vals(alphas, betas, plan, lem, H, discount):
#     Kab, Da, wab = model[:3]
#     lr_forward_pass(alphas, betas, plan, False, lem, H, discount, temp = 0.0)
#     lr_nodata_backward_pass(alphas, betas, plan, lem, H, discount, temp = 0.0)
#     return values


def lr_gradient_improve_plan(plan_params, alphas, betas, lem, H, discount, learn_rate = 0.1):
    old_val = -np.Infinity
    converged = False
    iter_number = 0

    ################################
    max_iter = 20
    conv_thres = 1e-2
    ################################

    while not converged: 
        max_v = plan_params.max(axis=1)
        rebased_v = plan_params - max_v[:,None]
        plan = np.exp(rebased_v - np.logaddexp.reduce(rebased_v, axis=1)[:,None])

        values = lr_compute_all_vals(alphas, betas, plan, lem, H, discount)

        new_val = values[0].dot(plan[0])
        # print new_val, plan[:5], values[:5]
        if np.abs(new_val - old_val) < np.abs(conv_thres*old_val):
            converged = True
            break
        else:
            old_val = new_val

        if iter_number >= max_iter:
            break

        P = -plan[:,None,:]*plan[:,:,None]
        step_size = P.shape[1]*P.shape[2]
        for i in xrange(P.shape[1]):
            P.flat[i*(P.shape[1]+1)::step_size] += plan[:,i]

        grad = np.einsum('ijk,ik->ij', P, values)
        # print 's', grad[0], P[0], values[0]
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            grad_norm = 1.0
        plan_params = plan_params + grad*(learn_rate/grad_norm)

        iter_number = iter_number + 1

    max_v = plan_params.max(axis=1)
    rebased_v = plan_params - max_v[:,None]
    plan = np.exp(rebased_v - np.logaddexp.reduce(rebased_v, axis=1)[:,None])
    values = lr_compute_all_vals(alphas, betas, plan, lem, H, discount)

    lr_forward_pass(alphas, betas, plan, True, lem, H, discount, temp = None)
    lr_backward_pass(alphas, betas, plan, True, lem, H, discount, temp = None)
    lr_forward_pass(alphas, betas, plan, True, lem, H, discount, temp = None)
    lr_backward_pass(alphas, betas, plan, True, lem, H, discount, temp = None)
    lr_forward_pass(alphas, betas, plan, True, lem, H, discount, temp = None)
    lr_backward_pass(alphas, betas, plan, True, lem, H, discount, temp = None)

    return plan, new_val, alphas, betas, iter_number



class LowRankOpenLoopPlanner(Planner):
    def __init__(self,
                H,
                H_rollout,
                discount,
                discrete_actions,
                action_kernels,
                planning_rank,
                start_temp,
                random_stream):
        self.H = H
        self.H_rollout = H_rollout
        self.action_kernels = action_kernels
        self.planning_rank = planning_rank
        self.stoch_plan = np.ones((H,len(action_kernels)))/len(action_kernels)
        self.discrete_actions = discrete_actions
        self.start_temp = start_temp
        self.discount = discount
        self.embedded_models = None
        self.random_stream = random_stream
    @profile
    def get_action(self, o_t):
        if self.embedded_models is None:
            raise RuntimeError('model was not initialized with the update_model method')
        H = self.H
        Kab, Da, wa, Ua, Va, imp_a, w_ter = self.embedded_models

        self.betas[0] = np.tensordot(Va, o_t, (1,0))

        ### Initialize the expected rewards for the plan ###
        ### Initialize to the immediate reward function ###
        self.alphas[:,:,:] = wa[None,:,:]
        self.alphas[-1,:,:] += w_ter

        self.alphas *= Da[None,:,:]


        self.stoch_plan[:,:] = 1.0/self.stoch_plan.shape[1]
        self.stoch_plan, plan_val, alphas, betas, _ = lr_improve_plan(self.stoch_plan, 
                                                self.alphas, 
                                                self.betas, 
                                                self.embedded_models, 
                                                self.H, 
                                                self.discount, 
                                                start_temp = self.start_temp)
        a = self.discrete_actions[sampled_argmax(self.stoch_plan[0], self.random_stream)]
        return a
    @profile
    def update_model(self, model):
        self.embedded_models = model.generate_embedded_model(self.action_kernels, 
                                            max_rank = self.planning_rank, 
                                            theta = None)
        self.alphas = np.zeros((self.H, len(self.action_kernels), self.embedded_models[1].shape[1]))
        self.betas = np.zeros_like(self.alphas)

        roll_alphas = np.zeros((self.H_rollout, len(self.action_kernels), self.embedded_models[1].shape[1]))
        roll_plan = np.ones((self.H_rollout,len(self.action_kernels)))/len(self.action_kernels)
        lr_backward_pass(roll_alphas,
            None,
            roll_plan,
            False,
            self.embedded_models,
            self.H_rollout,
            self.discount,
            None)
        self.embedded_models[-1][:,:] += roll_alphas[0]



    def evaluate_actions(self, o_t):
        if self.embedded_models is None:
            raise RuntimeError('model was not initialized with the update_model method')
        H = self.H
        Kab, Da, wa, Ua, Va, imp_a, w_ter = self.embedded_models

        self.betas[0] = np.tensordot(Va, o_t, (1,0))

        ### Initialize the expected rewards for the plan ###
        ### Initialize to the immediate reward function ###
        self.alphas[:,:,:] = wa[None,:,:]
        self.alphas[-1,:,:] += w_ter

        self.alphas *= Da[None,:,:]


        self.stoch_plan[:,:] = 1.0/self.stoch_plan.shape[1]
        self.stoch_plan, plan_val, alphas, betas, _ = lr_improve_plan(self.stoch_plan, 
                                                self.alphas, 
                                                self.betas, 
                                                self.embedded_models, 
                                                self.H, 
                                                self.discount, 
                                                start_temp = self.start_temp)
        return np.einsum('ij,ij->i', alphas[0], betas[0])


class LowRankGradientPlanner(Planner):
    def __init__(self,
                H,
                H_rollout,
                discount,
                discrete_actions,
                action_kernels,
                model,
                planning_rank,
                learn_rate,
                random_stream):
        self.H = H
        self.H_rollout = H_rollout
        self.action_kernels = action_kernels
        self.planning_rank = planning_rank
        self.stoch_plan = np.ones((H,len(discrete_actions)))/len(discrete_actions)
        self.plan_param = np.zeros_like(self.stoch_plan)
        self.discrete_actions = discrete_actions
        self.model = model
        self.learn_rate = learn_rate
        self.discount = discount
        self.embedded_models = None
        self.random_stream = random_stream
    @profile
    def get_action(self, o_t):
        stoch_plan, plan_val, alphas, betas = self.plan(o_t)
        a = self.discrete_actions[sampled_argmax(stoch_plan[0], self.random_stream)]
        return a

    @profile
    def update_model(self, model):
        if self.action_kernels is not None:
            self.embedded_models = model.generate_embedded_model( self.action_kernels, 
                                                max_rank = self.planning_rank, 
                                                theta = None)
        else:
            self.embedded_models = model.generate_embedded_model( max_rank = self.planning_rank, 
                                                theta = None)
        self.alphas = np.zeros((self.H, len(self.discrete_actions), self.embedded_models[1].shape[1]))
        self.betas = np.zeros_like(self.alphas)

        if self.H_rollout > 0:
            roll_alphas = np.zeros((self.H_rollout, len(self.discrete_actions), self.embedded_models[1].shape[1]))
            roll_plan = np.ones((self.H_rollout,len(self.discrete_actions)))/len(self.discrete_actions)
            lr_backward_pass(roll_alphas,
                None,
                roll_plan,
                False,
                self.embedded_models,
                self.H_rollout,
                self.discount,
                None)
            self.embedded_models[-1][:,:] = roll_alphas[0]

    def update_rewards(self, new_rewards):
        self.embedded_models = self.model.update_rewards(new_rewards, *self.embedded_models)
        if self.H_rollout > 0:
            roll_alphas = np.zeros((self.H_rollout, len(self.discrete_actions), self.embedded_models[1].shape[1]))
            roll_plan = np.ones((self.H_rollout,len(self.discrete_actions)))/len(self.discrete_actions)
            lr_backward_pass(roll_alphas,
                None,
                roll_plan,
                False,
                self.embedded_models,
                self.H_rollout,
                self.discount,
                None)
            self.embedded_models[-1][:,:] = roll_alphas[0]

    def evaluate_actions(self, o_t):
        if self.embedded_models is None:
            raise RuntimeError('model was not initialized with the update_model method')
        H = self.H
        Kab, Da, wa, Ua, Va, imp_a, w_ter = self.embedded_models

        self.betas[0] = np.tensordot(Va, o_t, (1,0))

        ### Initialize the expected rewards for the plan ###
        ### Initialize to the immediate reward function ###
        self.alphas[:,:,:] = wa[None,:,:]
        self.alphas[-1,:,:] += w_ter

        self.alphas *= Da[None,:,:]

        self.plan_param[:,:] = 0.0
        self.stoch_plan, plan_val, alphas, betas, _ = lr_gradient_improve_plan(self.plan_param, 
                                                self.alphas, 
                                                self.betas, 
                                                self.embedded_models, 
                                                self.H, 
                                                self.discount, 
                                                learn_rate = self.learn_rate)
        return np.einsum('ij,ij->i', alphas[0], betas[0])

    def plan(self, o_t):
        if self.embedded_models is None:
            raise RuntimeError('model was not initialized with the update_model method')
        H = self.H
        Kab, Da, wa, Ua, Va, imp_a, w_ter = self.embedded_models

        self.betas[0] = np.tensordot(Va, o_t, (1,0))

        ### Initialize the expected rewards for the plan ###
        ### Initialize to the immediate reward function ###
        self.alphas[:,:,:] = wa[None,:,:]
        self.alphas[-1,:,:] += w_ter

        self.alphas *= Da[None,:,:]

        self.plan_param[:,:] = 0.0
        self.stoch_plan, plan_val, alphas, betas, iter_number = lr_gradient_improve_plan(self.plan_param, 
                                                self.alphas, 
                                                self.betas, 
                                                self.embedded_models, 
                                                self.H, 
                                                self.discount, 
                                                learn_rate = self.learn_rate)
        # print iter_number
        return self.stoch_plan, plan_val, alphas, betas
"""
Implementation of the low rank planning algorithm ends here
"""
##########################################################################################


def c_compute_next_betas(t, betas, plan, model):
    Ua, Sa, Va, wa = model[:-1]
    betas[t] =  np.einsum('ijk,ik->j', Ua, np.tensordot(Va, betas[t-1], (1,0)) * Sa * plan[t-1][:,None])

def c_forward_pass(alphas, betas, plan, model, H, discount, argmax_step = False):
    Ua, Sa, Va, wa = model[:-1]
    t = 0
    if argmax_step:
            vals = np.einsum('ijk,ik->ij', Ua, np.tensordot(Va, betas[t], (1,0)) * Sa).dot(alphas[t+1])*discount + wa.dot(betas[t])
            plan[t,:] = 0.
            plan[t, vals.argmax()] = 1.0 
    for t in xrange(1, H):
        c_compute_next_betas(t, betas, plan, model)
        if argmax_step:
            vals = np.einsum('ijk,ik->ij', Ua, np.tensordot(Va, betas[t], (1,0)) * Sa).dot(alphas[t+1])*discount + wa.dot(betas[t])
            plan[t,:] = 0.
            plan[t, vals.argmax()] = 1.0 

def c_compute_previous_alphas(t, alphas, plan, model, discount):
    Ua, Sa, Va, wa = model[:-1]
    # print (np.einsum('j,ijk->ij',discount*alphas[t+1], Ua)*Sa).shape
    alphas[t] =  plan[t].dot((np.einsum('ik,ijk->ij', np.tensordot(discount*alphas[t+1], Ua, (0,1))* Sa, Va) + wa))

def c_backward_pass(alphas, betas, plan, model, H, discount, argmax_step = False):
    Ua, Sa, Va, wa = model[:-1]
    t = H
    if argmax_step and t>0:
            vals = np.einsum('ijk,ik->ij', Ua, np.tensordot(Va, betas[t-1], (1,0)) * Sa).dot(alphas[t])*discount + wa.dot(betas[t-1])
            plan[t-1,:] = 0.
            plan[t-1, vals.argmax()] = 1.0
    for t in xrange(H-1, -1, -1):
        c_compute_previous_alphas(t, alphas, plan, model, discount)
        if argmax_step and t>0:
            vals = np.einsum('ijk,ik->ij', Ua, np.tensordot(Va, betas[t-1], (1,0)) * Sa).dot(alphas[t])*discount + wa.dot(betas[t-1])
            plan[t-1,:] = 0.
            plan[t-1, vals.argmax()] = 1.0 

def c_compute_all_vals(alphas, betas, plan, lem, H, discount):
    Ua, Sa, Va, wa = lem[:-1]
    c_forward_pass(alphas, betas, plan, lem, H, discount)
    c_backward_pass(alphas, betas, plan, lem, H, discount)
    rew = np.einsum('ik,jk->ji', wa, betas)
    bprime = np.einsum('ijk,...ik->...ij', Ua, np.einsum('ijk,...j->...ik', Va, betas)*Sa[None,:,:])
    values = rew + np.einsum('ik,ijk->ij', alphas[1:,:]*discount, bprime)
    return values

def c_compute_plan_param_val_grad(plan_params, alphas, betas, lem, H, discount):
    Ua, Sa, Va, wa = lem[:-1]

    plan_params = np.reshape(plan_params, (H,-1))
    max_v = plan_params.max(axis=1)
    rebased_v = plan_params - max_v[:,None]
    plan = np.exp(rebased_v - np.logaddexp.reduce(rebased_v, axis=1)[:,None])

    c_forward_pass(alphas, betas, plan, lem, H, discount)
    c_backward_pass(alphas, betas, plan, lem, H, discount)
    rew = np.einsum('ik,jk->ji', wa, betas)
    bprime = np.einsum('ijk,...ik->...ij', Ua, np.einsum('ijk,...j->...ik', Va, betas)*Sa[None,:,:])
    values = rew + np.einsum('ik,ijk->ij', alphas[1:,:]*discount, bprime)

    P = -plan[:,None,:]*plan[:,:,None]
    step_size = P.shape[1]*P.shape[2]
    for i in xrange(P.shape[1]):
        P.flat[i*(P.shape[1]+1)::step_size] += plan[:,i]

    grad = np.einsum('ijk,ik->ij', P, values)

    return -values[0].dot(plan[0]), -grad.flatten()

def c_gradient_improve_plan(plan_params, alphas, betas, lem, H, discount, learn_rate = 0.1):
    old_val = -np.Infinity
    converged = False
    iter_number = 0

    ################################
    max_iter = 300
    conv_thres = 1e-4
    ################################

    while not converged: 
        max_v = plan_params.max(axis=1)
        rebased_v = plan_params - max_v[:,None]
        plan = np.exp(rebased_v - np.logaddexp.reduce(rebased_v, axis=1)[:,None])

        values = c_compute_all_vals(alphas, betas, plan, lem, H, discount)

        new_val = values[0].dot(plan[0])
        # print new_val, plan[:5], values[:5]
        if np.abs(new_val - old_val) < np.abs(conv_thres*old_val):
            converged = True
            break
        else:
            old_val = new_val

        if iter_number >= max_iter:
            break

        P = -plan[:,None,:]*plan[:,:,None]
        step_size = P.shape[1]*P.shape[2]
        for i in xrange(P.shape[1]):
            P.flat[i*(P.shape[1]+1)::step_size] += plan[:,i]

        grad = np.einsum('ijk,ik->ij', P, values)
        # print 's', grad[0], P[0], values[0]
        # print np.linalg.norm(grad)
        plan_params = plan_params + grad*(learn_rate/np.linalg.norm(grad))

        iter_number = iter_number + 1

    # res = scipy.optimize.minimize(c_compute_plan_param_val_grad, plan_params.flatten(), args = (alphas, betas, lem, H, discount), jac = True)
    # print 'post-opt', -res.fun
    # plan_params = res.x.reshape((H,-1))
    max_v = plan_params.max(axis=1)
    rebased_v = plan_params - max_v[:,None]
    plan = np.exp(rebased_v - np.logaddexp.reduce(rebased_v, axis=1)[:,None])

    values = c_compute_all_vals(alphas, betas, plan, lem, H, discount)
    c_forward_pass(alphas, betas, plan, lem, H, discount, argmax_step = True)
    values = c_compute_all_vals(alphas, betas, plan, lem, H, discount)
    c_backward_pass(alphas, betas, plan, lem, H, discount, argmax_step = True)
    values = c_compute_all_vals(alphas, betas, plan, lem, H, discount)

    return plan, values[0].dot(plan[0]), values, alphas, betas



class CompressedGradientPlanner(Planner):
    def __init__(self,
                H,
                H_rollout,
                discount,
                discrete_actions,
                action_kernels,
                planning_rank,
                learn_rate,
                random_stream):
        self.H = H
        self.H_rollout = H_rollout
        self.action_kernels = action_kernels
        self.planning_rank = planning_rank
        self.stoch_plan = np.ones((H,len(discrete_actions)))/len(discrete_actions)
        self.plan_param = np.zeros_like(self.stoch_plan)
        self.discrete_actions = discrete_actions
        self.learn_rate = learn_rate
        self.discount = discount
        self.model = None
        self.random_stream = random_stream
    @profile
    def get_action(self, o_t):
        if self.model is None:
            raise RuntimeError('model was not initialized with the update_model method')
        H = self.H
        Ua, Sa, Va, wa, w_ter = self.model

        self.betas[0,:] = o_t

        ### Initialize the expected rewards for the plan ###
        ### Initialize to the immediate reward function ###
        self.alphas[:,:] = self.random_stream.normal(0,1e-1, self.alphas.shape)
        self.alphas[-1,:] = w_ter


        self.plan_param[:,:] = 0.0
        self.stoch_plan, plan_val, values, alphas, betas = c_gradient_improve_plan(self.plan_param, 
                                                self.alphas, 
                                                self.betas, 
                                                self.model, 
                                                self.H, 
                                                self.discount, 
                                                learn_rate = self.learn_rate)
        print plan_val
        a = self.discrete_actions[sampled_argmax(self.stoch_plan[0], self.random_stream)]
        return a
    @profile
    def update_model(self, model):
        if self.action_kernels is not None:
            self.model = model.generate_embedded_model( self.action_kernels, 
                                                max_rank = self.planning_rank, 
                                                theta = None)
        else:
            self.model = model.generate_embedded_model( max_rank = self.planning_rank, 
                                                theta = None)
        self.alphas = np.zeros((self.H+1, self.model[0].shape[1]))
        self.betas = np.zeros((self.H, self.model[0].shape[1]))

        if self.H_rollout > 0:
            roll_alphas = np.zeros((self.H_rollout+1, self.alphas.shape[1]))
            roll_plan = np.ones((self.H_rollout,len(self.discrete_actions)))/len(self.discrete_actions)
            c_backward_pass(roll_alphas,
                None,
                roll_plan,
                self.model,
                self.H_rollout,
                self.discount)
            self.model[-1][:] += roll_alphas[0]

    def evaluate_actions(self, o_t):
        if self.embedded_models is None:
            raise RuntimeError('model was not initialized with the update_model method')
        H = self.H
        Ua, Sa, Va, wa, w_ter = self.model

        self.betas[0] = o_t

        ### Initialize the expected rewards for the plan ###
        ### Initialize to the immediate reward function ###
        self.alphas[:,:] = wa[None,:,:]
        self.alphas[-1,:] += w_ter

        self.plan_param[:,:] = 0.0
        self.stoch_plan, plan_val, values, alphas, betas, _ = c_gradient_improve_plan(self.plan_param, 
                                                self.alphas, 
                                                self.betas, 
                                                self.model, 
                                                self.H, 
                                                self.discount, 
                                                learn_rate = self.learn_rate)
        return values[0]
##########################################################################################
"""
Implementation of the low rank value iteration algorithm starts here
"""
def lr_compute_Qa(v_tp1, ra, Ua, betas, discount):
        qa = (ra + np.tensordot(discount * v_tp1, Ua, ((0,), (1,))))
        qa = np.einsum('ij,ijk->ik', qa, betas)
        return qa

def lr_AVI(discount, H, lem, rtol, atol, old_theta = None):
    Uas, Vas, ra, X_tp1 = lem

    betas = np.tensordot(Vas, X_tp1, (1,1))
    if old_theta is None:
    	v_tp1 = np.zeros(X_tp1.shape[0])
    else:
    	v_tp1 = np.max(np.einsum('ij,ijk->ik', old_theta, betas), axis=0)

    for i in xrange(H):
        old_v_tp1 = v_tp1
        v_tp1 = np.max(lr_compute_Qa(v_tp1, ra, Uas, betas, discount), axis=0)
        if np.allclose(old_v_tp1, v_tp1, rtol=rtol, atol=atol):
            break

    return (ra + np.tensordot(discount * v_tp1, Uas, ((0,), (1,))))

@profile
def lr_evaluate_actions(X_t, lem, thetas):
    Uas, Vas, ra, X_tp1 = lem

    if X_t.ndim == 1:
        X_t = X_t.reshape((1,-1))

    betas = np.tensordot(Vas, X_t, (1,1))
    return np.einsum('ij,ijk->ik', thetas, betas)
@profile
def compute_lr_action_models(model, planning_rank, action_kernels):
        if model.CompressedX_t.matrices is None:
            return (np.zeros((len(action_kernels),1, 1)), 
                    np.zeros((len(action_kernels), model.dim, 1)), 
                    np.zeros((len(action_kernels), 1)), 
                    np.zeros((1, model.dim)))

        A_kernel = action_kernels(model.A_t.get_matrix(model.CompressedX_t.matrices[0].shape[0]))
        Xa = [ model.get_actions_model(a_k.reshape((-1,))) for a_k in A_kernel]

        if Xa[0] is None:
            return (np.zeros((len(action_kernels),1, 1)), 
                    np.zeros((len(action_kernels), model.dim, 1)), 
                    np.zeros((len(action_kernels), 1)), 
                    np.zeros((1, model.dim)))

        Fa = []
        for Ua, Sa, Va, w in Xa:
            Da = Sa/(Sa**2 + model.lamb)
            Uap = (w[:,None]*Ua) * Da[None,:]
            Fa.append((Uap, Va))
        Uas, Vas = zip(*Fa)
        Uas, Vas = np.array(Uas), np.array(Vas)


        rank = min(planning_rank, Uas.shape[2])
        Uas, Vas = Uas[:,:,:rank], Vas[:,:,:rank]

        R_t = model.R_t.get_matrix(Uas.shape[1]).squeeze()
        ra = np.tensordot(R_t, Uas, ((0,), (1,)))

        X_tp1 = model.X_tp1.get_matrix(Uas.shape[1])

        return Uas, Vas, ra, X_tp1

class LowRankAVI(Planner):
    def __init__(self, H, discount, model, discrete_actions, action_kernels,
                planning_rank, random_stream, keep_theta = True, 
                converge_rtol = 1e-3, converge_atol = 1e-8):
        self.H = H
        self.discount = discount
        self.action_kernels = action_kernels
        self.discrete_actions = discrete_actions
        self.lem = None
        self.model = model
        self.random_stream = random_stream
        self.planning_rank = planning_rank
        self.keep_theta = keep_theta
        self.thetas = None
        self.rtol = converge_rtol
        self.atol = converge_atol

    @profile
    def get_action(self, o_t):
        if self.lem is None:
            raise RuntimeError('model was not initialized with the update_model method')
        qa = lr_evaluate_actions(o_t, self.lem, self.thetas)
        a = self.discrete_actions[sampled_argmax(qa, self.random_stream)]
        return a

    def evaluate_actions(self, o_t):
    	return lr_evaluate_actions(o_t, self.lem, self.thetas)

    @profile
    def update_model(self, model):
        self.lem = compute_lr_action_models(model, self.planning_rank, self.action_kernels)
        self.thetas = lr_AVI(self.discount, self.H, self.lem, rtol = self.rtol, atol = self.atol, 
                            old_theta = (self.thetas if self.keep_theta else None))

"""
Implementation of the low rank value iteration algorithm ends here
"""
##########################################################################################


##########################################################################################
"""
Implementation of the LSPI algorithm starts here
"""

def solveAb(A, b, lamb):
    U,S,Vt = np.linalg.svd(A, full_matrices = False)
    Sp = S/(lamb + S**2)
    return Vt.T.dot(Sp*U.T.dot(b))

def lstd_solve(discount, phi_t, r_t, phi_tp1, lamb):
    A = phi_t.T.dot(phi_t - discount*phi_tp1)
    b = phi_t.T.dot(r_t)
    return solveAb(A, b, lamb)

class LSPI(Planner):
    def __init__(self, H, discount, lamb, discrete_actions, phi, rnd_stream):
        self.lamb = lamb
        self.phi = phi
        self.discrete_actions = discrete_actions
        self.random_stream = rnd_stream
        self.discount = discount
        self.A, self.b = None, None
        self.H = H
        self.theta = np.zeros(phi.size())
        self.past_theta = np.zeros((5, phi.size()))

    def solve(self, X_t, A_t, R_t, X_tp1, is_terminal):
        self.reset_convergence_test()
        phi_t = self.phi(X_t, A_t)
        for i in xrange(self.H):
            old_theta = self.theta
            self.theta = lstd_solve(self.discount,
                phi_t,
                R_t,
                self.phi(X_tp1, self.get_action(X_tp1))*(1-is_terminal)[:,None],
                self.lamb)
            print i
            if self.convergence_check(self.theta):
                break

    def get_action(self, o_t):
        if o_t.ndim > 1:
            return np.vstack(( self.get_action(o) for o in o_t ))
        else:
            qa = self.evaluate_actions(o_t)
            return self.discrete_actions[sampled_argmax(qa, self.random_stream)]

    def evaluate_actions(self, o_t):
        return self.phi(o_t, self.discrete_actions).dot(self.theta)

    def convergence_check(self, theta):
        i3 = (self.i5 - 2) % 5
        i4 = (self.i5 - 1) % 5

        if i3 > self.i5:
            i3 = np.arange(self.i5, i3+5) % 5
        else:
            i3 = np.arange(i3, self.i5)

        if i4 > self.i5:
            i4 = np.arange(self.i5, i4+5) % 5
        else:
            i4 = np.arange(i4, self.i5)



        avg3, avg4, avg5 = self.avg_theta
        self.past_theta[self.i5] = theta
        self.avg_theta = (np.mean(self.past_theta[i3], axis=0),
            np.mean(self.past_theta[i4], axis=0),
            np.mean(self.past_theta, axis=0))

        if self.i5 == 4:
            self.more_than_five = True
        self.i5 = (self.i5 + 1) % 5

        return ((np.abs([avg3-self.avg_theta[0],
                    avg4-self.avg_theta[1],
                    avg5-self.avg_theta[2]]) < 1e-3).all(axis=1).any()
                and
                self.more_than_five)

    def reset_convergence_test(self):
        self.more_than_five = False
        self.avg_theta = (np.Infinity, np.Infinity, np.Infinity)
        self.i5 = 0

"""
Implementation of the LSPI algorithm ends here
"""
##########################################################################################

class QLearning(Planner):
    def __init__(self,
                learn_rate,
                discount,
                discrete_actions,
                phi,
                max_iterations,
                rnd_stream,
                max_unchange_count = 10,
                converge_rtol = 1e-3,
                converge_atol = 1e-8):
        from representation import ActionToIndex
        self.phi = phi
        self.discrete_actions = discrete_actions
        self.random_stream = rnd_stream
        self.discount = discount
        self.theta = np.zeros((phi.size(), self.discrete_actions.shape[0]))
        self.actiontoindex = ActionToIndex(self.discrete_actions)
        self.learn_rate = learn_rate
        self.max_iterations = max_iterations
        self.rtol = converge_rtol
        self.atol = converge_atol
        self.max_unchange_count = max_unchange_count

    def solve(self, trajs):
        _, X_t, A_t, R_t, _, X_tp1, is_terminal = zip(*trajs)
        X_t = np.vstack(X_t)
        A_t = np.vstack(A_t)
        R_t = np.hstack(R_t)
        X_tp1 = np.vstack(X_tp1)
        is_terminal = np.hstack(is_terminal)

        phi_t = self.phi(X_t)
        phi_tp1 = self.phi(X_tp1)
        Aindex = self.actiontoindex(A_t)[:,0]

        unchanged_count = 0
        oldV_t = None
        converged = False

        for i in xrange(self.max_iterations):
            V_t = phi_t.dot(self.theta)[np.arange(phi_t.shape[0]),Aindex]
            Q_tp1 = phi_tp1.dot(self.theta)
            maxA = Q_tp1.argmax(axis=1)
            delta = R_t + self.discount * Q_tp1[np.arange(Q_tp1.shape[0]), maxA]*(1.-is_terminal) - V_t
            self.theta[:, Aindex] += (self.learn_rate*delta[:,None]*phi_t).T
            if oldV_t is not None and np.allclose(V_t, oldV_t, rtol=self.rtol, atol=self.atol):
                unchanged_count += 1
            else:
                unchanged_count = 0
                oldV_t = V_t
            if unchanged_count >= self.max_unchange_count:
                converged = True
                break

        if not converged:
            print 'max iterations hit with QLearning'

    def get_action(self, o_t):
        if o_t.ndim > 1:
            return np.vstack(( self.get_action(o) for o in o_t ))
        else:
            qa = self.evaluate_actions(o_t)
            return self.discrete_actions[sampled_argmax(qa, self.random_stream)]

    def evaluate_actions(self, o_t):
        return self.phi(o_t).dot(self.theta)