import mrl
from mrl.utils.misc import AttrDict, flatten_state
import numpy as np
from copy import deepcopy
import time

class StandardTrain(mrl.Module):
  def __init__(self):
    super().__init__('train', required_agent_modules = ['env', 'policy', 'optimize'], locals=locals())

  def _setup(self):
    assert hasattr(self.config, 'optimize_every')
    self.optimize_every = self.config.optimize_every
    self.env_steps = 0
    self.reset_idxs = []

  def __call__(self, num_steps : int, render=False, dont_optimize=False, dont_train=False):
    """
    Runs num_steps steps in the environment, saves collected experiences,
    and trains at every step
    """
    if not dont_train:
      self.agent.train_mode()
    env = self.env
    state = env.state

    for _ in range(num_steps // env.num_envs):
      
      action = self.policy(state)
      next_state, reward, done, info = env.step(action)

      if self.reset_idxs:
        env.reset(self.reset_idxs)
        for i in self.reset_idxs:
          done[i] = True
          if not 'done_observation' in info[i]:
            if isinstance(next_state, np.ndarray):
              info[i].done_observation = next_state[i]
            else:
              for key in next_state:
                info[i].done_observation = {k: next_state[k][i] for k in next_state}
        next_state = env.state
        self.reset_idxs = []

      state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info)
      self.process_experience(experience)
      import ipdb;ipdb.set_trace()

      if render:
        time.sleep(0.02)
        env.render()
      
      for _ in range(env.num_envs):
        self.env_steps += 1
        if self.env_steps % self.optimize_every == 0 and not dont_optimize:
          self.optimize()
    
    # If using MEP prioritized replay, fit the density model
    if self.config.prioritized_mode == 'mep':
      self.prioritized_replay.fit_density_model()
      self.prioritized_replay.update_priority()
  
  def reset_next(self, idxs):
    """Resets specified envs on next step"""
    self.reset_idxs = idxs

  def save(self, save_folder):
    self._save_props(['env_steps'], save_folder)

  def load(self, save_folder):
    self._load_props(['env_steps'], save_folder)




class AspTrain(mrl.Module):
  def __init__(self, max_steps=50):
    super().__init__('train', required_agent_modules = ['env', 'policy_A','policy_B', 'optimize'], locals=locals())

  def _setup(self):
    assert hasattr(self.config, 'optimize_every')
    self.optimize_every = self.config.optimize_every
    self.env_steps = 0
    self.reset_idxs = []
    self.max_steps = max_steps

  def __call__(self, num_ep : int, render=False, dont_optimize=False, dont_train=False):
    """
    Runs num_steps steps in the environment, saves collected experiences,
    and trains at every step
    """
    if not dont_train:
      self.agent.train_mode()
    env = self.env
    state_dim = self.env.state_dim
    num_envs = env.num_envs

    state = env.reset()
    self.alice_traj = []
    self.bob_traj = []

    for _ in range(num_ep // env.num_envs):

      # Alice's turn

      dones_A = np.zeros((num_envs,))
      steps_A = np.zeros((num_envs,))
      goal_A = state  # Goal for Alice is the initial state
      goal_B = np.zeros((num_envs, state_dim)) # Goal for Bob are the final states for Alice

      while not np.all(dones_A):

        state = self.relabel_state(goal_A)
        action = self.policy_A(state)
        next_state, reward, done, info = env.step(action)
        
        for i, (d, info) in enumerate(zip(done, info)):
          if dones_A[i]:
            continue
          steps_A[i] += 1
          if d:
            dones_A[i] = 1
            final_states[i] = state

        state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info)
        experience.dont_record = dones_A
        self.alice_traj.append(experience)     

        if render:
          time.sleep(0.02)
          env.render()

      # Bob's turn

      dones_B = np.zeros((num_envs,))
      steps_B = np.zeros((num_envs,))
      success_B = np.zeros((num_envs,))

      while not np.all(dones_B):

        state = self.relabel_state(goal_B)
        action = self.policy_B(state)
        next_state, reward, done, info = env.step(action)
        
        for i, (d, info) in enumerate(zip(done, info)):
          if dones_B[i]:
            continue
          steps_B[i] += 1
          if d:
            dones_B[i] = 1
            success_B[i] = info['is_success']

        state, experience = debug_vectorized_experience(state, action, next_state, reward, done, info)
        experience.dont_record = dones_B
        self.bob_traj.append(experience)    

        if render:
          time.sleep(0.02)
          env.render()
      
      # Compute Alice's reward based on Bob's experience
      for i, (t_a, t_b) in enumerate(zip(steps_A, steps_B)):
        if not success_B[i]:
          t_b = self.max_steps - t_a
        self.alice_traj[t_a].reward[i] = max(0,t_b-t_a)

      # Now process experiences
      for exp_a, exp_b in zip(self.alice_traj, self.bob_traj):
        self.process_experience(exp_a)
        self.process_experience(exp_b) 
      
      for _ in range(env.num_envs):
        self.env_steps += 1
        if self.env_steps % self.optimize_every == 0 and not dont_optimize:
          self.optimize()
    
    # If using MEP prioritized replay, fit the density model
    if self.config.prioritized_mode == 'mep':
      self.prioritized_replay.fit_density_model()
      self.prioritized_replay.update_priority()
  
  def reset_next(self, idxs):
    """Resets specified envs on next step"""
    self.reset_idxs = idxs

  def relabel_state(self, state, goals):
    """Should be called by the policy module to relabel states with Alice's goals"""

    return {
        'observation': state['observation'],
        'achieved_goal': state['achieved_goal'],
        'desired_goal': goals
    }

  def save(self, save_folder):
    self._save_props(['env_steps'], save_folder)

  def load(self, save_folder):
    self._load_props(['env_steps'], save_folder)

def debug_vectorized_experience(state, action, next_state, reward, done, info):
  """Gym returns an ambiguous "done" signal. VecEnv doesn't 
  let you fix it until now. See ReturnAndObsWrapper in env.py for where
  these info attributes are coming from."""
  experience = AttrDict(
    state = state,
    action = action,
    reward = reward,
    info = info
  )
  next_copy = deepcopy(next_state) # deepcopy handles dict states

  for idx in np.argwhere(done):
    i = idx[0]
    if isinstance(next_copy, np.ndarray):
      next_copy[i] = info[i].done_observation
    else:
      assert isinstance(next_copy, dict)
      for key in next_copy:
        next_copy[key][i] = info[i].done_observation[key]
  
  experience.next_state = next_copy
  experience.trajectory_over = done
  experience.done = np.array([info[i].terminal_state for i in range(len(done))], dtype=np.float32)
  experience.reset_state = next_state
  experience.dont_record = np.zeros(len(reward)) # Record or not trajectory in replay buffer
  
  return next_state, experience