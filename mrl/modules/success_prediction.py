"""
Success Prediction Module
"""

import mrl
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import os
from mrl.replays.online_her_buffer import OnlineHERBuffer

class GoalSuccessPredictor(mrl.Module):
  """Predicts success using a learned discriminator"""

  def __init__(self, batch_size = 50, history_length = 200, optimize_every=250, log_every=5000, k_steps=1):
    super().__init__(
      'success_predictor',
      required_agent_modules=[
        'env', 'replay_buffer', 'goal_discriminator'
      ],
      locals=locals())
    self.log_every = log_every
    self.batch_size = batch_size
    self.history_length = history_length
    self.optimize_every = optimize_every
    self.opt_steps = 0
    self.k_steps = k_steps


  def _setup(self):
    super()._setup()
    assert isinstance(self.replay_buffer, OnlineHERBuffer)
    assert self.env.goal_env
    self.n_envs = self.env.num_envs
    self.optimizer = torch.optim.Adam(self.goal_discriminator.model.parameters())
    self.test_tensor = None

    """ Point Maze Test tensor """
    if self.config['other_args']['env'] == 'pointmaze':
      
      h = 0.1
      x_min, x_max = -0.5, 9.6
      y_min, y_max = -0.5, 9.6
      xx,yy = np.meshgrid(np.arange(x_min, x_max, h),
                              np.arange(y_min, y_max, h))
      goal_test_tensor = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).type(torch.FloatTensor)
      init_state_tensor = torch.zeros((goal_test_tensor.shape[0], 2))

      self.test_tensor = torch.cat((init_state_tensor, goal_test_tensor), 1).to(self.config.device)


  def _optimize(self):
    self.opt_steps += 1

    if len(self.replay_buffer.buffer.trajectories) > self.batch_size and self.opt_steps % self.optimize_every == 0:
      trajs = self.replay_buffer.buffer.sample_trajectories(self.batch_size, group_by_buffer=True, from_m_most_recent=self.history_length)
      successes = np.array([np.any(np.isclose(traj, 0.), axis=0) for traj in trajs[2]])

      start_states = np.array([t[0] for t in trajs[0]])
      behav_goals =  np.array([t[0] for t in trajs[7]])
      states = np.concatenate((start_states, behav_goals), -1)

      targets = self.torch(successes)
      inputs = self.torch(states)

      # k_steps optimization
      for _ in range(self.k_steps):
        # outputs here have not been passed through sigmoid
        outputs = self.goal_discriminator(inputs)
        loss = F.binary_cross_entropy_with_logits(outputs, targets)

        
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      if hasattr(self, 'logger'):
        self.logger.add_histogram('predictions', torch.sigmoid(outputs), self.log_every)
        self.logger.add_histogram('targets', targets, self.log_every)

        if self.test_tensor is not None:
          with torch.no_grad():
            space_pred = torch.sigmoid(self.goal_discriminator(self.test_tensor))
            goals_pred = torch.sigmoid(self.goal_discriminator(inputs))

          self.logger.add_embedding('behav_goals', self.torch(behav_goals) ,self.log_every, upper_tag='success_pred')
          self.logger.add_embedding('success_labels', targets ,self.log_every, upper_tag='success_pred')
          self.logger.add_embedding('goals_pred', goals_pred ,self.log_every, upper_tag='success_pred')
          self.logger.add_embedding('space_pred', space_pred ,self.log_every, upper_tag='success_pred')

  def __call__(self, *states_and_maybe_goals):
    """Input / output are numpy arrays"""
    states = np.concatenate(states_and_maybe_goals, -1)
    return self.numpy(torch.sigmoid(self.goal_discriminator(self.torch(states))))

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
      'opt_state_dict': self.optimizer.state_dict()
    }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    checkpoint = torch.load(path)
    self.optimizer.load_state_dict(checkpoint['opt_state_dict'])
