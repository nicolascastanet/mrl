from mrl.import_all import *
from argparse import Namespace
import gym
import time


def make_ddpg_agent(base_config=default_ddpg_config,
                    args=Namespace(env='InvertedPendulum-v2',
                                   tb='',
                                   parent_folder='/tmp/mrl',
                                   layers=(256, 256),
                                   num_envs=None),
                    agent_name_attrs=['env', 'seed', 'tb'],
                    **kwargs):

  if callable(base_config):
    base_config = base_config()
  config = base_config

  if hasattr(args, 'num_envs') and args.num_envs is None:
    import multiprocessing as mp
    args.num_envs = max(mp.cpu_count() - 1, 1)

  if not hasattr(args, 'prefix'):
    args.prefix = 'ddpg'
  if not args.tb:
    args.tb = str(time.time())

  merge_args_into_config(args, config)

  config.agent_name = make_agent_name(config, agent_name_attrs, prefix=args.prefix)

  
  base_modules = {
      k: v
      for k, v in dict(module_train=StandardTrain(),
                       module_eval=EpisodicEval(),
                       module_policy=ActorPolicy(),
                       module_logger=Logger(),
                       module_state_normalizer=Normalizer(MeanStdNormalizer()),
                       module_replay=OnlineHERBuffer(),
                       module_action_noise=ContinuousActionNoise(GaussianProcess,
                                                                 std=ConstantSchedule(config.action_noise)),
                       module_algorithm=DDPG()).items() if not k in config
  }

  config.update(base_modules)

  if type(args.env) is str:
    env = lambda: gym.make(args.env)
    eval_env = env
  else:
    env = args.env
    eval_env = env
  
  if hasattr(args, 'eval_env') and args.eval_env is not None:
    if type(args.eval_env) is str:
      eval_env = lambda: gym.make(args.eval_env)
    else:
      eval_env = args.eval_env


    
  config.module_train_env = EnvModule(env, num_envs=config.num_envs, seed=config.seed)
  config.module_eval_env = EnvModule(eval_env, num_envs=config.num_eval_envs, name='eval_env', seed=config.seed + 1138)

  layer_norm = nn.LayerNorm if (hasattr(args, 'layer_norm') and args.layer_norm) else nn.Identity

  e = config.module_eval_env
  config.module_actor = PytorchModel(
      'actor', lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, layer_norm, make_activ(config.activ)), e.action_dim, e.max_action))
  config.module_critic = PytorchModel(
      'critic', lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, layer_norm, make_activ(config.activ)), 1))

  if e.goal_env:
    config.never_done = True # important for standard Gym goal environments, which are never done

  return config


def make_td3_agent(base_config=spinning_up_td3_config,
                   args=Namespace(env='InvertedPendulum-v2',
                                  tb='',
                                  prefix='td3',
                                  parent_folder='/tmp/mrl',
                                  layers=(256, 256),
                                  num_envs=None),
                   agent_name_attrs=['env', 'seed', 'tb'],
                   **kwargs):
  
  config = make_ddpg_agent(base_config, args, agent_name_attrs, **kwargs)
  del config.module_algorithm
  config.module_algorithm = TD3()

  layer_norm = nn.LayerNorm if (hasattr(args, 'layer_norm') and args.layer_norm) else nn.Identity
  
  e = config.module_eval_env
  config.module_critic2 = PytorchModel('critic2',
      lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, layer_norm, make_activ(config.activ), False), 1, False))

  return config


def make_sac_agent(base_config=spinning_up_sac_config,
                   args=Namespace(env='InvertedPendulum-v2',
                                  tb='',
                                  prefix='sac',
                                  parent_folder='/tmp/mrl',
                                  layers=(256, 256),
                                  num_envs=None),
                   agent_name_attrs=['env', 'seed', 'tb'],
                   **kwargs):
  
  config = make_ddpg_agent(base_config, args, agent_name_attrs, **kwargs)
  e = config.module_eval_env
  layer_norm = nn.LayerNorm if (hasattr(args, 'layer_norm') and args.layer_norm) else nn.Identity
  
  del config.module_actor
  del config.module_action_noise
  del config.module_policy
  config.module_policy = StochasticActorPolicy()
  del config.module_algorithm
  config.module_algorithm = SAC()

  config.module_actor = PytorchModel(
      'actor', lambda: StochasticActor(FCBody(e.state_dim + e.goal_dim, args.layers, layer_norm, make_activ(config.activ)), 
        e.action_dim, e.max_action, log_std_bounds = (-20, 2)))

  config.module_critic2 = PytorchModel('critic2',
      lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, layer_norm, make_activ(config.activ), False), 1, False))

  return config



def make_Alice_and_Bob(config):
  """
    Set Alice and Bob policies / replay / algo etc ... from agent config
    TO DO : implem default Agent config
  """
  # Alice
  config.policy_A.required_agent_modules = [
            'actor_A', 'action_noise', 'env', 'replay_buffer_A'
        ]
  config.policy_A.module_name = 'policy_A'
  config.policy_A.actor = config.actor_A
  config.policy_A.replay_buffer = config.replay_A


  config.Alice.required_agent_modules = ['actor_A','critic_A','replay_buffer_A', 'env']
  config.Alice.actor = config.actor_A
  config.Alice.critic = config.critic_A
  config.Alice.replay_buffer = config.replay_A


  # Bob
  config.policy_B.required_agent_modules = [
            'actor_B', 'action_noise', 'env', 'replay_buffer_B'
        ]
  config.policy_B.module_name = 'policy_B'
  config.policy_B.actor = config.actor_B
  config.policy_B.replay_buffer = config.replay_B


  config.Bob.required_agent_modules = ['actor_B','critic_B','replay_buffer_B', 'env']
  config.Bob.actor = config.actor_B
  config.Bob.critic = config.critic_B
  config.Bob.replay_buffer = config.replay_B

  config.evaluation.required_agent_modules = ['policy_B', 'eval_env']
  config.evaluation.policy = config.policy_B
  


  """

  # Alice
  dict_a = {'actor': config.actor._copy({'name':'actor', 
                                        'model_fn':config.actor.model_fn}),
            'critic': config.actor._copy({'name':'critic', 
                                        'model_fn':config.critic.model_fn}),                   
            'replay_buffer': config.replay._copy()
            }

  config.actor_A = dict_a['actor']
  config.critic_A = dict_a['critic']
  config.replay_A = dict_a['replay_buffer']
  
  
  config.policy_A = config.policy._copy(name = 'policy_A')
  config.policy_A.actor = config.actor_A
  config.policy_A.replay_buffer = config.replay_A

  config.Alice = config.algorithm._copy(name = 'Alice')
  config.Alice.actor = config.actor_A
  config.Alice.critic = config.critic_A
  config.Alice.replay_buffer = config.replay_A
  
  

  # Bob
  dict_b = {'actor': config.actor._copy({'name':'actor', 
                                        'model_fn':config.actor.model_fn}),
            'critic': config.actor._copy({'name':'critic', 
                                        'model_fn':config.critic.model_fn}),                   
            'replay_buffer': config.replay._copy()
            }

  config.actor_B = dict_b['actor']
  config.critic_B = dict_b['critic']
  config.replay_B = dict_b['replay_buffer']
  
  config.policy_B = config.policy._copy(name = 'policy_B')
  config.policy_B.actor = config.actor_B
  config.policy_B.replay_buffer = config.replay_B

  config.Bob = config.algorithm._copy(name = 'Bob')
  config.Bob.actor = config.actor_B
  config.Bob.critic = config.critic_B
  config.Bob.replay_buffer = config.replay_B

  # Other config
  config.evaluation.policy = config.policy_B # Only Bob policy when test time"""

  return config