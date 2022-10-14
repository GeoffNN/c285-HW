from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        # Note that this function will not be differentiated since it only uses numpy arrays.
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        obs = ptu.from_numpy(observation)
        distribution = self(obs)
        if sample:
            action = distribution.sample()
        else:
            action = distribution.mean
        return ptu.to_numpy(action)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        mean = self.mean_net(observation)
        logstd = self.logstd.clip(*self.log_std_bounds)
        std = torch.exp(logstd)
        action_distribution = sac_utils.SquashedNormal(mean, std)
        return action_distribution

    def get_action_and_log_prob_diff(self, obs: torch.Tensor):
        distribution = self(obs)
        action = distribution.rsample()
        log_prob = self(obs).log_prob(action).sum(-1)
        return action, log_prob

    def update_actor(self, obs: torch.Tensor, critic):
        action, log_prob = self.get_action_and_log_prob_diff(obs)

        # Update actor
        q_value = torch.min(*critic(obs, action))
        actor_loss = torch.mean(self.alpha * log_prob - q_value)

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        return actor_loss.item()

    def update_temperature(self, obs: torch.Tensor):
        action, log_prob = self.get_action_and_log_prob_diff(obs)
        # TODO(gnegiar): debug why alpha_loss is sometimes negative.
        alpha_loss = torch.mean(-self.alpha * (log_prob + self.target_entropy).detach())

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return alpha_loss.item()

    def update(self, obs: np.ndarray, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        obs = ptu.from_numpy(obs)

        actor_loss = self.update_actor(obs, critic)
        alpha_loss = self.update_temperature(obs)

        return actor_loss, alpha_loss, self.alpha