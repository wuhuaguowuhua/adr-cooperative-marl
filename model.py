import numpy as np
import torch
import torch.nn as nn

from distributions import Categorical
from utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class FCNetwork(nn.Module):
    def __init__(self, dims, out_layer=None):
        """
        Simple MLP with ReLU between layers and optional last layer.
        dims: e.g. (obs, h1, h2, ..., hk)
        """
        super().__init__()
        input_size = dims[0]
        h_sizes = dims[1:]

        mods = [nn.Linear(input_size, h_sizes[0])]
        for i in range(len(h_sizes) - 1):
            mods.append(nn.ReLU())
            mods.append(nn.Linear(h_sizes[i], h_sizes[i + 1]))

        if out_layer:
            mods.append(out_layer)

        self.layers = nn.Sequential(*mods)

    def forward(self, x):
        return self.layers(x)

    def hard_update(self, source):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)

    def soft_update(self, source, t: float):
        for target_param, source_param in zip(self.parameters(), source.parameters()):
            target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super().__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        # For non-recurrent policies, keep a dummy size=1 for API compatibility
        return self._hidden_size if self._recurrent else 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is (T*N, feat); unflatten to (T,N,feat)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N)

            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )
                outputs.append(rnn_scores)

            x = torch.cat(outputs, dim=0)
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super().__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class Policy(nn.Module):
    def __init__(self, obs_space, action_space, base=None, base_kwargs=None):
        super().__init__()

        obs_shape = obs_space.shape
        if base_kwargs is None:
            base_kwargs = {}

        self.base = MLPBase(obs_shape[0], **base_kwargs)

        num_outputs = action_space.n
        # Linear head that returns a FixedCategorical and exposes logits(x)
        self.dist = Categorical(self.base.output_size, num_outputs)

        # Per-agent learnable diversity preference (sigmoid -> [0, 1])
        self.diversity_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def diversity_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.diversity_logit)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    # --- Standard API used elsewhere in the codebase --------------------------
    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)   # [B,1]
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)   # [B,1]
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    # --- NEW: logits for SND/diagnostics -------------------------------------
    def actor_logits(self, inputs, rnn_hxs, masks):
        """
        Compute actor logits without sampling.
        Returns:
            value:  [B,1]
            logits: [B, |A|]
            new_rnn_hxs: RNN state
        """
        value, actor_features, new_rnn_hxs = self.base(inputs, rnn_hxs, masks)

        # Preferred path: our head exposes logits(x)
        if hasattr(self.dist, "logits") and callable(getattr(self.dist, "logits")):
            logits = self.dist.logits(actor_features)
        else:
            # Fallback: build a distribution and fetch its logits/probs
            dist = self.dist(actor_features)
            if hasattr(dist, "logits"):
                logits = dist.logits
            elif hasattr(dist, "probs"):
                logits = torch.log(dist.probs + 1e-8)
            else:
                raise AttributeError(
                    "Cannot extract logits from the policy distribution. "
                    "Please implement `logits(x)` in distributions.Categorical "
                    "or expose `.logits`/`.probs` on the returned distribution."
                )

        return value, logits, new_rnn_hxs
