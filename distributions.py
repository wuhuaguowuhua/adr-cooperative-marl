import torch
import torch.nn as nn

from utils import init

"""
Custom Categorical head compatible with this codebase.
- FixedCategorical: wraps torch.distributions.Categorical but:
  * sample() -> [B, 1]
  * log_probs(actions) accepts index [B,1] or one-hot [B,K], returns [B,1]
  * mode() -> [B,1]
- Categorical: a linear head that outputs FixedCategorical and exposes logits(x).
"""

# ---- FixedCategorical --------------------------------------------------------
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        # Torch returns [B]; our code expects [B,1]
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        """
        actions: index [B,1] (Long) or one-hot [B,K] (float/bool)
        return: [B,1]
        """
        a = actions
        # one-hot -> index
        if a.dim() > 1 and a.size(-1) > 1:
            a = a.argmax(dim=-1, keepdim=True)
        a = a.squeeze(-1)
        if a.dtype != torch.long:
            a = a.long()
        lp = super().log_prob(a)  # [B]
        return lp.unsqueeze(-1)   # [B,1]

    def mode(self):
        # keepdim=True to match [B,1]
        return self.probs.argmax(dim=-1, keepdim=True)


# ---- Linear head that returns FixedCategorical -------------------------------
class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        # Return distribution object with logits
        logits = self.linear(x)
        return FixedCategorical(logits=logits)

    def logits(self, x):
        # Exposed for Policy.actor_logits()
        return self.linear(x)
