# snd_rescale.py
import os
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

class SNDRescaler:
    """
    离散动作 SND 控制器，支持多种多样性度量方式：
    - l2: L2距离（默认，原始方法）
    - js: JS散度
    - entropy_var: 策略熵方差
    - action_var: 动作分布方差（agent维方差再取均值）
    - action_std: 动作分布标准差（V7前曾用，替代L2；数值约0.1-0.5，更稳定）
    
    通过环境变量 SND_METRIC 选择度量方式。
    """

    def __init__(
        self,
        snd_des: float,
        tau: float,
        clip_range: tuple,
        logit_clip: float = 12.0,
        eps: float = 1e-8,
        mode: str = "lower",
    ):
        self.snd_des = float(max(0.0, snd_des))
        self.tau = float(tau)
        self.clip_min, self.clip_max = clip_range
        self.logit_clip = float(logit_clip)
        self.eps = float(eps)
        self.mode = str(mode).lower()
        self.snd_ema: Optional[torch.Tensor] = None
        
        # 从环境变量读取度量方式
        self.metric = os.getenv("SND_METRIC", "l2").lower()

        self._runtime_alpha: Optional[float] = None
        self.last_alpha: float = 1.0
        
        # 调试输出：显示初始化配置
        print(f"[SND] Initialized: mode='{self.mode}', metric='{self.metric}', "
              f"des={self.snd_des}, clip=[{self.clip_min}, {self.clip_max}]")

    # ==================== 度量方式 ====================
    
    @staticmethod
    def _to_probs(logits_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [F.softmax(L.float(), dim=-1) for L in logits_list]

    @staticmethod
    def _stack_probs(per_logits: List[torch.Tensor]) -> torch.Tensor:
        probs = [F.softmax(L.float(), dim=-1) for L in per_logits]
        return torch.stack(probs, dim=0)

    def _compute_l2(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        """原始L2距离"""
        n = len(probs_list)
        if n < 2:
            return probs_list[0].new_tensor(0.0)
        acc = probs_list[0].new_tensor(0.0)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                d = (probs_list[i] - probs_list[j]).pow(2).sum(-1).mean()
                acc = acc + torch.sqrt(torch.clamp(d, min=1e-8))
                cnt += 1
        return acc / max(1, cnt)

    def _compute_js(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        """JS散度"""
        n = len(probs_list)
        if n < 2:
            return probs_list[0].new_tensor(0.0)
        acc = probs_list[0].new_tensor(0.0)
        cnt = 0
        for i in range(n):
            for j in range(i + 1, n):
                p_i = probs_list[i].clamp(min=self.eps)
                p_j = probs_list[j].clamp(min=self.eps)
                m = 0.5 * (p_i + p_j)
                kl_im = (p_i * (p_i / m).log()).sum(-1)
                kl_jm = (p_j * (p_j / m).log()).sum(-1)
                js = 0.5 * (kl_im + kl_jm)
                acc = acc + js.mean()
                cnt += 1
        return acc / max(1, cnt)

    def _compute_entropy_var(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        """策略熵方差"""
        entropies = []
        for p in probs_list:
            p_clamped = p.clamp(min=self.eps)
            entropy = -(p_clamped * p_clamped.log()).sum(-1)
            entropies.append(entropy.mean())
        entropy_stack = torch.stack(entropies)
        return entropy_stack.var()

    def _compute_action_var(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        """动作分布方差"""
        probs_stack = torch.stack(probs_list, dim=0)
        var = probs_stack.var(dim=0)
        return var.mean()

    def _compute_action_std(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        """
        动作分布标准差（V7前曾用，替代L2）。
        对 agent 维度求 std，再对 (batch, action) 取均值。
        数值通常在 0.1~0.5，比 L2(约0.04) 大，alpha 调节更稳定。
        """
        if len(probs_list) < 2:
            return probs_list[0].new_tensor(0.0)
        P = torch.stack(probs_list, dim=0)  # [N, B, A]
        std = P.std(dim=0)  # [B, A]
        return std.mean()

    def _compute_diversity(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        """根据metric计算多样性"""
        if self.metric == "js":
            return self._compute_js(probs_list)
        elif self.metric == "entropy_var":
            return self._compute_entropy_var(probs_list)
        elif self.metric == "action_var":
            return self._compute_action_var(probs_list)
        elif self.metric == "action_std":
            return self._compute_action_std(probs_list)
        else:  # 默认 l2
            return self._compute_l2(probs_list)

    # ==================== EMA更新 ====================
    
    def _update_ema(self, cur: torch.Tensor) -> torch.Tensor:
        cur = cur.detach()
        if self.snd_ema is None:
            self.snd_ema = cur
        else:
            self.snd_ema = self.tau * cur + (1.0 - self.tau) * self.snd_ema
        return self.snd_ema

    def compute_snd(self, logits_list: List[torch.Tensor]) -> tuple:
        probs = self._to_probs(logits_list)
        snd_cur = self._compute_diversity(probs)
        snd_ema = self._update_ema(snd_cur)
        return snd_cur, snd_ema

    # ==================== Alpha控制 ====================
    
    def set_runtime_alpha(self, alpha: Optional[float]):
        self._runtime_alpha = None if alpha is None else float(alpha)

    def get_runtime_alpha(self) -> float:
        return float(self.last_alpha if self._runtime_alpha is None else self._runtime_alpha)

    def rescale_logits(self, logits_list: List[torch.Tensor]):
        """返回: (原logits_list, snd_ema, alpha_eff, raw_alpha_for_debug)"""
        _, snd_ema = self.compute_snd(logits_list)
        if (self.snd_des <= 0) or (snd_ema is None):
            alpha = 1.0
            raw = 1.0
        else:
            raw = float((self.snd_des + self.eps) / (float(snd_ema) + self.eps))
            if self.mode == "upper":
                alpha = min(1.0, raw)
            elif self.mode == "lower":
                alpha = max(1.0, raw)
            else:  # exact
                alpha = raw

        if self._runtime_alpha is not None:
            alpha = float(self._runtime_alpha)

        alpha_eff = max(self.clip_min, min(self.clip_max, alpha))
        self.last_alpha = float(alpha_eff)
        # 返回4个值：原logits, snd_ema, alpha_eff, raw（用于调试）
        return logits_list, float(snd_ema), float(alpha_eff), float(raw)

    # ==================== 概率空间缩放 ====================
    
    def scale_centered(self, logits_list: List[torch.Tensor], alpha: Optional[float] = None) -> List[torch.Tensor]:
        """概率空间中心化缩放：P'_i = (1-α)·P̄ + α·P_i"""
        if alpha is None:
            alpha = self.last_alpha
        a = float(max(self.clip_min, min(self.clip_max, float(alpha))))

        P = self._stack_probs(logits_list)
        bar = P.mean(dim=0)
        bar = torch.clamp(bar, 1e-8, 1.0)
        out: List[torch.Tensor] = []
        for i in range(P.size(0)):
            Pi = (1.0 - a) * bar + a * P[i]
            Pi = torch.clamp(Pi, 1e-8, 1.0)
            Pi = Pi / Pi.sum(dim=-1, keepdim=True)
            Li = torch.log(Pi)
            Li = torch.nan_to_num(Li, nan=0.0, posinf=self.logit_clip, neginf=-self.logit_clip)
            Li = Li.clamp(-self.logit_clip, self.logit_clip)
            out.append(Li)
        return out
