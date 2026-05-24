# pg_asnd.py - Performance-Guided Adaptive SND (PG-ASND)
# 基于性能反馈的自适应社会规范多样性控制器
"""
核心思想：
1. 性能停滞 → 增加多样性（促进探索）
2. 性能提升 → 保持当前多样性（继续当前策略）
3. 性能下降 → 减少多样性（促进收敛）

关键创新：
- 自适应目标值：des不再固定，而是根据性能动态调整
- 双向控制：可以增加或减少多样性
- 性能窗口：使用滑动窗口平滑性能判断
"""

import os
import torch
import torch.nn.functional as F
from typing import List, Optional, Deque
from collections import deque
import numpy as np


class PerformanceGuidedASND:
    """
    Performance-Guided Adaptive SND Controller
    
    根据训练性能自动调整多样性目标值
    """
    
    def __init__(
        self,
        initial_des: float = 0.05,
        des_min: float = 0.01,
        des_max: float = 0.15,
        tau: float = 0.01,
        clip_range: tuple = (0.5, 2.0),
        performance_window: int = 50,
        adjustment_rate: float = 0.002,
        stagnation_threshold: float = 0.01,
        logit_clip: float = 12.0,
        eps: float = 1e-8,
    ):
        """
        Args:
            initial_des: 初始多样性目标值
            des_min: 目标值下限
            des_max: 目标值上限
            tau: EMA平滑系数
            clip_range: alpha裁剪范围
            performance_window: 性能评估窗口大小
            adjustment_rate: 目标值调整速率
            stagnation_threshold: 性能停滞判断阈值（相对变化率）
            logit_clip: logits裁剪范围
            eps: 数值稳定性
        """
        self.snd_des = float(initial_des)
        self.des_min = float(des_min)
        self.des_max = float(des_max)
        self.tau = float(tau)
        self.clip_min, self.clip_max = clip_range
        self.performance_window = performance_window
        self.adjustment_rate = float(adjustment_rate)
        self.stagnation_threshold = float(stagnation_threshold)
        self.logit_clip = float(logit_clip)
        self.eps = float(eps)
        
        # 状态变量
        self.snd_ema: Optional[torch.Tensor] = None
        self.last_alpha: float = 1.0
        self._runtime_alpha: Optional[float] = None
        
        # 性能追踪
        self.performance_history: Deque[float] = deque(maxlen=performance_window)
        self.des_history: List[float] = []
        
        # 状态追踪
        self.state = "neutral"  # "exploring", "exploiting", "neutral"
        self.steps_since_adjustment = 0
        self.min_steps_between_adjustments = 10000  # 防止频繁调整
        
        print(f"[PG-ASND] Initialized: des={self.snd_des:.3f}, "
              f"range=[{self.des_min}, {self.des_max}], "
              f"window={performance_window}")
    
    # ==================== 多样性计算 ====================
    
    @staticmethod
    def _to_probs(logits_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [F.softmax(L.float(), dim=-1) for L in logits_list]
    
    def _compute_action_var(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        """动作分布方差作为多样性度量"""
        probs_stack = torch.stack(probs_list, dim=0)
        var = probs_stack.var(dim=0)
        return var.mean()
    
    def _update_ema(self, cur: torch.Tensor) -> torch.Tensor:
        cur = cur.detach()
        if self.snd_ema is None:
            self.snd_ema = cur
        else:
            self.snd_ema = self.tau * cur + (1.0 - self.tau) * self.snd_ema
        return self.snd_ema
    
    # ==================== 性能分析 ====================
    
    def update_performance(self, reward: float):
        """更新性能历史"""
        self.performance_history.append(reward)
        self.steps_since_adjustment += 1
    
    def _analyze_performance(self) -> str:
        """
        分析性能趋势
        Returns: "improving", "stagnating", "declining"
        """
        if len(self.performance_history) < self.performance_window // 2:
            return "neutral"
        
        history = list(self.performance_history)
        n = len(history)
        
        # 分成前后两半比较
        first_half = np.mean(history[:n//2])
        second_half = np.mean(history[n//2:])
        
        # 计算相对变化
        if abs(first_half) < self.eps:
            relative_change = second_half - first_half
        else:
            relative_change = (second_half - first_half) / (abs(first_half) + self.eps)
        
        if relative_change > self.stagnation_threshold:
            return "improving"
        elif relative_change < -self.stagnation_threshold:
            return "declining"
        else:
            return "stagnating"
    
    def _adjust_target(self):
        """根据性能趋势调整目标值"""
        if self.steps_since_adjustment < self.min_steps_between_adjustments:
            return
        
        trend = self._analyze_performance()
        old_des = self.snd_des
        
        if trend == "stagnating":
            # 性能停滞 → 增加多样性促进探索
            self.snd_des = min(self.des_max, self.snd_des + self.adjustment_rate)
            self.state = "exploring"
        elif trend == "declining":
            # 性能下降 → 减少多样性促进收敛
            self.snd_des = max(self.des_min, self.snd_des - self.adjustment_rate)
            self.state = "exploiting"
        else:  # improving
            # 性能提升 → 保持当前策略
            self.state = "neutral"
        
        if abs(old_des - self.snd_des) > self.eps:
            self.steps_since_adjustment = 0
            print(f"[PG-ASND] Adjusted: des={old_des:.4f} → {self.snd_des:.4f} ({trend})")
        
        self.des_history.append(self.snd_des)
    
    # ==================== Alpha控制 ====================
    
    def set_runtime_alpha(self, alpha: Optional[float]):
        self._runtime_alpha = None if alpha is None else float(alpha)
    
    def get_runtime_alpha(self) -> float:
        return float(self.last_alpha if self._runtime_alpha is None else self._runtime_alpha)
    
    def compute_alpha(self, logits_list: List[torch.Tensor]) -> tuple:
        """
        计算多样性和alpha
        Returns: (snd_ema, alpha_eff, alpha_raw, snd_des)
        """
        probs = self._to_probs(logits_list)
        snd_cur = self._compute_action_var(probs)
        snd_ema = self._update_ema(snd_cur)
        
        # 调整目标值
        self._adjust_target()
        
        # 计算alpha
        if self.snd_des <= 0 or snd_ema is None:
            alpha_raw = 1.0
        else:
            alpha_raw = float((self.snd_des + self.eps) / (float(snd_ema) + self.eps))
        
        alpha_eff = max(self.clip_min, min(self.clip_max, alpha_raw))
        self.last_alpha = float(alpha_eff)
        
        return float(snd_ema), float(alpha_eff), float(alpha_raw), float(self.snd_des)
    
    def rescale_logits(self, logits_list: List[torch.Tensor]):
        """兼容SNDRescaler接口"""
        snd_ema, alpha_eff, alpha_raw, snd_des = self.compute_alpha(logits_list)
        return logits_list, snd_ema, alpha_eff, alpha_raw
    
    @property
    def mode(self):
        return f"pg-adaptive({self.state})"
    
    # ==================== Logits缩放 ====================
    
    def scale_centered(self, logits_list: List[torch.Tensor], alpha: Optional[float] = None) -> List[torch.Tensor]:
        """概率空间中心化缩放"""
        if alpha is None:
            alpha = self.last_alpha
        a = float(max(self.clip_min, min(self.clip_max, float(alpha))))
        
        P = torch.stack(self._to_probs(logits_list), dim=0)
        bar = P.mean(dim=0).clamp(1e-8, 1.0)
        
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
    
    # ==================== 正则化损失 ====================
    
    def compute_diversity_loss(self, coef: float = 0.2) -> float:
        """计算多样性正则化损失"""
        if self.snd_ema is None:
            return 0.0
        return coef * (float(self.snd_ema) - self.snd_des) ** 2
    
    # ==================== 状态信息 ====================
    
    def get_state_info(self) -> dict:
        """获取当前状态信息"""
        return {
            "snd_des": self.snd_des,
            "snd_ema": float(self.snd_ema) if self.snd_ema is not None else 0.0,
            "alpha": self.last_alpha,
            "state": self.state,
            "trend": self._analyze_performance() if len(self.performance_history) > 10 else "neutral",
        }


# ==================== 简化版：基于规则的自适应SND ====================

class RuleBasedASND:
    """
    基于简单规则的自适应SND，更易于调试和理解
    
    规则：
    1. 如果 snd_ema < des * 0.5 → 大幅增加alpha（放大多样性）
    2. 如果 snd_ema < des * 0.8 → 适度增加alpha
    3. 如果 snd_ema > des * 1.2 → 适度减少alpha
    4. 如果 snd_ema > des * 1.5 → 大幅减少alpha
    5. 否则 → alpha ≈ 1.0（保持现状）
    """
    
    def __init__(
        self,
        snd_des: float = 0.05,
        tau: float = 0.01,
        base_clip: tuple = (0.5, 2.0),
        logit_clip: float = 12.0,
        eps: float = 1e-8,
    ):
        self.snd_des = float(max(0.0, snd_des))
        self.tau = float(tau)
        self.clip_min, self.clip_max = base_clip
        self.logit_clip = float(logit_clip)
        self.eps = float(eps)
        
        self.snd_ema: Optional[torch.Tensor] = None
        self.last_alpha: float = 1.0
        
        print(f"[RB-ASND] Initialized: des={self.snd_des:.3f}, clip=[{self.clip_min}, {self.clip_max}]")
    
    @staticmethod
    def _to_probs(logits_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [F.softmax(L.float(), dim=-1) for L in logits_list]
    
    def _compute_action_var(self, probs_list: List[torch.Tensor]) -> torch.Tensor:
        probs_stack = torch.stack(probs_list, dim=0)
        return probs_stack.var(dim=0).mean()
    
    def _update_ema(self, cur: torch.Tensor) -> torch.Tensor:
        cur = cur.detach()
        if self.snd_ema is None:
            self.snd_ema = cur
        else:
            self.snd_ema = self.tau * cur + (1.0 - self.tau) * self.snd_ema
        return self.snd_ema
    
    def compute_alpha(self, logits_list: List[torch.Tensor]) -> tuple:
        """基于规则计算alpha"""
        probs = self._to_probs(logits_list)
        snd_cur = self._compute_action_var(probs)
        snd_ema = self._update_ema(snd_cur)
        snd_ema_val = float(snd_ema)
        
        if self.snd_des <= 0:
            return snd_ema_val, 1.0, 1.0, self.snd_des
        
        ratio = snd_ema_val / (self.snd_des + self.eps)
        
        # 基于规则的alpha计算
        if ratio < 0.5:
            # 多样性严重不足 → 大幅放大
            alpha = 1.8
        elif ratio < 0.8:
            # 多样性不足 → 适度放大
            alpha = 1.3
        elif ratio > 1.5:
            # 多样性过高 → 大幅压缩
            alpha = 0.6
        elif ratio > 1.2:
            # 多样性偏高 → 适度压缩
            alpha = 0.8
        else:
            # 接近目标 → 保持
            alpha = 1.0
        
        alpha_eff = max(self.clip_min, min(self.clip_max, alpha))
        self.last_alpha = alpha_eff
        
        return snd_ema_val, alpha_eff, alpha, self.snd_des
    
    def scale_centered(self, logits_list: List[torch.Tensor], alpha: Optional[float] = None) -> List[torch.Tensor]:
        if alpha is None:
            alpha = self.last_alpha
        a = float(max(self.clip_min, min(self.clip_max, float(alpha))))
        
        P = torch.stack(self._to_probs(logits_list), dim=0)
        bar = P.mean(dim=0).clamp(1e-8, 1.0)
        
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
    
    def compute_diversity_loss(self, coef: float = 0.2) -> float:
        if self.snd_ema is None:
            return 0.0
        return coef * (float(self.snd_ema) - self.snd_des) ** 2
    
    def rescale_logits(self, logits_list: List[torch.Tensor]):
        """兼容SNDRescaler接口"""
        snd_ema, alpha_eff, alpha_raw, snd_des = self.compute_alpha(logits_list)
        return logits_list, snd_ema, alpha_eff, alpha_raw
    
    def set_runtime_alpha(self, alpha: Optional[float]):
        if alpha is not None:
            self.last_alpha = float(alpha)
    
    def get_runtime_alpha(self) -> float:
        return self.last_alpha
    
    @property
    def mode(self):
        return "rule-adaptive"


if __name__ == "__main__":
    # 测试
    print("Testing PG-ASND...")
    controller = PerformanceGuidedASND(initial_des=0.05)
    
    # 模拟性能更新
    for i in range(100):
        reward = -50 + i * 0.1 + np.random.randn() * 2
        controller.update_performance(reward)
    
    print(f"State: {controller.get_state_info()}")
    print("Test passed!")
