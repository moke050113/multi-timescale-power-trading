"""
工业可转移负荷模型
建模工业用户可在时段间转移的生产负荷
"""
import numpy as np
from typing import Dict, Tuple
from ..utils.config import INDUSTRIAL_LOAD_PARAMS, SIM_PARAMS


class IndustrialLoadModel:
    """
    工业可转移负荷模型

    建模思路:
        工业用户总负荷中有一定比例可在日内时段间转移,
        但需满足:
        - 转移窗口约束 (连续生产时长)
        - 最低生产负荷率
        - 转移惩罚成本
        - 总生产任务量不变

    可调能力:
        将高峰时段的部分负荷转移到低谷时段,
        总用电量不变, 但用电时序改变。
    """

    def __init__(self, params: Dict = None):
        self.p = params or INDUSTRIAL_LOAD_PARAMS
        self.total_power = self.p["total_power"]             # kW
        self.transferable_ratio = self.p["transferable_ratio"]  # 可转移比例
        self.transfer_window = self.p["transfer_window"]     # 转移窗口 (小时)
        self.transfer_cost = self.p["transfer_cost"]         # 转移惩罚 (元/MWh)
        self.min_production = self.p["min_production"]       # 最低负荷率
        self.hours = SIM_PARAMS["hours_per_day"]

        # 可转移功率上限
        self.max_transferable = self.total_power * self.transferable_ratio
        # 不可转移的刚性功率
        self.rigid_power = self.total_power * (1 - self.transferable_ratio)

    def get_base_profile(self) -> np.ndarray:
        """
        生成基础负荷曲线 (未优化前)
        模拟典型工业用电模式: 白天高负荷, 夜间低负荷

        Returns:
            base_profile: 24时段基础负荷 (kW), shape=(24,)
        """
        profile = np.zeros(self.hours)
        for h in range(self.hours):
            if 8 <= h <= 18:
                profile[h] = self.total_power * 0.9
            elif 6 <= h <= 7 or 19 <= h <= 21:
                profile[h] = self.total_power * 0.6
            else:
                profile[h] = self.total_power * self.min_production
        return profile

    def get_transfer_capacity(self) -> Dict[str, float]:
        """
        获取可转移负荷的调节能力

        Returns:
            dict: {
                "max_transferable_power": 可转移最大功率 (kW),
                "rigid_power": 刚性负荷功率 (kW),
                "max_transfer_energy_per_window": 单个转移窗口内最大转移电量 (kWh),
                "transfer_cost": 转移单位电量的惩罚成本 (元/MWh),
            }
        """
        max_energy = self.max_transferable * self.transfer_window
        return {
            "max_transferable_power": self.max_transferable,
            "rigid_power": self.rigid_power,
            "max_transfer_energy_per_window": max_energy,
            "transfer_cost": self.transfer_cost,
        }

    def optimize_transfer(self, price_signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        根据价格信号优化负荷转移方案
        将可转移负荷从高价时段转移到低价时段

        Args:
            price_signal: 24时段价格信号 (元/MWh)

        Returns:
            (optimized_profile, transfer_cost):
                optimized_profile: 优化后负荷曲线 (kW)
                transfer_cost: 转移惩罚总成本 (元)
        """
        base = self.get_base_profile()
        optimized = base.copy()

        # 找出可转移的时段 (高于刚性负荷的部分)
        transferable = base - self.rigid_power
        transferable = np.minimum(transferable, self.max_transferable)
        transferable = np.maximum(transferable, 0)

        # 按价格排序: 从最高价时段减少, 在最低价时段增加
        price_rank_high_to_low = np.argsort(-price_signal)
        price_rank_low_to_high = np.argsort(price_signal)

        total_to_transfer = 0.0
        reductions = np.zeros(self.hours)

        # 从高价时段减少负荷
        for idx in price_rank_high_to_low:
            reducible = transferable[idx] * 0.5  # 最多减少50%的可转移部分
            if reducible > 0:
                reductions[idx] = reducible
                total_to_transfer += reducible

        # 在低价时段增加等量负荷
        additions = np.zeros(self.hours)
        remaining = total_to_transfer
        for idx in price_rank_low_to_high:
            if reductions[idx] > 0:
                continue  # 不在这个时段既减又加
            space = self.max_transferable - (base[idx] - self.rigid_power)
            add = min(remaining, max(0, space))
            additions[idx] = add
            remaining -= add
            if remaining <= 0:
                break

        # 如果低价时段没吸收完, 把剩余的加回去
        if remaining > 0:
            for idx in price_rank_low_to_high:
                if additions[idx] > 0:
                    space = self.max_transferable - additions[idx]
                    extra = min(remaining, space)
                    additions[idx] += extra
                    remaining -= extra
                    if remaining <= 0:
                        break

        optimized = base - reductions + additions

        # 确保不低于刚性负荷
        optimized = np.maximum(optimized, self.rigid_power)

        # 计算转移成本
        transferred_energy = reductions.sum()  # kWh
        transfer_cost = transferred_energy / 1000.0 * self.transfer_cost  # 元

        return optimized, transfer_cost

    def get_adjustment_bounds(self, price_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取负荷调整上下限 (供环境调用)

        Args:
            price_signal: 价格信号

        Returns:
            (lower_bound, upper_bound): 各时段可调整量 (kW), 相对基础曲线
                lower_bound: 可减少的量 (负值)
                upper_bound: 可增加的量 (正值)
        """
        base = self.get_base_profile()
        transferable = np.minimum(base - self.rigid_power, self.max_transferable)
        transferable = np.maximum(transferable, 0)

        # 可减少的上限
        lower = -transferable * 0.5

        # 可增加的上限 (受总转移量守恒约束)
        total_reducible = -lower.sum()
        upper = np.full(self.hours, total_reducible / self.hours)

        return lower, upper
