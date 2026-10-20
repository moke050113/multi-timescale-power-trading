"""
电动汽车负荷模型
建模 EV 集群的充放电可调能力, 输出各时段可调功率上下限
"""
import numpy as np
from typing import Dict, Tuple
from ..utils.config import EV_PARAMS, SIM_PARAMS


class EVLoadModel:
    """
    电动汽车集群模型

    建模思路:
        将 N 辆 EV 聚合为一个虚拟储能池,
        计算每个时段可用于充电/放电的功率上下限,
        供交易智能体调用。

    约束:
        - 充放电功率上下限
        - SOC 上下限
        - 到达/离开时间
        - 必须满足最低充电需求 (目标SOC)
    """

    def __init__(self, params: Dict = None):
        self.p = params or EV_PARAMS
        self.num_ev = self.p["num_ev"]
        self.battery_cap = self.p["battery_cap"]       # kWh
        self.charge_power = self.p["charge_power"]       # kW
        self.discharge_power = self.p["discharge_power"] # kW
        self.soc_min = self.p["soc_min"]
        self.soc_max = self.p["soc_max"]
        self.target_soc = self.p["target_soc"]
        self.soc_start = self.p.get("soc_start", self.soc_min)   # 到站初始 SOC
        self.charge_eff = self.p.get("charge_eff", 1.0)          # 网侧→电池
        self.discharge_eff = self.p.get("discharge_eff", 1.0)    # 电池→网侧
        self.arrive_hour = self.p["arrive_hour"]
        self.depart_hour = self.p["depart_hour"]
        self.hours = SIM_PARAMS["hours_per_day"]

    def get_available_capacity(self, hour: int, current_soc: float = None) -> Dict[str, float]:
        """
        计算某时段单辆 EV 的平均可调容量

        Args:
            hour: 当前时段 (0-23)
            current_soc: 当前 SOC (0~1), 若为 None 则取均值

        Returns:
            dict: {
                "charge_max": 最大充电功率 (kW),
                "discharge_max": 最大放电功率 (kW),
                "energy_deficit": 距离目标的缺额电量 (kWh),
                "is_available": 是否在可用时段
            }
        """
        if current_soc is None:
            current_soc = (self.soc_min + self.target_soc) / 2

        # 判断是否在停车时段
        is_available = self._is_parked(hour)

        if not is_available:
            return {
                "charge_max": 0.0,
                "discharge_max": 0.0,
                "energy_deficit": 0.0,
                "is_available": False,
            }

        # 充电功率上限
        charge_max = self.charge_power

        # 放电功率上限 (受 SOC 约束)
        remaining_energy = (current_soc - self.soc_min) * self.battery_cap
        discharge_max = min(self.discharge_power, remaining_energy)

        # 距离目标的缺额
        target_energy = self.target_soc * self.battery_cap
        current_energy = current_soc * self.battery_cap
        energy_deficit = max(0, target_energy - current_energy)

        return {
            "charge_max": charge_max,
            "discharge_max": max(0, discharge_max),
            "energy_deficit": energy_deficit,
            "is_available": True,
        }

    def get_cluster_profile(self, initial_soc: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算全天24时段的集群可调功率上下限

        Args:
            initial_soc: EV 到达时的初始 SOC

        Returns:
            (charge_upper, discharge_upper): 各时段充/放电功率上限 (kW), shape=(24,)
        """
        if initial_soc is None:
            initial_soc = self.soc_min + 0.1

        charge_upper = np.zeros(self.hours)
        discharge_upper = np.zeros(self.hours)

        # 模拟 SOC 变化
        soc = initial_soc
        for h in range(self.hours):
            cap = self.get_available_capacity(h, soc)
            if cap["is_available"]:
                # 集群总功率 = 单车功率 × 车辆数 × 可用比例
                availability = self._get_availability_ratio(h)
                charge_upper[h] = cap["charge_max"] * self.num_ev * availability
                discharge_upper[h] = cap["discharge_max"] * self.num_ev * availability

                # 假设以50%功率充电来估算SOC变化
                charged_energy = cap["charge_max"] * 0.5
                soc = min(self.soc_max, soc + charged_energy / self.battery_cap)
            else:
                charge_upper[h] = 0.0
                discharge_upper[h] = 0.0

        return charge_upper, discharge_upper

    def get_daily_energy_demand(self) -> float:
        """计算集群日均充电需求 (MWh)"""
        avg_deficit = (self.target_soc - self.soc_min) * self.battery_cap * 0.6
        return avg_deficit * self.num_ev / 1000.0  # kWh → MWh

    def _is_parked(self, hour: int) -> bool:
        """判断某时段 EV 是否在停车场"""
        if self.arrive_hour > self.depart_hour:
            return hour >= self.arrive_hour or hour < self.depart_hour
        else:
            return self.arrive_hour <= hour < self.depart_hour

    def _get_availability_ratio(self, hour: int) -> float:
        """
        获取某时段可用 EV 比例 (简化模型)
        到达和离开时段只有部分 EV 可用
        """
        if hour == self.arrive_hour:
            return 0.5
        elif hour == self.depart_hour - 1:
            return 0.5
        elif self._is_parked(hour):
            return 0.9
        return 0.0
