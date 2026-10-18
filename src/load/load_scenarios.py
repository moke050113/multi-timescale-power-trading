"""
负荷场景构建器
将电动汽车和工业负荷组合为完整的用户侧负荷场景
"""
import numpy as np
from typing import Dict, Tuple
from .ev_model import EVLoadModel
from .industrial_load import IndustrialLoadModel
from ..utils.config import EV_PARAMS, INDUSTRIAL_LOAD_PARAMS, SIM_PARAMS


class LoadScenarioBuilder:
    """
    构建不同柔性负荷充裕度的场景

    场景类型:
        1. 柔性充裕: EV 数量多, 工业可转移比例高
        2. 柔性适中: 默认参数
        3. 柔性受限: EV 数量少, 可转移比例低
        4. 纯刚性: 无柔性资源 (基准对照)
    """

    SCENARIOS = {
        "柔性充裕": {
            "ev": {**EV_PARAMS, "num_ev": 100, "charge_power": 10},
            "industrial": {**INDUSTRIAL_LOAD_PARAMS, "transferable_ratio": 0.5},
        },
        "柔性适中": {
            "ev": EV_PARAMS,
            "industrial": INDUSTRIAL_LOAD_PARAMS,
        },
        "柔性受限": {
            "ev": {**EV_PARAMS, "num_ev": 20, "charge_power": 7},
            "industrial": {**INDUSTRIAL_LOAD_PARAMS, "transferable_ratio": 0.15},
        },
        "纯刚性": {
            "ev": {**EV_PARAMS, "num_ev": 0},
            "industrial": {**INDUSTRIAL_LOAD_PARAMS, "transferable_ratio": 0.0},
        },
        "柔性大规模": {
            "ev": {**EV_PARAMS, "num_ev": 300, "charge_power": 7},
            "industrial": {**INDUSTRIAL_LOAD_PARAMS, "total_power": 600, "transferable_ratio": 0.5},
        },
    }

    def __init__(self, scenario_name: str = "柔性适中"):
        if scenario_name not in self.SCENARIOS:
            raise ValueError(f"未知负荷场景: {scenario_name}, 可选: {list(self.SCENARIOS.keys())}")

        self.scenario_name = scenario_name
        cfg = self.SCENARIOS[scenario_name]

        self.ev_model = EVLoadModel(cfg["ev"])
        self.industrial_model = IndustrialLoadModel(cfg["industrial"])
        self.hours = SIM_PARAMS["hours_per_day"]

    def build_daily_profile(self, ev_initial_soc: float = None) -> Dict:
        """
        构建完整的日负荷曲线

        Returns:
            dict: {
                "total_load": 总负荷曲线 (kW), (24,)
                "ev_charge_upper": EV 充电上限 (kW), (24,)
                "ev_discharge_upper": EV 放电上限 (kW), (24,)
                "industrial_base": 工业基础负荷 (kW), (24,)
                "industrial_optimized": 优化后工业负荷 (kW), (24,)
                "flexible_capacity_mwh": 总柔性可调容量 (MWh)
            }
        """
        # EV 集群
        ev_charge, ev_discharge = self.ev_model.get_cluster_profile(ev_initial_soc)

        # 工业负荷
        # 用平均价格信号做初步优化
        dummy_price = np.ones(self.hours)
        dummy_price[9:13] = 1.3
        dummy_price[18:22] = 1.5
        dummy_price[0:7] = 0.7
        industrial_base = self.industrial_model.get_base_profile()
        industrial_opt, _ = self.industrial_model.optimize_transfer(dummy_price)

        # 总负荷 = 工业负荷 + EV 基础充电负荷
        ev_base_charge = ev_charge * 0.5  # 假设50%充电率
        total_load = industrial_opt + ev_base_charge

        # 柔性容量
        ev_energy = self.ev_model.get_daily_energy_demand()
        industrial_transferable = self.industrial_model.get_transfer_capacity()
        flexible_capacity = (
            ev_energy +
            industrial_transferable["max_transfer_energy_per_window"] * 2 / 1000.0
        )

        return {
            "total_load": total_load,
            "ev_charge_upper": ev_charge,
            "ev_discharge_upper": ev_discharge,
            "industrial_base": industrial_base,
            "industrial_optimized": industrial_opt,
            "flexible_capacity_mwh": flexible_capacity,
        }

    def get_summary(self) -> str:
        """返回场景摘要信息"""
        profile = self.build_daily_profile()
        ev_demand = self.ev_model.get_daily_energy_demand()
        ind_cap = self.industrial_model.get_transfer_capacity()

        return (
            f"负荷场景: {self.scenario_name}\n"
            f"  日均总负荷: {profile['total_load'].sum() / 1000:.1f} MWh\n"
            f"  EV 日充电需求: {ev_demand:.2f} MWh\n"
            f"  工业可转移功率: {ind_cap['max_transferable_power']:.0f} kW\n"
            f"  总柔性容量: {profile['flexible_capacity_mwh']:.2f} MWh"
        )

    @staticmethod
    def list_scenarios():
        return list(LoadScenarioBuilder.SCENARIOS.keys())
