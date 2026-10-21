"""
柔性负荷执行层

与 flexible_dispatcher (只给功率包络) 的区别: 本模块**推进设备物理状态**并返回
执行后的网侧净负荷, 供市场结算使用。

硬性物理约束 (全部在执行层内落实, 不靠奖励惩罚去兜):
    EV   : 充电功率上限 / 停车时段可用车辆比例 / SOC 上下限;
           离站前必须达到 target_soc; V2G 放电量受"当前可放电量"与
           "离站前还能充回多少"双重限制, 并计入充放电往返损耗
    工业 : 每小时负荷拆为刚性部分与可转移部分; 可转移部分只改变时刻分布,
           **日任务电量守恒**; 增载落点必须在转移窗口内且不超过可转移功率上限

所以"设备按时完成"是构造性保证, 不是优化结果 —— 报告与测试都按这个口径写。
柔性强度只调"多大程度择优", 时刻排序使用已公布的日前价格 (合法信息)。
"""
import numpy as np
from typing import Dict, List

from ..utils.config import SIM_PARAMS


class FlexibleExecutor:
    """EV 集群 + 工业可转移负荷的代表日执行模型 (× days_per_month 得月度电量)。"""

    def __init__(
        self,
        ev_model,
        ind_model,
        include_flexible: bool = True,
        hours: int = SIM_PARAMS["hours_per_day"],
        days_per_month: int = SIM_PARAMS["days_per_month"],
    ):
        self.ev = ev_model
        self.ind = ind_model
        self.include_flexible = include_flexible
        self.hours = int(hours)
        self.days_per_month = int(days_per_month)

        # ---- EV 集群参数 (kWh / kW) ----
        self.ev_n = int(getattr(ev_model, "num_ev", 0))
        self.ev_cap = self.ev_n * float(getattr(ev_model, "battery_cap", 0.0))
        self.soc_min = float(getattr(ev_model, "soc_min", 0.2))
        self.soc_max = float(getattr(ev_model, "soc_max", 0.9))
        self.target_soc = float(getattr(ev_model, "target_soc", 0.8))
        self.soc_start = float(getattr(ev_model, "soc_start", self.soc_min))
        self.eta_in = float(getattr(ev_model, "charge_eff", 1.0))
        self.eta_out = float(getattr(ev_model, "discharge_eff", 1.0))
        self.arrive_hour = int(getattr(ev_model, "arrive_hour", 18))
        self.depart_hour = int(getattr(ev_model, "depart_hour", 8))

        self.avail = np.array([self._availability_ratio(h) for h in range(self.hours)])
        self.charge_limit = float(getattr(ev_model, "charge_power", 0.0)) * self.ev_n * self.avail
        self.discharge_limit = (float(getattr(ev_model, "discharge_power", 0.0)) * self.ev_n
                                * self.avail)
        self.parked: List[int] = self._parked_sequence()

        # ---- 工业参数 (kW, 每小时电量数值上等于 kWh) ----
        self.ind_base = np.asarray(self.ind.get_base_profile(), dtype=float)
        self.rigid_power = float(getattr(self.ind, "rigid_power",
                                self.ind_base.max() * 0.0))
        self.max_transferable = float(getattr(self.ind, "max_transferable", 0.0))
        self.transfer_window = int(getattr(self.ind, "transfer_window", self.hours))
        self.transfer_cost = float(getattr(self.ind, "transfer_cost", 0.0))
        # 每小时拆分: 刚性部分不动, 可转移部分可重排
        self.ind_fixed = np.minimum(self.ind_base, self.rigid_power)
        self.ind_flex = np.maximum(self.ind_base - self.ind_fixed, 0.0)
        self.ind_room = np.maximum(self.max_transferable - self.ind_flex, 0.0)
        self.ind_task = float(self.ind_base.sum())

    # ------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------
    def neutral_profile(self) -> np.ndarray:
        """柔性强度为 0 的日网侧净负荷 (MWh/h), 作为合同参考电量 R_m 的形状来源。"""
        return self.execute_day(np.zeros(self.hours), 0.0)["net_load_mwh"]

    def execute_day(self, price_signal: np.ndarray, intensity: float = 0.0) -> Dict:
        """
        按柔性强度执行一个代表日。

        Args:
            price_signal: 已公布的日前 24 时段价格 (元/MWh)
            intensity: 柔性强度 ∈ [-1, 1]
                符号只对 EV 有意义: > 0 只充电, < 0 在高价时段 V2G 放电 (放电后离站前充回)
                绝对值控制力度: EV 放电量 / 工业转移量 按 |intensity| 线性缩放
                工业部分方向恒为"高价时段 → 窗口内低价时段", 与符号无关
                = 0: 中性日程 (EV 按停车窗口均摊充电恰好达标, 工业不转移)

        Returns:
            dict: net_load_mwh(24,), energy_mwh, penalty, ev / ind 诊断
                  net_load_mwh 与 penalty 都是**单个代表日**的量; 折算到本月由调用方统一乘
                  days_per_month × demand_scale (见 TradingEnvironment.step 的现货分支)。
        """
        price = np.asarray(price_signal, dtype=float)
        assert price.shape == (self.hours,), f"价格信号长度须为 {self.hours}"
        k = 0.0 if not self.include_flexible else float(np.clip(intensity, -1.0, 1.0))

        ev_grid, ev_diag = self._execute_ev(price, k)
        ind_load, ind_diag, penalty = self._execute_industrial(price, k)

        net = (ind_load + ev_grid) / 1000.0
        return {
            "net_load_mwh": net,
            "energy_mwh": float(net.sum()),
            "month_energy_mwh": float(net.sum()) * self.days_per_month,
            "penalty": float(penalty),
            "intensity": k,
            "ev": ev_diag,
            "ind": ind_diag,
        }

    # ------------------------------------------------------------
    # EV
    # ------------------------------------------------------------
    def _discharge_cap_bat(self) -> np.ndarray:
        """
        逐时段电池侧可放电量 (kWh)。

        上限取三者最小: 设备放电功率 / "净负荷不得转负" (以工业刚性负荷为下限,
        即用户侧聚合体最多把自己该时段的用电抵掉, 不对外净送电) / 全局可行性上限。
        """
        guard = np.minimum(self.discharge_limit, self.ind_fixed)
        return guard / max(self.eta_out, 1e-9)

    def _ev_feasibility_limits(self) -> Dict[str, float]:
        """由物理参数直接算出"离站前必须达标"前提下的放电量上限 (不做试错折半)。"""
        cap = self.ev_cap
        task_bat = max(0.0, (self.target_soc - self.soc_start)) * cap        # 电池需净增 (kWh)
        headroom_bat = max(0.0, (self.soc_max - self.target_soc)) * cap       # 达标后仍可容纳
        drainable_bat = max(0.0, (self.soc_start - self.soc_min)) * cap       # 高于 soc_min 的可放
        chg_cap_grid = float(self.charge_limit[self.parked].sum())            # 窗内可注入 (kWh)
        recover_bat = max(0.0, chg_cap_grid * self.eta_in - task_bat)         # 扣掉任务后还能充回
        power_cap_bat = float(self._discharge_cap_bat()[self.parked].sum())
        max_dis_bat = min(drainable_bat, headroom_bat, recover_bat, power_cap_bat)
        return {
            "task_bat_kwh": task_bat,
            "headroom_bat_kwh": headroom_bat,
            "drainable_bat_kwh": drainable_bat,
            "chg_cap_grid_kwh": chg_cap_grid,
            "max_dis_bat_kwh": max_dis_bat,
            "neutral_grid_charge_kwh": task_bat / max(self.eta_in, 1e-9),
        }

    def _execute_ev(self, price: np.ndarray, k: float):
        """返回网侧逐时段电量 (kWh, 正=买入) 与诊断。"""
        if self.ev_n == 0 or not self.parked:
            z = np.zeros(self.hours)
            return z, {"soc_depart": self.soc_start, "unmet_kwh": 0.0,
                       "discharged_kwh": 0.0, "charged_kwh": 0.0,
                       "max_dis_bat_kwh": 0.0, "task_bat_kwh": 0.0}

        lim = self._ev_feasibility_limits()
        task_bat = lim["task_bat_kwh"]
        # 离站达标前提: 充电能量必须塞得进停车窗口的功率上限
        assert lim["neutral_grid_charge_kwh"] <= lim["chg_cap_grid_kwh"] + 1e-9, (
            f"EV 配置不可行: 任务需网侧充电 {lim['neutral_grid_charge_kwh']:.1f} kWh, "
            f"停车窗口仅可注入 {lim['chg_cap_grid_kwh']:.1f} kWh"
        )

        want_bat = 0.0 if k >= 0 else float(np.clip(-k, 0.0, 1.0)) * lim["max_dis_bat_kwh"]
        grid_charge_need = (task_bat + want_bat) / max(self.eta_in, 1e-9)

        cheap = sorted(self.parked, key=lambda h: (price[h], h))
        dear = sorted(self.parked, key=lambda h: (-price[h], h))

        # 电池侧放电: 优先高价时段; 总量 ≤ drainable ⇒ 时序上不会跌破 soc_min
        dis_bat, left_dis = self._fill(want_bat, dear, self._discharge_cap_bat())
        assert left_dis <= 1e-6, f"EV 放电功率上限不足, 未安排 {left_dis:.3f} kWh"

        # 同一时段不同时充/放 (物理上无意义且会虚增往返损耗): 放电时段先排除在充电落点之外
        idle = np.where(dis_bat <= 1e-12, self.charge_limit, 0.0)
        charge, left_chg = self._place_charge(grid_charge_need, price, k, cheap, idle)
        if left_chg > 1e-9:
            spill, left_chg = self._fill(left_chg, cheap, self.charge_limit - charge)
            charge = charge + spill
        assert left_chg <= 1e-6, (
            f"EV 充电功率上限不足, 未安排 {left_chg:.3f} kWh (时段数 {len(self.parked)})"
        )

        res = self._simulate(charge, dis_bat)
        assert res["feasible"], f"EV 执行结果违反物理约束: {res['violations']}"

        net_grid = res["charge_grid"] - res["discharge_grid"]
        return net_grid, {
            "soc_depart": float(res["soc_end"] / max(self.ev_cap, 1e-9)),
            "unmet_kwh": float(max(0.0, self.target_soc * self.ev_cap - res["soc_end"])),
            "discharged_kwh": float(res["discharge_grid"].sum()),
            "charged_kwh": float(res["charge_grid"].sum()),
            "max_dis_bat_kwh": lim["max_dis_bat_kwh"],
            "task_bat_kwh": task_bat,
            "soc_min_violation_kwh": res["violations"].get("below_soc_min", 0.0),
            "soc_max_violation_kwh": res["violations"].get("above_soc_max", 0.0),
        }

    def _place_charge(self, grid_energy: float, price: np.ndarray, k: float,
                      cheap: List[int], limits: np.ndarray):
        """充电日程落点: k>0 填满低价时段; k<=0 按可用比例均摊, 余量补到低价时段。"""
        charge = np.zeros(self.hours)
        left = max(0.0, float(grid_energy))
        if k > 0:
            return self._fill(left, cheap, limits)

        w = self.avail[self.parked]
        tot = float(w.sum())
        if tot <= 0:
            return charge, left
        for wt, h in zip(w, self.parked):
            # 份额按"原始需求量"分摊 (不能用递减后的 left, 否则几何衰减会把余量倒进低价时段)
            charge[h] = min(grid_energy * wt / tot, limits[h])
        remaining_share = grid_energy - float(charge.sum())
        extra, remaining = self._fill(remaining_share, cheap, limits - charge)
        return charge + extra, remaining

    def _simulate(self, charge: np.ndarray, discharge_bat: np.ndarray) -> Dict:
        """
        按停留时序积分 SOC。

        这里**不做任何裁剪或重归一化**: 违反物理边界就返回 feasible=False 并列出越界量,
        由调用方断言, 避免"电量悄悄消失"造成的假守恒。
        """
        e = self.soc_start * self.ev_cap
        ch = np.zeros(self.hours)
        dis_grid = np.zeros(self.hours)
        violations = {}
        for h in self.parked:
            d_bat = float(discharge_bat[h])
            c = float(charge[h])
            e = e - d_bat + c * self.eta_in
            ch[h] = c
            dis_grid[h] = d_bat * self.eta_out
            if e < self.soc_min * self.ev_cap - 1e-6:
                violations.setdefault("below_soc_min", self.soc_min * self.ev_cap - e)
            if e > self.soc_max * self.ev_cap + 1e-6:
                violations.setdefault("above_soc_max", e - self.soc_max * self.ev_cap)
        if e < self.target_soc * self.ev_cap - 1e-6:
            violations.setdefault("depart_soc", self.target_soc * self.ev_cap - e)
        return {
            "feasible": bool(not violations),
            "violations": violations,
            "soc_end": float(e),
            "charge_grid": ch,
            "discharge_grid": dis_grid,
            "charge": ch,
            "discharge": dis_grid,
        }

    # ------------------------------------------------------------
    # 工业
    # ------------------------------------------------------------
    def _execute_industrial(self, price: np.ndarray, k: float):
        """可转移部分重排, 日任务电量守恒; 返回逐时段负荷 (kW→kWh) 与诊断。"""
        if self.max_transferable <= 0 or self.ind_flex.sum() <= 0:
            return self.ind_base.copy(), {"transferred_mwh": 0.0, "unmet_kwh": 0.0}, 0.0

        movable_total = float(self.ind_flex.sum())
        want = min(abs(k), 1.0) * movable_total
        src = np.argsort(-price)
        dst = np.argsort(price)

        reductions = np.zeros(self.hours)
        additions = np.zeros(self.hours)
        moved = 0.0
        for h in src:
            if want <= 1e-12 or moved >= want - 1e-9:
                break
            cut = min(self.ind_flex[int(h)], want - moved)
            if cut <= 1e-9:
                continue
            # 先在候选落点上试装, 装不满整笔就整笔回滚 —— 否则 Σ增载 > Σ减载, 日任务不守恒
            trial = []
            placed = 0.0
            for j in self._window_hours(int(h), dst):
                if placed >= cut - 1e-9:
                    break
                if reductions[j] > 0:
                    continue
                room = self.ind_room[j] - additions[j]
                add = min(cut - placed, room)
                if add > 1e-9:
                    trial.append((j, add))
                    placed += add
            if placed + 1e-9 < cut:
                continue
            for j, add in trial:
                additions[j] += add
            reductions[int(h)] = cut
            moved += cut

        load = self.ind_base - reductions + additions
        penalty = moved / 1000.0 * self.transfer_cost
        unmet = abs(self.ind_task - float(load.sum()))
        return load, {"transferred_mwh": moved / 1000.0, "unmet_kwh": float(unmet)}, penalty

    def _window_hours(self, h: int, dst_order: np.ndarray) -> List[int]:
        """转移窗口内、按价格从低到高排序的落点时段 (窗口约束 × 择优)。"""
        w = self.transfer_window
        window = {(h + d) % self.hours for d in range(1, w + 1)}
        return [int(j) for j in dst_order if int(j) in window]

    # ------------------------------------------------------------
    def _fill(self, target: float, order: List[int], limits: np.ndarray):
        """按时段优先级装填电量, 返回 (逐时段数组, 未能装填的余量)。"""
        out = np.zeros(self.hours)
        left = max(0.0, float(target))
        for h in order:
            add = min(left, max(0.0, float(limits[h])))
            out[h] += add
            left -= add
            if left <= 1e-12:
                break
        return out, float(left)

    def _availability_ratio(self, hour: int) -> float:
        fn = getattr(self.ev, "_get_availability_ratio", None)
        return float(fn(hour)) if fn else (1.0 if self._is_parked(hour) else 0.0)

    def _is_parked(self, hour: int) -> bool:
        fn = getattr(self.ev, "_is_parked", None)
        if fn:
            return bool(fn(hour))
        if self.arrive_hour > self.depart_hour:
            return hour >= self.arrive_hour or hour < self.depart_hour
        return self.arrive_hour <= hour < self.depart_hour

    def _parked_sequence(self) -> List[int]:
        """以到站时刻为起点的停留时序, 保证 SOC 积分覆盖完整停车窗口。"""
        seq = [h for h in range(self.hours) if self.avail[h] > 0]
        if not seq:
            return []
        start = min(seq, key=lambda h: (h - self.arrive_hour) % self.hours)
        return sorted(seq, key=lambda h: (h - start) % self.hours)
