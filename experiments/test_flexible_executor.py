"""
柔性负荷执行层验收测试 (审计修复路线 第 3 步 / 任务 #38)

验证目标 (对应审计"最低验收门槛"里的 设备按时完成 + 无虚假退款 + 电量守恒):
    1. EV 离站 SOC 达标: 任意柔性强度下 soc_depart >= target_soc, unmet_kwh == 0
    2. EV 电池能量账目自洽: 充电×η_in - 放电÷η_out == SOC 增量, 且网侧电量 = 充 - 放
    3. 无虚假退款: V2G 放电因往返损耗使**网侧总电量增加**, 不可能"放出去就凭空少买"
    4. 纯时序转移: k>0 只改充电落点, 网侧总电量与中性日程完全相同
    5. 工业日任务守恒: Σ 时段负荷 == 基础日任务电量, 转移惩罚 = 转移电量/1000 × 单价 (可手算)
    6. 功率/SOC 边界: 逐时段充放电不超设备上限
    7. 纯刚性场景 / include_flexible=False: 柔性强度对净负荷无影响
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.load.ev_model import EVLoadModel
from src.load.industrial_load import IndustrialLoadModel
from src.load.load_scenarios import LoadScenarioBuilder
from src.load.flexible_executor import FlexibleExecutor
from src.utils.config import EV_PARAMS, INDUSTRIAL_LOAD_PARAMS, SIM_PARAMS

TOL = 1e-6
HOURS = SIM_PARAMS["hours_per_day"]
INTENSITIES = [-1.0, -0.75, -0.5, -0.2, 0.0, 0.2, 0.5, 0.75, 1.0]


def _prices(rng, scale=400.0):
    """一条有峰谷结构的"已公布日前价"曲线。"""
    base = scale + rng.normal(0, 40, HOURS)
    peak = np.zeros(HOURS)
    peak[9:13] = 120.0
    peak[18:22] = 200.0
    return np.maximum(base + peak, 30.0)


def _make_executor(scenario: str, include_flexible: bool = True) -> FlexibleExecutor:
    b = LoadScenarioBuilder(scenario)
    return FlexibleExecutor(b.ev_model, b.industrial_model, include_flexible=include_flexible)


def _run_over_intensities(ex: FlexibleExecutor, price: np.ndarray):
    return {k: ex.execute_day(price, k) for k in INTENSITIES}


def test_ev_departure_soc_and_battery_bookkeeping():
    """EV: 全强度扫描下离站达标 + 电池账目自洽 + 不越 SOC 边界。"""
    rng = np.random.default_rng(0)
    price = _prices(rng)
    for scenario in LoadScenarioBuilder.list_scenarios():
        ex = _make_executor(scenario)
        for k, res in _run_over_intensities(ex, price).items():
            ev = res["ev"]
            if ex.ev_n == 0:
                assert ev["charged_kwh"] == 0.0 and ev["discharged_kwh"] == 0.0
                continue
            assert ev["soc_depart"] >= ex.target_soc - TOL, (scenario, k, ev)
            assert ev["unmet_kwh"] <= TOL * 1000, (scenario, k, ev)
            assert ev["soc_depart"] <= ex.soc_max + TOL, (scenario, k, ev)
            # 电池侧: 净增 = 充电×η_in - 放电÷η_out
            gained = ev["charged_kwh"] * ex.eta_in - ev["discharged_kwh"] / ex.eta_out
            expect = (ev["soc_depart"] - ex.soc_start) * ex.ev_cap
            assert abs(gained - expect) < 1e-4, (scenario, k, gained, expect)
            # 网侧: 日净电量 = 工业日任务 + EV 充电 - EV 放电
            expect_net = ex.ind_task + ev["charged_kwh"] - ev["discharged_kwh"]
            assert abs(res["net_load_mwh"].sum() * 1000.0 - expect_net) < 1e-4, (scenario, k)
    print(f"  [PASS] {len(LoadScenarioBuilder.list_scenarios())} 个负荷场景 × "
          f"{len(INTENSITIES)} 个强度: EV 离站达标, 电池能量账目自洽")


def test_grid_energy_bookkeeping_matches_net_load():
    """网侧净负荷 = 工业负荷 + EV 充电 - EV 放电 (逐时段, 无隐藏缩放)。"""
    rng = np.random.default_rng(1)
    ex = _make_executor("柔性适中")
    for k in INTENSITIES:
        price = _prices(rng)
        res = ex.execute_day(price, k)
        ind_load, _, _ = ex._execute_industrial(price, k)
        ev_grid, _ = ex._execute_ev(price, k)
        recon = (ind_load + ev_grid) / 1000.0
        assert np.allclose(recon, res["net_load_mwh"], atol=1e-9), k
        assert abs(res["energy_mwh"] - res["net_load_mwh"].sum()) < TOL
        assert abs(res["month_energy_mwh"] - res["energy_mwh"] * ex.days_per_month) < TOL
        assert np.all(res["net_load_mwh"] >= -TOL), (k, res["net_load_mwh"].min())
    print("  [PASS] 网侧净负荷可由工业+EV 逐时段还原, 月度电量 = 日电量 × 天数")


def test_discharge_never_creates_free_energy():
    """无虚假退款: V2G 放电因 η<1 使网侧总电量单调增加 (不可能是净赚电量)。"""
    rng = np.random.default_rng(2)
    ex = _make_executor("柔性适中")
    price = _prices(rng, scale=350.0)
    e_by_strength = {a: ex.execute_day(price, -a)["energy_mwh"]
                     for a in (0.0, 0.2, 0.5, 0.75, 1.0)}
    for a, b in zip(sorted(e_by_strength)[:-1], sorted(e_by_strength)[1:]):
        assert e_by_strength[a] <= e_by_strength[b] + TOL, (a, b, e_by_strength)
    neutral, deep = e_by_strength[0.0], e_by_strength[1.0]
    assert deep > neutral, (deep, neutral)
    loss_expect = ex._ev_feasibility_limits()["max_dis_bat_kwh"] * (
        1.0 / ex.eta_in - ex.eta_out) / 1000.0
    assert abs((deep - neutral) - loss_expect) < 1e-6, (deep - neutral, loss_expect)
    print(f"  [PASS] 满强度 V2G 使网侧电量增加 {deep - neutral:.4f} MWh "
          f"= 往返损耗 {loss_expect:.4f} MWh (无凭空电量)")


def test_positive_intensity_only_reschedules_charging():
    """k>0 只改变充电落点: 网侧 EV 总电量与中性日程完全一致。"""
    rng = np.random.default_rng(3)
    ex = _make_executor("柔性适中")
    price = _prices(rng)
    neutral = ex.execute_day(price, 0.0)
    for k in (0.2, 0.5, 1.0):
        res = ex.execute_day(price, k)
        assert abs(res["ev"]["charged_kwh"] - neutral["ev"]["charged_kwh"]) < 1e-6, k
        assert res["ev"]["discharged_kwh"] == 0.0, k
    # 同一价格曲线上, k=1 的充电落点必须比中性日程更便宜 (择优确实生效)
    cost_smart = float(np.sum(price * np.maximum(ex._execute_ev(price, 1.0)[0], 0.0)))
    cost_neutral = float(np.sum(price * np.maximum(ex._execute_ev(price, 0.0)[0], 0.0)))
    assert cost_smart < cost_neutral - 1.0, (cost_smart, cost_neutral)
    print(f"  [PASS] 正强度下 EV 网侧总电量不变 (纯时序转移), "
          f"充电成本 {cost_smart:,.0f} < 中性 {cost_neutral:,.0f} 元")


def test_industrial_task_conservation_and_penalty():
    """工业: 日任务电量守恒, 惩罚可手算, 负荷不低于刚性下限。"""
    rng = np.random.default_rng(4)
    for scenario in ("柔性适中", "柔性充裕", "柔性受限", "柔性大规模"):
        ex = _make_executor(scenario)
        for k in INTENSITIES:
            price = _prices(rng)
            res = ex.execute_day(price, k)
            ind = res["ind"]
            # 工业逐时负荷单独复算, 保证守恒口径清晰
            ind_load, ind_diag, penalty = ex._execute_industrial(price, k)
            assert abs(float(ind_load.sum()) - ex.ind_task) < 1e-6, (scenario, k)
            assert ind_diag["unmet_kwh"] < 1e-6, (scenario, k, ind_diag)
            assert abs(penalty - ind["transferred_mwh"] * ex.transfer_cost) < 1e-9, (scenario, k)
            assert abs(penalty - res["penalty"]) < 1e-9
            assert np.all(ind_load >= ex.ind_fixed - 1e-9), (scenario, k)
            assert np.all(ind_load <= ex.ind_fixed + ex.max_transferable + 1e-6), (scenario, k)
            assert ind["transferred_mwh"] <= ex.ind_flex.sum() / 1000.0 + TOL
        print(f"    {scenario}: 日任务 {ex.ind_task:.0f} kWh 守恒, "
              f"满强度转移 {ex.execute_day(_prices(rng), 1.0)['ind']['transferred_mwh']:.3f} MWh")
    print("  [PASS] 工业可转移负荷只重排时刻: 日任务电量守恒 + 惩罚 = 转移量×单价")


def test_device_power_limits():
    """逐时段充放电功率不超设备上限 (集群口径: 单车功率 × 车辆数 × 可用比例)。"""
    rng = np.random.default_rng(5)
    ex = _make_executor("柔性充裕")
    for k in INTENSITIES:
        price = _prices(rng)
        ev_grid, ev = ex._execute_ev(price, k)
        charge = np.maximum(ev_grid, 0.0)
        disch = np.maximum(-ev_grid, 0.0)
        # 同一时段既充又放已被 _fill 的容量约束排除, 这里只校验各自上限
        assert np.all(charge <= ex.charge_limit + 1e-6), (k, np.max(charge - ex.charge_limit))
        assert np.all(disch <= ex.discharge_limit + 1e-6), (k, np.max(disch - ex.discharge_limit))
        assert abs(ev["charged_kwh"] - charge.sum()) < 1e-6
        assert abs(ev["discharged_kwh"] - disch.sum()) < 1e-6
    print("  [PASS] EV 逐时段充/放电功率均在设备上限内")


def test_rigid_cases_ignore_intensity():
    """纯刚性场景与 include_flexible=False: 柔性强度不改变净负荷。"""
    rng = np.random.default_rng(6)
    ex = _make_executor("纯刚性")
    base = ex.ind_base
    for k in INTENSITIES:
        price = _prices(rng)
        res = ex.execute_day(price, k)
        ev_grid, _ = ex._execute_ev(price, k)
        assert np.all(ev_grid == 0.0), k
        assert res["penalty"] == 0.0, (k, res["penalty"])
        assert res["ind"]["transferred_mwh"] == 0.0, k
        ind_load, _, _ = ex._execute_industrial(price, k)
        assert np.allclose(ind_load, base, atol=1e-12), k
        assert abs(res["energy_mwh"] - base.sum() / 1000.0) < TOL, k

    ex2 = _make_executor("柔性适中", include_flexible=False)
    price = _prices(rng)
    for k in (-1.0, 0.5, 1.0):
        assert np.allclose(ex2.execute_day(price, k)["net_load_mwh"],
                           ex2.neutral_profile(), atol=1e-12), k
    print("  [PASS] 纯刚性 / 关闭柔性时, 柔性强度对网侧净负荷无影响")


def test_neutral_profile_is_intensity_zero():
    """neutral_profile() 必须等于 execute_day(0 价, k=0), 供合同参考电量 R_m 取形状。"""
    ex = _make_executor("柔性适中")
    neutral = ex.neutral_profile()
    same = ex.execute_day(np.zeros(HOURS), 0.0)["net_load_mwh"]
    assert np.allclose(neutral, same, atol=1e-12)
    assert abs(neutral.sum() - same.sum()) < TOL
    assert np.all(neutral >= 0)
    lim = ex._ev_feasibility_limits()
    assert lim["chg_cap_grid_kwh"] >= lim["neutral_grid_charge_kwh"] > 0
    print(f"  [PASS] 中性日净负荷 {neutral.sum():.3f} MWh, EV 窗内可注入 "
          f"{lim['chg_cap_grid_kwh']:.0f} kWh ≥ 任务需 {lim['neutral_grid_charge_kwh']:.0f} kWh")


def test_executor_works_with_default_params():
    """直接用默认 EV/工业参数构造 (不经场景), 确认可跨模块复用。"""
    ex = FlexibleExecutor(EVLoadModel(EV_PARAMS), IndustrialLoadModel(INDUSTRIAL_LOAD_PARAMS))
    rng = np.random.default_rng(7)
    for k in INTENSITIES:
        res = ex.execute_day(_prices(rng), k)
        assert res["ev"]["soc_depart"] >= ex.target_soc - TOL
        assert res["ind"]["unmet_kwh"] < 1e-6
        assert np.isfinite(res["net_load_mwh"]).all()
    assert ex.ev_n == EV_PARAMS["num_ev"]
    print("  [PASS] 默认参数下 EV 50 辆 / 工业 200 kW 执行闭环可用")


if __name__ == "__main__":
    tests = [
        ("中性剖面与可行性前提", test_neutral_profile_is_intensity_zero),
        ("EV 离站达标/电池账目", test_ev_departure_soc_and_battery_bookkeeping),
        ("网侧电量逐时段还原", test_grid_energy_bookkeeping_matches_net_load),
        ("放电不产生免费电量", test_discharge_never_creates_free_energy),
        ("正强度仅重排充电", test_positive_intensity_only_reschedules_charging),
        ("工业任务守恒/惩罚手算", test_industrial_task_conservation_and_penalty),
        ("设备功率上限", test_device_power_limits),
        ("刚性场景/关闭柔性", test_rigid_cases_ignore_intensity),
        ("默认参数复用", test_executor_works_with_default_params),
    ]
    failed = []
    for name, fn in tests:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as exc:
            failed.append((name, exc))
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
    print(f"\n通过 {len(tests) - len(failed)}/{len(tests)}")
    if failed:
        sys.exit(1)
