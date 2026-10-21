"""
运行源锁 (审计修复路线 第 5 步 / 审计发现 N05)

只写"训练种子 = 42"不足以复现: 同一批种子配上改过的源代码会给出另一组数字,
而结果 JSON 里看不出源代码变过。这里把决定数值的源代码文件哈希、git 提交号、
解释器与库版本一起写进 meta, 让发布出来的每个 JSON 都能追溯到具体代码版本。

复现前先比对 source_lock.files: 哈希不一致 = 不同代码, 发布数字不能拿来对账。
"""
import hashlib
import os
import subprocess
import sys

# 改变这些文件里的任何一行, 结果数字都可能变 —— 全部落进源锁
LOCKED_FILES = (
    "src/utils/config.py",
    "src/market/environment.py",
    "src/market/settlement.py",
    "src/market/market_models.py",
    "src/load/flexible_executor.py",
    "src/load/industrial_load.py",
    "src/load/ev_model.py",
    "src/load/load_scenarios.py",
    "src/agent/sac_agent.py",
    "src/agent/multi_agent.py",
    "src/agent/rule_policies.py",
    "experiments/run_four_arm_benchmark.py",
)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _pkg_version(name: str):
    mod = sys.modules.get(name)
    if mod is None:
        try:
            import importlib
            mod = importlib.import_module(name)
        except Exception:
            return None
    return getattr(mod, "__version__", None)


def _git(root: str, *args):
    try:
        # 显式按 UTF-8 解码: Windows 上 text=True 会用本地 GBK, 遇到中文路径 (如 report/课程设计报告.md)
        # 会在读取线程里抛 UnicodeDecodeError, 于是脏文件清单被静默吞成空 —— 源锁就漏记改动。
        out = subprocess.run(["git", *args], cwd=root, capture_output=True,
                             encoding="utf-8", errors="replace", check=True).stdout.strip()
        return out
    except Exception:
        return None


def _changed_paths(root: str):
    """工作树相对 HEAD 的改动 (含未跟踪), 不解析 porcelain 列宽, 避免格式/换行过滤器差异。

    返回 None 表示 git 问不通 —— 与"问过、确实干净"是两件事, 不能都记成干净。
    """
    out, failed = set(), False
    # -c core.quotePath=false: 否则中文路径被转成 "\350\257\276..." 八进制串, 记录没法读
    for cmd in (("diff", "--name-only", "HEAD"), ("ls-files", "--others", "--exclude-standard")):
        text = _git(root, "-c", "core.quotePath=false", *cmd)
        if text is None:
            failed = True
        else:
            out.update(line for line in text.splitlines() if line)
    return None if failed else sorted(out)


def source_lock(root: str = None) -> dict:
    """当前代码树 + 运行环境的指纹 (键序稳定, 可直接写 JSON)。"""
    if root is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    files = {}
    for rel in LOCKED_FILES:
        path = os.path.join(root, *rel.split("/"))
        files[rel] = _file_sha256(path) if os.path.isfile(path) else "MISSING"
    changed = _changed_paths(root)
    head = _git(root, "rev-parse", "HEAD")
    lock = {
        "git_commit": head,
        "git_dirty": bool(changed) if (head and changed is not None) else None,
        "git_dirty_paths": changed,
        "python": sys.version.split()[0],
        "numpy": _pkg_version("numpy"),
        "torch": _pkg_version("torch"),
        "files": files,
    }
    missing = [k for k, v in files.items() if v == "MISSING"]
    if missing:
        raise RuntimeError(f"源锁清单里的文件不存在 (改名后清单未更新): {missing}")
    return lock
