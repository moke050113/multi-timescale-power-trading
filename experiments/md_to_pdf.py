"""
课程设计报告 Markdown -> PDF 转换器 (基于 PyMuPDF Story)
- 中文/CJK: 依赖系统 "Microsoft YaHei" 字体
- 公式: LaTeX -> 可读 Unicode (Story 不支持 MathJax)
- 图片: 相对路径 -> base64 内嵌
- 输出: report/课程设计报告.pdf (A4, 页边距, 页码)
"""
import os
import re
import base64
import pymupdf as fitz
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))          # .../experiments
ROOT = os.path.dirname(HERE)                               # 项目根
REPORT_DIR = os.path.join(ROOT, "report")
MD_PATH = os.path.join(REPORT_DIR, "课程设计报告.md")
PDF_PATH = os.path.join(REPORT_DIR, "课程设计报告.pdf")

# ------------------------------------------------------------
# LaTeX -> Unicode
# ------------------------------------------------------------
GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "Gamma": "Γ", "delta": "δ",
    "Delta": "Δ", "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "pi": "π", "Pi": "Π", "rho": "ρ",
    "sigma": "σ", "Sigma": "Σ", "tau": "τ", "phi": "φ", "Phi": "Φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "Omega": "Ω",
}
OPS = {
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥", r"\approx": "≈",
    r"\cdot": "·", r"\times": "×", r"\pm": "±", r"\in": "∈", r"\to": "→",
    r"\rightarrow": "→", r"\sum": "Σ", r"\langle": "⟨", r"\rangle": "⟩",
    r"\setminus": "\\", r"\neq": "≠", r"\infty": "∞", r"\partial": "∂",
    r"\sim": "∼", r"\dots": "…", r"\ldots": "…",
    r"\quad": "  ", r"\qquad": "   ", r"\,": " ", r"\;": " ", r"\:": " ",
    r"\!": "", r"\ ": " ", r"\%": "%", r"\_": "_",
}
FUNCS = ["dim", "max", "min", "log", "ln", "sin", "cos", "tanh", "exp", "arg"]
# 只表示括号大小的装饰宏: 直接丢弃 (Story 没有 TeX 的定界符尺寸概念)
DELIM_SIZE = re.compile(r"\\(?:biggl|biggr|Biggl|Biggr|bigl|bigr|Bigl|Bigr"
                        r"|bigg|Bigg|left|right|big|Big)(?![a-zA-Z])")
# 函数名转带空格的英文词, 否则 "_{m,h}\max(" 会粘成 "_m,hmax("
FUNC_MACRO = re.compile(r"\\(" + "|".join(FUNCS) + r")(?![a-zA-Z])")


def _match_brace(s, start):
    """s[start]=='{' -> (inner, index_after_closing_brace)"""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1:i], i + 1
    return s[start + 1:], len(s)


def _expand_frac(s):
    while True:
        idx = s.find("\\frac")
        if idx == -1:
            break
        j = idx + 5
        while j < len(s) and s[j] == " ":
            j += 1
        if j < len(s) and s[j] == "{":
            num, e1 = _match_brace(s, j)
            k = e1
            while k < len(s) and s[k] == " ":
                k += 1
            if k < len(s) and s[k] == "{":
                den, e2 = _match_brace(s, k)
                s = s[:idx] + "(" + num + ")/(" + den + ")" + s[e2:]
                continue
        s = s[:idx] + "frac" + s[idx + 5:]
    return s


def _expand_underbrace(s):
    while True:
        idx = s.find("\\underbrace")
        if idx == -1:
            break
        j = idx + len("\\underbrace")
        if j < len(s) and s[j] == "{":
            inner, e1 = _match_brace(s, j)
            tail = s[e1:]
            m = re.match(r"\s*_\{", tail)
            if m:
                bstart = e1 + m.end() - 1
                _lab, e2 = _match_brace(s, bstart)
                s = s[:idx] + inner + s[e2:]
            else:
                s = s[:idx] + inner + s[e1:]
            continue
        s = s[:idx] + "underbrace" + s[idx + len("\\underbrace"):]
    return s


def latex_to_unicode(s):
    s = s.strip()
    # \text{X} / \mathrm{X} / \mathbf{X} -> X
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|mathcal|operatorname)\{([^}]*)\}", r"\1", s)
    # 嵌套结构: \frac / \underbrace (平衡括号, 支持嵌套)
    s = _expand_frac(s)
    s = _expand_underbrace(s)
    s = _expand_frac(s)
    # \bar{X} / \bar X -> X̄ (combining macron)
    s = re.sub(r"\\bar\s*\{?([A-Za-z])\}?", lambda m: m.group(1) + "̄", s)
    s = re.sub(r"\\hat\{?([A-Za-z])\}?", lambda m: m.group(1) + "̂", s)
    # \left \right \Big 等定界符尺寸宏: 丢弃
    s = DELIM_SIZE.sub("", s)
    # 函数名 (需在 greek 兜底前处理, 否则退化成裸单词且无空格)
    s = FUNC_MACRO.sub(lambda m: " " + m.group(1) + " ", s)
    # escaped braces
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    # operators (before greek so \leq etc. caught)
    for k, v in sorted(OPS.items(), key=lambda x: -len(x[0])):
        s = s.replace(k, v)
    # greek
    def gk(m):
        return GREEK.get(m.group(1), m.group(1))
    s = re.sub(r"\\([A-Za-z]+)", gk, s)
    # subscripts: _{...} -> _inner ; _x -> _x
    s = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
    # superscripts: ^{...} -> unicode if simple, else ^inner
    # 上标: Story 用的雅黑字体缺 Unicode 上标数字块 (⁻⁶ 会渲染成空白), 只有 Latin-1 的 ¹²³ 可靠
    lat1 = {"1": "¹", "2": "²", "3": "³"}

    def sup(m):
        inner = m.group(1)
        return lat1.get(inner, "^(" + inner + ")") if re.fullmatch(r"[-+0-9mhn]+", inner) \
            else "^(" + inner + ")"
    s = re.sub(r"\^\{([^{}]*)\}", sup, s)
    s = re.sub(r"\^(\d+)", lambda m: lat1.get(m.group(1), "^(" + m.group(1) + ")"), s)
    # cleanup spacing
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _html_escape(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def process_math(text):
    # 数学式转成 Unicode 后必须转义尖括号: "<10⁻⁶>" 这类串会被 markdown 当成 HTML 标签整段吞掉
    def disp(m):
        return "\n\n> " + _html_escape(latex_to_unicode(m.group(1))) + "\n\n"
    text = re.sub(r"\$\$(.+?)\$\$", disp, text, flags=re.S)
    # inline $...$ (avoid $$ leftovers)
    text = re.sub(r"\$([^$\n]+?)\$",
                  lambda m: _html_escape(latex_to_unicode(m.group(1))), text)
    # 正文里裸露的 "<" 同样会被当标签起点 (如 "残差 < 1e-6"); 保留 <br> 这类真标签
    text = re.sub(r"<(?![a-zA-Z/])", "&lt;", text)
    return text


# ------------------------------------------------------------
# 构建 HTML
# ------------------------------------------------------------
def img_to_base64(html):
    def repl(m):
        src = m.group(1)
        if src.startswith("data:"):
            return m.group(0)
        path = src.replace("\\", "/")
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(REPORT_DIR, path))
        if not os.path.exists(path):
            return m.group(0)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
        return m.group(0).replace(m.group(1), f"data:image/{ext};base64,{b64}")
    return re.sub(r'<img[^>]*src="([^"]+)"', repl, html)


CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
       font-size: 10.5pt; line-height: 1.7; color: #1a1a1a; }
h1 { font-size: 20pt; color: #1B4F72; text-align: center;
     border-bottom: 2px solid #2E86DE; padding-bottom: 8px; margin-top: 0; }
h2 { font-size: 15pt; color: #1B4F72; border-left: 5px solid #2E86DE;
     padding-left: 8px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 12.5pt; color: #21618C; margin-top: 16px; page-break-after: avoid; }
p { margin: 6px 0; text-align: justify; }
table { border-collapse: collapse; margin: 10px auto; font-size: 9.5pt; }
th, td { border: 1px solid #999; padding: 4px 9px; }
th { background: #EAF2F8; }
tr:nth-child(even) td { background: #FBFCFD; }
blockquote { border-left: 3px solid #F39C12; background: #FEF9E7;
     margin: 8px 0; padding: 6px 12px; color: #6E5B00; }
code { font-family: "Consolas", "Microsoft YaHei", monospace;
     background: #F4F6F7; padding: 1px 4px; font-size: 9.5pt; }
pre { font-family: "Consolas", "Microsoft YaHei", monospace; background: #F4F6F7;
     border: 1px solid #D5DBDB; padding: 8px; font-size: 9pt; white-space: pre-wrap;
     line-height: 1.4; }
img { max-width: 100%; display: block; margin: 10px auto; }
hr { border: none; border-top: 1px solid #ccc; margin: 16px 0; }
ul, ol { margin: 6px 0; }
li { margin: 3px 0; }
strong { color: #154360; }
"""


def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md = f.read()
    md = process_math(md)
    body = markdown.markdown(
        md, extensions=["tables", "fenced_code", "sane_lists", "nl2br"]
    )
    body = img_to_base64(body)
    html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"

    # 调试: 保存 HTML
    with open(os.path.join(REPORT_DIR, "_report_debug.html"), "w", encoding="utf-8") as f:
        f.write(html)

    story = fitz.Story(html=html)
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (18 * 2.83465, 20 * 2.83465, -18 * 2.83465, -20 * 2.83465)

    tmp = os.path.join(REPORT_DIR, "_report_raw.pdf")
    with fitz.DocumentWriter(tmp) as writer:
        more = 1
        while more > 0:
            dev = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()

    # 页码
    doc = fitz.open(tmp)
    for i, page in enumerate(doc):
        r = page.rect
        page.insert_text(
            fitz.Point(r.width / 2 - 12, r.height - 12),
            f"{i + 1}", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4),
        )
    doc.save(PDF_PATH, garbage=4, deflate=True)
    doc.close()
    try:
        os.remove(tmp)
    except OSError:
        pass

    n = fitz.open(PDF_PATH)
    print(f"生成: {PDF_PATH}")
    print(f"页数: {n.page_count}, 大小: {os.path.getsize(PDF_PATH)/1024:.0f} KB")
    n.close()


if __name__ == "__main__":
    main()
