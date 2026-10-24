"""
文昌发射场 · 当日发射窗口 + 上升段闭合链 + GMAT 状态（单文件）
===============================================================

统一约定（写死，不再有歧义）：
  · 猪排图 time  = 探测器穿出地球影响球(SOI)的时刻 t_SOI（不是发射时刻！）
  · 猪排图 vinf1 = SOI 处的地心速率 v_SOI（不是无穷远剩余速度）
    真 V∞ = sqrt(v_SOI² − 2μ/r_SOI)（能量反推）

链路（全部倒推，固定文昌）：
  (1) 猪排图：指定到达 或 自动扫该列 |V∞| 最小 → v_SOI 与方向 s
  (2) v_SOI → 真 V∞ → 逃逸双曲线 (e, θ1, ν∞, 停泊周期 T)
  (3) SOI → 近地点(TMI)：双曲开普勒 → t_TMI
  (4) 当日窗口（顺行 + 射向走廊）→ 窗内最优 t_opt（i* = max(|δ∞|,φ)）
  (5) 定时闭合解 tmi_hit：滑行<1圈，出SOI精确=departure
  (6) ★上升段闭合链：给定上升时长/航程，联立解
        共面  det[r_s(t_lift), r_ins, s] = 0        → 解地面射向 A0
        计时  t_ins + coast(入轨→近地点) = t_ign(A1) → 解离台时刻
      【教训1】离台 ≠ t_opt − 上升时长：downrange 相位使滑行缩短约
      3.8 min、地球在上升段里转过 ~4.5°——必须解联立，不能做算术减法。
      【教训2】计时根必落 在 dsec ∈ (−T·60, 0)（0<滑行<T 的物理约束），
      外层扫描必须用这个区间——扫描窗开错位置会拿到"假闭合解"。
  (7) GMAT 直填状态：A1(理想·发射钟) / A1c(★全链闭合，仅闭合时打印) /
      A2(网格钟·对接dihuo2) / B(定时闭合·准点出SOI)

两个时钟：
  发射钟 A1/A1c：离台→入轨→滑行→点火→出SOI（出SOI偏离departure属正常，
                 由 dihuo2.py 差分修正吸收；或直接把实际出SOI时刻写进
                 departure 重新生成格子）
  网格钟 A2/B  ：t_TMI = departure − Δt(rp→SOI)，出SOI ≡ departure

用法：
  python dihuo1.py 2020-07-25                     # 自动扫该列|V∞|最小到达
  python dihuo1.py 2020-07-25 2021-02-11          # 指定到达
  python dihuo1.py 2020-07-25 2021-02-11 --alt 200 --ascent 18 --downrange 1600
  python dihuo1.py "2020-07-21 17:59:19" 2021-01-30    # 出发可带时刻(交棒钟)
  python dihuo1.py --golden 2020-07-15 2020-08-03 --max-coast 30
       # 扫出发期：只保留 滑行≤max-coast、在主窗口、射向在走廊 的命中解，
       # 按 gap 排序；选定 D* 后 python dihuo1.py D* 查全链闭合报告

只依赖 numpy。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 常数
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418
R_EARTH = 6378.137
R_SOI = 9.24e5
SIDEREAL_DEG_PER_DAY = 360.98564736629
J2000 = datetime(2000, 1, 1, 12, 0, 0)
AU = 1.495978707e8
MU_SUN = 1.32712440018e11
OBLIQUITY = math.radians(23.4392911)

WENCHANG = (19.6145, 110.9518)   # 固定文昌
AZ_LO, AZ_HI = 90.0, 175.0       # 文昌东南安全走廊

SCAN_PRE_H = 30.0                # 窗口扫描范围（相对 t_TMI，小时）
SCAN_POST_H = 42.0

_ASC_T_DEF = 18.0                # 默认上升段时长（min），以实际运载时序为准
_ASC_D_DEF = 1600.0              # 默认 downrange（km）

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def unit(v):
    return v / np.linalg.norm(v)


def wrap360(x: float) -> float:
    return float(x) % 360.0


def fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%d %H:%M:%S")


def fmt_bj(t: datetime) -> str:
    return (t + timedelta(hours=8)).strftime("%m-%d %H:%M:%S")


def fmt_gmat(t: datetime) -> str:
    """GMAT UTCGregorian：19 Jul 2020 00:30:18.634"""
    return "%02d %s %04d %02d:%02d:%06.3f" % (
        t.day, _MONTHS[t.month - 1], t.year,
        t.hour, t.minute, t.second + t.microsecond / 1e6)


def parse_dt(s: str) -> datetime:
    """宽容解析：YYYY-MM-DD 或 'YYYY-MM-DD HH:MM[:SS]'"""
    s = s.strip()
    for f in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    raise ValueError("时间格式：YYYY-MM-DD 或 'YYYY-MM-DD HH:MM[:SS]'")


# ---------------------------------------------------------------------------
# 时间 / GMST
# ---------------------------------------------------------------------------
def julian_date(t: datetime) -> float:
    return (t - J2000).total_seconds() / 86400.0 + 2451545.0


def gmst_deg(t: datetime) -> float:
    d = julian_date(t) - 2451545.0
    T = d / 36525.0
    g = (280.46061837 + 360.98564736629 * d
         + 0.000387933 * T * T - T ** 3 / 38710000.0)
    return g % 360.0


# ---------------------------------------------------------------------------
# (2) v_SOI → 真 V∞ → 双曲线
# ---------------------------------------------------------------------------
@dataclass
class Hyperbola:
    vinf_vec: np.ndarray
    vinf: float
    ra_deg: float
    dec_deg: float
    rp: float
    e: float
    p: float
    a: float
    period_min: float
    theta1_deg: float
    vp: float
    vc: float
    dv1: float
    nu_inf_deg: float
    t_soi_h: float          # 近地点 → SOI 的飞行时间（小时）

    @property
    def s(self) -> np.ndarray:
        return self.vinf_vec / self.vinf


def true_vinf_from_vsoi(v_soi: float) -> float:
    """SOI 处速率 → 无穷远剩余速度（能量守恒）"""
    v2 = v_soi * v_soi - 2.0 * MU_EARTH / R_SOI
    if v2 <= 0:
        raise ValueError("v_SOI 太小，未到逃逸能量")
    return math.sqrt(v2)


def build_hyperbola(vinf_vec, rp: float) -> Hyperbola:
    v = np.asarray(vinf_vec, dtype=float)
    vm = float(np.linalg.norm(v))
    if vm <= 0:
        raise ValueError("|V∞| 必须为正")
    s = v / vm
    mu = MU_EARTH
    e = 1.0 + rp * vm ** 2 / mu
    p = rp * (1.0 + e)
    a = -mu / vm ** 2
    vp = math.sqrt(mu * p) / rp
    vc = math.sqrt(mu / rp)
    theta1 = math.degrees(math.acos(1.0 / e))
    nu_inf = math.acos(-1.0 / e)
    ch = (R_SOI / abs(a) + 1.0) / e
    Hh = math.acosh(max(1.0, ch))
    Mh = e * math.sinh(Hh) - Hh
    t_soi = Mh * math.sqrt(abs(a) ** 3 / mu)
    return Hyperbola(
        vinf_vec=v, vinf=vm,
        ra_deg=math.degrees(math.atan2(s[1], s[0])) % 360.0,
        dec_deg=math.degrees(math.asin(max(-1.0, min(1.0, float(s[2]))))),
        rp=rp, e=e, p=p, a=a,
        period_min=2.0 * math.pi * math.sqrt(rp ** 3 / mu) / 60.0,
        theta1_deg=theta1, vp=vp, vc=vc, dv1=vp - vc,
        nu_inf_deg=math.degrees(nu_inf),
        t_soi_h=t_soi / 3600.0)


# ---------------------------------------------------------------------------
# (5) 给定发射时刻 → 平面 → 倾角/方位角/滑行
# ---------------------------------------------------------------------------
@dataclass
class Geometry:
    t_launch: datetime
    branch: int
    inc_deg: float
    coast_min: float
    az_deg: float


def site_state(t: datetime, lat_deg: float, lon_deg: float):
    al = math.radians(wrap360(gmst_deg(t) + lon_deg))
    ph = math.radians(lat_deg)
    r = np.array([math.cos(ph) * math.cos(al),
                  math.cos(ph) * math.sin(al),
                  math.sin(ph)])
    z = np.array([0.0, 0.0, 1.0])
    east = unit(np.cross(z, r))
    north = unit(np.cross(r, east))
    return r, north, east


def geometry_at(hyp: Hyperbola, t: datetime, lat_deg: float,
                lon_deg: float, branch: int) -> Optional[Geometry]:
    r_s, north, east = site_state(t, lat_deg, lon_deg)
    s = hyp.s
    n = np.cross(r_s, s)
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None
    n = branch * n / nn
    i = math.acos(max(-1.0, min(1.0, float(n[2]))))
    raan = math.atan2(n[0], -n[1])
    P0 = np.array([math.cos(raan), math.sin(raan), 0.0])
    Q0 = np.cross(n, P0)
    nu_inf = math.radians(hyp.nu_inf_deg)
    p_hat = unit(math.cos(nu_inf) * s - math.sin(nu_inf) * np.cross(n, s))
    q_hat = np.cross(n, p_hat)
    ph_site = math.atan2(float(r_s @ q_hat), float(r_s @ p_hat))
    du = wrap360(math.degrees(-ph_site))
    coast = du / 360.0 * hyp.period_min
    vh = unit(np.cross(n, r_s))
    az = wrap360(math.degrees(math.atan2(float(vh @ east),
                                         float(vh @ north))))
    return Geometry(t_launch=t, branch=branch, inc_deg=math.degrees(i),
                    coast_min=coast, az_deg=az)


def prograde_at(hyp, t, lat, lon) -> Optional[Geometry]:
    """t 时刻的顺行几何（i≤90°）"""
    for br in (+1, -1):
        g = geometry_at(hyp, t, lat, lon, br)
        if g is not None and g.inc_deg <= 90.0:
            return g
    return None


# ---------------------------------------------------------------------------
# (8) TMI 状态：由轨道面构造（ICRF, km, km/s）
# ---------------------------------------------------------------------------
def state_from_plane(hyp: Hyperbola, n: np.ndarray):
    """由轨道面 n̂（含渐近线 s）构造 TMI 近地点点火状态"""
    n = unit(n)
    if n[2] < 0:                       # 顺行支
        n = -n
    nu = math.radians(hyp.nu_inf_deg)
    p_hat = unit(math.cos(nu) * hyp.s - math.sin(nu) * np.cross(n, hyp.s))
    q_hat = np.cross(n, p_hat)
    return hyp.rp * p_hat, hyp.vp * q_hat


def tmi_state(hyp: Hyperbola, t_launch: datetime, lat: float, lon: float):
    """
    选定发射时刻 → 轨道面 → TMI(近地点)点火状态。
    理想模型：入轨点=测站头顶（上升段闭合见 ascent_chain/方案A1c）。
    六个分量由轨道面唯一决定，与"几点经过近地点"无关。
    """
    g = prograde_at(hyp, t_launch, lat, lon)
    if g is None:
        raise ValueError("该时刻无顺行发射几何")
    r_s, _, _ = site_state(t_launch, lat, lon)
    rv, vv = state_from_plane(hyp, np.cross(r_s, hyp.s))
    return g, rv, vv


# ---------------------------------------------------------------------------
# (6) 当日窗口 + 窗内最优时刻
# ---------------------------------------------------------------------------
@dataclass
class Window:
    t_open: datetime
    t_close: datetime
    t_opt: datetime
    i_opt: float
    az_opt: float
    coast_opt: float

    @property
    def length_min(self) -> float:
        return (self.t_close - self.t_open).total_seconds() / 60.0


def daily_window(hyp: Hyperbola, t_tmi: datetime, lat: float, lon: float,
                 dep: datetime, az_lo: float = AZ_LO, az_hi: float = AZ_HI,
                 step_min: float = 1.0,
                 pre_h: float = SCAN_PRE_H,
                 post_h: float = SCAN_POST_H) -> List[Window]:
    """
    锚定扫描：[t_TMI-pre_h, t_TMI+post_h]；贴边截断段丢弃；
    按"实际 SOI 穿越 ≈ departure"排序，full[0] 为主窗口。
    """
    t0 = t_tmi - timedelta(hours=pre_h)
    t1 = t_tmi + timedelta(hours=post_h)
    n = int((t1 - t0).total_seconds() // 60 // step_min)

    samples = []
    for k in range(n + 1):
        t = t0 + timedelta(minutes=k * step_min)
        g = prograde_at(hyp, t, lat, lon)
        ok = g is not None and az_lo <= g.az_deg <= az_hi
        samples.append((t, ok, g))

    raw: List[Window] = []
    k = 0
    while k < len(samples):
        if not samples[k][1]:
            k += 1
            continue
        j = k
        while j + 1 < len(samples) and samples[j + 1][1]:
            j += 1
        seg = [samples[m][2] for m in range(k, j + 1) if samples[m][2]]
        if seg:
            gb = min(seg, key=lambda x: x.inc_deg)
            raw.append(Window(samples[k][0], samples[j][0], gb.t_launch,
                              gb.inc_deg, gb.az_deg, gb.coast_min))
        k = j + 1

    refined: List[Window] = []
    for w in raw:
        lo, hi = w.t_open, w.t_close

        def i_at(t, _lat=lat, _lon=lon):
            g = prograde_at(hyp, t, _lat, _lon)
            return 1e9 if g is None else g.inc_deg

        for _ in range(60):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            if i_at(m1) < i_at(m2):
                hi = m2
            else:
                lo = m1
        t_opt = lo + (hi - lo) / 2
        g = prograde_at(hyp, t_opt, lat, lon)
        if g:
            refined.append(Window(w.t_open, w.t_close, g.t_launch,
                                  g.inc_deg, g.az_deg, g.coast_min))

    m = timedelta(minutes=2 * step_min)
    full = [w for w in refined if w.t_open > t0 + m and w.t_close < t1 - m]
    if not full:
        full = refined

    def soi_gap(w: Window) -> float:
        t_soi = (w.t_opt + timedelta(minutes=w.coast_opt)
                 + timedelta(hours=hyp.t_soi_h))
        return abs((t_soi - dep).total_seconds())

    full.sort(key=soi_gap)
    return full


def tmi_hit(hyp, t_tmi, lat, lon, max_revs=0, n_scan=1440):
    """
    定时闭合解：解 t_L + coast(t_L) + k·T = t_TMI（顺行）。
    max_revs=0 ⇒ 滑行不足一圈。命中解出 SOI 精确 = departure。
    """
    T = hyp.period_min
    roots = []
    for k in range(max_revs + 1):
        lo = t_tmi - timedelta(minutes=(k + 1) * T)
        hi = t_tmi - timedelta(minutes=k * T)

        def resid(t):
            g = prograde_at(hyp, t, lat, lon)
            if g is None:
                return None
            return (t - t_tmi).total_seconds() / 60.0 + g.coast_min + k * T

        prev = None
        for idx in range(n_scan + 1):
            t = lo + (hi - lo) * (idx / n_scan)
            f = resid(t)
            if f is None:
                prev = None
                continue
            if prev and prev[1] * f <= 0 and abs(f - prev[1]) < T * 0.5:
                a, b, fa = prev[0], t, prev[1]
                for _ in range(70):
                    mm = a + (b - a) / 2
                    fm = resid(mm)
                    if fm is None:
                        break
                    if fa * fm <= 0:
                        b = mm
                    else:
                        a, fa = mm, fm
                ts = a + (b - a) / 2
                g = prograde_at(hyp, ts, lat, lon)
                if g:
                    roots.append(g)
                prev = None
                continue
            prev = (t, f)
    return roots


# ---------------------------------------------------------------------------
# ★ (6b) 上升段闭合链：离台 → 入轨 → 近地点（联立共面+计时，不做算术减法）
# ---------------------------------------------------------------------------
def _rot_gmst(t: datetime) -> np.ndarray:
    g = math.radians(gmst_deg(t))
    c, s_ = math.cos(g), math.sin(g)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _site_unit(lat_deg: float, lon_deg: float) -> np.ndarray:
    ph, l = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(ph) * math.cos(l),
                     math.cos(ph) * math.sin(l), math.sin(ph)])


def _bigcircle_dest(lat0, lon0, A0, D, r_orb):
    """从(lat0,lon0)沿方位A0走大圆D(km)，返回到达点(纬,经)rad"""
    d = D / r_orb
    A, ph0, l0 = math.radians(A0), math.radians(lat0), math.radians(lon0)
    ph2 = math.asin(math.sin(ph0) * math.cos(d)
                    + math.cos(ph0) * math.sin(d) * math.cos(A))
    l2 = l0 + math.atan2(math.sin(A) * math.sin(d) * math.cos(ph0),
                         math.cos(d) - math.sin(ph0) * math.sin(ph2))
    return ph2, l2


def ascent_chain(hyp: Hyperbola, t_anchor: datetime, t_a: float, D: float,
                 lat0: float, lon0: float) -> Optional[dict]:
    """
    自洽上升段解（固定上升时长 t_a min、航程 D km）。
    未知 (dsec=离台偏移 s, A0=地面射向°)；残差：
      res1 = det[r_s(t_lift), r_ins, s]              共面（km³）→ 解 A0
      res2 = t_ins + coast(入轨→近地点) − t_anchor   计时（min）→ 解 dsec
    t_anchor = 方案A1 的近地点历元（发射钟锚点）。

    【关键】外层扫描区间取 dsec ∈ (−T·60, 0]：
      0 < coast < T（滑行不足一圈）⇒ dsec = −coast·60 必落此区间。
      扫描窗开错位置会撞到边缘拿"假闭合解"（教训：曾输出残差+21 min
      却标 ✓ 的方案A1c）。
    返回 dict(t_lift,t_ins,A0,lat_ins,lon_ins,n,i_deg,du,coast,res1,res2,
              closed) 或 None。
    """
    s = hyp.s
    T = hyp.period_min
    nu = math.radians(hyp.nu_inf_deg)
    r_orb = hyp.rp

    def chain(dsec: float, A0: float):
        t_lift = t_anchor - timedelta(minutes=t_a) + timedelta(seconds=dsec)
        t_ins = t_lift + timedelta(minutes=t_a)
        r_s_in = R_EARTH * (_rot_gmst(t_lift) @ _site_unit(lat0, lon0))
        ph2, l2 = _bigcircle_dest(lat0, lon0, A0, D, r_orb)
        r_ins = _rot_gmst(t_ins) @ (r_orb * _site_unit(math.degrees(ph2),
                                                       math.degrees(l2)))
        cx = np.cross(r_s_in, s)
        if float(np.linalg.norm(cx)) < 1e-9:
            return None
        det = float(np.dot(cx, r_ins))
        n = cx / np.linalg.norm(cx)
        if n[2] < 0:
            n = -n
        p_hat = unit(math.cos(nu) * s - math.sin(nu) * np.cross(n, s))
        q_hat = np.cross(n, p_hat)
        ph_ins = math.atan2(float(r_ins @ q_hat), float(r_ins @ p_hat))
        du = (-math.degrees(ph_ins)) % 360.0
        coast = du / 360.0 * T
        res2 = (t_ins + timedelta(minutes=coast)
                - t_anchor).total_seconds() / 60.0
        return dict(t_lift=t_lift, t_ins=t_ins, A0=A0,
                    lat_ins=math.degrees(ph2),
                    lon_ins=((math.degrees(l2) + 180) % 360) - 180,
                    n=n, i_deg=math.degrees(math.acos(n[2])),
                    du=du, coast=coast, res1=det, res2=res2)

    def inner(dsec):
        """固定 dsec：A0 ∈ [60,170] 扫描找变号区间，二分解 res1=0"""
        prev, br = None, None
        for A0 in range(60, 172, 4):
            c = chain(dsec, float(A0))
            if c is None:
                prev = None
                continue
            if prev is not None and prev[1] * c["res1"] <= 0:
                br = (prev[0], float(A0))
                break
            prev = (float(A0), c["res1"])
        if br is None:
            return None
        lo, hi = br
        flo = chain(dsec, lo)["res1"]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            c = chain(dsec, mid)
            if c is None:
                break
            fm = c["res1"]
            if flo * fm <= 0:
                hi = mid
            else:
                lo, flo = mid, fm
        return chain(dsec, 0.5 * (lo + hi))

    # ---- 外层（修正版）：dsec ∈ (−T·60, 0]，0<coast<T 的物理区间 ----
    prev, br, best = None, None, None
    for dsec in range(-int(T * 60) - 60, 1, 90):
        c = inner(float(dsec))
        if c is None:
            prev = None
            continue
        if best is None or abs(c["res2"]) < abs(best["res2"]):
            best = c
        if prev is not None and prev[1] * c["res2"] <= 0:
            br = (prev[0], float(dsec))
            break
        prev = (float(dsec), c["res2"])
    if br is None:
        return dict(best, closed=False) if best else None
    lo, hi = br
    c_lo = inner(lo)
    if c_lo is None:
        return dict(best, closed=False) if best else None
    flo = c_lo["res2"]
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        cm = inner(mid)
        if cm is None:
            lo = mid
            continue
        if flo * cm["res2"] <= 0:
            hi = mid
        else:
            lo, flo = mid, cm["res2"]
    c = inner(0.5 * (lo + hi))
    return dict(c, closed=True) if c else dict(best, closed=False)


# ---------------------------------------------------------------------------
# (1) 猪排图：Standish 历表 + 通用变量 Lambert
# ---------------------------------------------------------------------------
_ELEM = {
    "earth": ((1.00000261, 0.01671123, -0.00001531,
               100.46457166, 102.93768193, 0.0),
              (0.00000562, -0.00004392, -0.01294668,
               35999.37244981, 0.32327364, 0.0)),
    "mars": ((1.52371034, 0.09339410, 1.84969142,
              -4.55343205, -23.94362959, 49.55953891),
             (0.00001847, 0.00007882, -0.00813131,
              19140.30268499, 0.44441088, -0.29257343)),
}


def _kepler(M, e):
    E = M
    for _ in range(80):
        E = E - (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    return E


def helio_state(body, t):
    (a, e, I, L, wbar, Om), (da, de, dI, dL, dw, dO) = _ELEM[body]
    T = (julian_date(t) - 2451545.0) / 36525.0
    a += da * T; e += de * T; I += dI * T
    L += dL * T; wbar += dw * T; Om += dO * T
    a *= AU
    M = math.radians((L - wbar) % 360.0)
    E = _kepler(M, e)
    n = math.sqrt(MU_SUN / a ** 3)
    Ed = n / (1 - e * math.cos(E))
    xv = a * (math.cos(E) - e); yv = a * math.sqrt(1 - e * e) * math.sin(E)
    vxv = -a * math.sin(E) * Ed
    vyv = a * math.sqrt(1 - e * e) * math.cos(E) * Ed
    w = math.radians(wbar - Om)
    cw, sw = math.cos(w), math.sin(w)
    cO, sO = math.cos(math.radians(Om)), math.sin(math.radians(Om))
    cI, sI = math.cos(math.radians(I)), math.sin(math.radians(I))
    P = np.array([cw * cO - sw * sO * cI, cw * sO + sw * cO * cI, sw * sI])
    Q = np.array([-sw * cO - cw * sO * cI, -sw * sO + cw * cO * cI, cw * sI])
    return xv * P + yv * Q, vxv * P + vyv * Q


def _stumpff(z):
    if z > 1e-8:
        s = math.sqrt(z)
        return (1 - math.cos(s)) / z, (s - math.sin(s)) / s ** 3
    if z < -1e-8:
        s = math.sqrt(-z)
        return (1 - math.cosh(s)) / z, (math.sinh(s) - s) / s ** 3
    return 1 / 2 - z / 24 + z * z / 720, 1 / 6 - z / 120 + z * z / 5040


def lambert(r1, r2, tof_s, mu=MU_SUN):
    R1, R2 = np.linalg.norm(r1), np.linalg.norm(r2)
    cd = float(r1 @ r2) / (R1 * R2)
    c = np.cross(r1, r2)
    dnu = math.acos(max(-1.0, min(1.0, cd)))
    if c[2] < 0:
        dnu = 2 * math.pi - dnu
    A = math.sin(dnu) * math.sqrt(R1 * R2 / (1 - cd))
    lo, hi = -4 * math.pi ** 2 + 1e-8, 4 * math.pi ** 2
    for _ in range(220):
        z = 0.5 * (lo + hi)
        C, S = _stumpff(z)
        y = R1 + R2 + A * (z * S - 1) / math.sqrt(C)
        if y < 0:
            lo = z
            continue
        x = math.sqrt(y / C)
        tt = (x ** 3 * S + A * math.sqrt(y)) / math.sqrt(mu)
        if tt < tof_s:
            lo = z
        else:
            hi = z
    z = 0.5 * (lo + hi)
    C, S = _stumpff(z)
    y = R1 + R2 + A * (z * S - 1) / math.sqrt(C)
    f = 1 - y / R1
    g = A * math.sqrt(y / mu)
    gd = 1 - y / R2
    return (np.array(r2) - f * np.array(r1)) / g, \
           (gd * np.array(r2) - np.array(r1)) / g


def vinf_at(departure, arrival):
    """给定出发/到达，返回地心赤道系 V∞ 矢量（日心→赤道）"""
    r1, v1e = helio_state("earth", departure)
    r2, v2m = helio_state("mars", arrival)
    v1, v2 = lambert(r1, r2, (arrival - departure).total_seconds())
    vinf_ecl = v1 - v1e
    ce, se = math.cos(OBLIQUITY), math.sin(OBLIQUITY)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    return Rx @ vinf_ecl, float(np.linalg.norm(v2 - v2m))


def scan_column(departure, day_lo=120, day_hi=260):
    """扫该 departure 列里 |V∞| 最小的一格（能量最优到达）"""
    best = None
    for k in range(day_lo, day_hi + 1):
        arr = departure + timedelta(days=k)
        try:
            v, vmars = vinf_at(departure, arr)
        except Exception:
            continue
        m = float(np.linalg.norm(v))
        if best is None or m < best[1]:
            best = (arr, m, v, vmars)
    if best is None:
        raise ValueError("该出发日列扫描无有效解（历表/Lambert 均失败）")
    return best


# ---------------------------------------------------------------------------
# (9) 黄金日扫描：筛"滑行短"的出发日期
# ---------------------------------------------------------------------------
def golden_scan(dep_lo: datetime, dep_hi: datetime, alt: float = 200.0,
                max_coast: Optional[float] = None, top: int = 8):
    """
    扫出发日期区间。判据（k=0 命中解）：
      ① 落进主窗口  ② 射向在走廊内  ③ 滑行 ≤ max_coast（如指定，如 30/20 min）
    按 gap = |t_hit − t_opt| 排序（gap 小 ⇒ 倾角贴 i*）。
    选定 D* 后务必 python dihuo1.py D* 查全链闭合报告（勿混其他格参数）。
    """
    out = []
    d = dep_lo.replace(hour=0, minute=0, second=0, microsecond=0)
    dhi = dep_hi.replace(hour=0, minute=0, second=0, microsecond=0)
    while d <= dhi:
        try:
            r = compute(d, alt=alt, ascent=0.0)   # 扫描不做上升段闭合（提速）
        except Exception:
            d += timedelta(days=1)
            continue
        if r["wins"] and r["hits"]:
            w = r["wins"][0]
            for h in r["hits"]:
                if (w.t_open <= h.t_launch <= w.t_close
                        and AZ_LO <= h.az_deg <= AZ_HI
                        and (max_coast is None or h.coast_min <= max_coast)):
                    gap = abs((h.t_launch - w.t_opt).total_seconds()) / 60.0
                    out.append((gap, d, r["arr"], h, w))
        d += timedelta(days=1)

    if not out:
        print("扫描期内无满足判据的格子（窗口缺失 / 不入走廊 / "
              "滑行 > %s min）。" % ("限值" if max_coast else "1圈"))
        return out

    out.sort(key=lambda x: x[0])
    print("=" * 70)
    print("黄金日扫描 · %s ~ %s · 停泊 %d km · 滑行限值 %s"
          % (fmt(dep_lo)[:10], fmt(dep_hi)[:10], alt,
             ("≤%g min" % max_coast) if max_coast else "不设"))
    print("判据：命中解在主窗口内、射向走廊内、滑行短；gap 越小倾角越贴 i*")
    print("=" * 70)
    for gap, d, arr, h, w in out[:top]:
        print("%s  gap=%6.1f min  i=%.3f° A=%.2f° 滑行 %5.1f min | "
              "自动到达 %s | 窗内最优 i*=%.3f°"
              % (fmt(d)[:10], gap, h.inc_deg, h.az_deg, h.coast_min,
                 fmt(arr)[:10], w.i_opt))
    gap, d, arr, h, w = out[0]
    print("-" * 70)
    print("首选出发日 D* = %s（滑行 %.1f min，i=%.3f°）"
          % (fmt(d)[:10], h.coast_min, h.inc_deg))
    print("下一步：python dihuo1.py %s   ← 全链闭合报告 + GMAT 状态"
          % fmt(d)[:10])
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def compute(departure: datetime, alt: float = 200.0,
            arrival: Optional[datetime] = None,
            ascent: float = _ASC_T_DEF,
            downrange: float = _ASC_D_DEF):
    if alt <= 0:
        raise ValueError("停泊高度必须为正")
    lat, lon = WENCHANG
    rp = R_EARTH + alt

    if arrival is not None:
        vec, vmars = vinf_at(departure, arrival)
        v_soi = float(np.linalg.norm(vec))
        auto = False
    else:
        arrival, v_soi, vec, vmars = scan_column(departure)
        auto = True

    vinf = true_vinf_from_vsoi(v_soi)
    s = vec / np.linalg.norm(vec)
    hyp = build_hyperbola(s * vinf, rp)
    t_tmi = departure - timedelta(seconds=hyp.t_soi_h * 3600.0)

    wins = daily_window(hyp, t_tmi, lat, lon, dep=departure)
    hits = tmi_hit(hyp, t_tmi, lat, lon, max_revs=0)

    dl_tmi = ((wrap360(gmst_deg(t_tmi) + lon) - hyp.ra_deg + 180) % 360) - 180

    # ---- GMAT 初始状态 ----
    gmat_states = []   # (label, t_launch, epoch, geom, rv, vv)
    asc = None
    if wins:
        w = wins[0]
        gA, rvA, vvA = tmi_state(hyp, w.t_opt, lat, lon)
        t_ign = w.t_opt + timedelta(minutes=w.coast_opt)
        gmat_states.append((
            "方案A1：★最优发射·严格弹道点火（发射 %s UTC + 滑行 %.2f min，"
            "i=%.4f°；理想化：入轨点=测站头顶）"
            % (fmt(w.t_opt), w.coast_opt, w.i_opt),
            w.t_opt, t_ign, gA, rvA, vvA))

        # ★ 上升段闭合链（锚定同一近地点历元 t_ign）
        if ascent > 0:
            try:
                asc = ascent_chain(hyp, t_ign, ascent, downrange, lat, lon)
            except Exception:
                asc = None
            if asc:
                asc["t_a"], asc["D"], asc["t_anchor"] = ascent, downrange, t_ign
                naive = w.t_opt - timedelta(minutes=ascent)
                asc["naive"] = naive
                asc["delta_lift_min"] = ((asc["t_lift"] - naive)
                                         .total_seconds() / 60.0)
                # 守卫：只有真闭合才允许进 GMAT 块
                if asc.get("closed"):
                    rvC, vvC = state_from_plane(hyp, asc["n"])
                    gmat_states.append((
                        "方案A1c：★全链闭合·上升段平面（北京离台 %s + 上升 "
                        "%.0f min + 滑行 %.2f min → 同一历元）"
                        % (fmt_bj(asc["t_lift"]), asc["t_a"], asc["coast"]),
                        asc["t_lift"], t_ign, None, rvC, vvC))

        gmat_states.append((
            "方案A2：A1同一状态·网格钟历元（出SOI=departure，对接日心段用）",
            w.t_opt, t_tmi, gA, rvA, vvA))
    for k, gh in enumerate(hits):
        _, rvB, vvB = tmi_state(hyp, gh.t_launch, lat, lon)
        gmat_states.append((
            "方案B%d：定时闭合·发射 %s UTC 滑行 %.2f min（i=%.4f°，"
            "出SOI=departure）"
            % (k + 1, fmt(gh.t_launch), gh.coast_min, gh.inc_deg),
            gh.t_launch, t_tmi, gh, rvB, vvB))

    return dict(departure=departure, arr=arrival, v_soi=v_soi, vmars=vmars,
                vinf=vinf, hyp=hyp, t_tmi=t_tmi, wins=wins, hits=hits,
                dl_tmi=dl_tmi, alt=alt, auto=auto, asc=asc,
                gmat_states=gmat_states)


def report(r) -> str:
    hyp = r["hyp"]
    lat, lon = WENCHANG
    asc = r["asc"]
    L = []
    L.append("=" * 70)
    L.append("文昌发射场 · 当日发射窗口 + 上升段闭合链 + GMAT 状态")
    L.append("=" * 70)
    L.append("[猪排图]")
    L.append("  departure(SOI)  %s UTC" % fmt(r["departure"]))
    L.append("  arrival(%s) %s UTC" % (
        "自动·该列|V∞|最小" if r["auto"] else "指定", fmt(r["arr"])))
    L.append("  v_SOI   %.5f km/s （猪排图给的 vinf1，按 SOI 处速率）" % r["v_soi"])
    L.append("  → 真 V∞ %.5f km/s （能量反推，差 %.5f）"
             % (r["vinf"], r["v_soi"] - r["vinf"]))
    L.append("  |v∞(火星)| %.4f km/s" % r["vmars"])
    L.append("")
    L.append("[双曲线]")
    L.append("  e=%.5f  θ1=%.4f°  ν∞=%.4f°  T=%.4f min  ΔV1=%.4f km/s"
             % (hyp.e, hyp.theta1_deg, hyp.nu_inf_deg,
                hyp.period_min, hyp.dv1))
    L.append("  α∞=%.4f°  δ∞=%.4f°" % (hyp.ra_deg, hyp.dec_deg))
    L.append("")
    L.append("[SOI → 近地点(TMI)]")
    L.append("  Δt(SOI→rp) = %.4f h = %.3f d" % (hyp.t_soi_h, hyp.t_soi_h / 24))
    L.append("  t_TMI = %s UTC（网格钟）" % fmt(r["t_tmi"]))
    L.append("")
    L.append("[当日窗口 · 文昌 %.4f°N %.4f°E 停泊 %d km]"
             % (lat, lon, r["alt"]))
    if not r["wins"]:
        L.append("  扫描范围内无顺行安全窗口")
    for idx, w in enumerate(r["wins"]):
        tag = "主窗口（本出发日）" if idx == 0 else "相邻恒星日窗口"
        L.append("  ── %s ──" % tag)
        L.append("  开窗 %s UTC（北京 %s）"
                 % (fmt(w.t_open), fmt_bj(w.t_open)))
        L.append("  关窗 %s UTC（北京 %s）  窗长 %.2f h"
                 % (fmt(w.t_close), fmt_bj(w.t_close), w.length_min / 60))
        t_soi_act = w.t_opt + timedelta(minutes=w.coast_opt) \
            + timedelta(hours=hyp.t_soi_h)
        L.append("  ★ 当日最优时刻 %s UTC（北京 %s）"
                 % (fmt(w.t_opt), fmt_bj(w.t_opt)))
        if abs(hyp.dec_deg) >= lat:
            note = "理论最小 i_min = max(|δ∞|,φ)（δ∞>φ 时同时 az 最小）"
        else:
            note = "（窗内最小；δ∞<φ 时全局最小点不在走廊内）"
        L.append("    i* = %.4f°  %s" % (w.i_opt, note))
        L.append("    A* = %.3f°  滑行 %.2f min" % (w.az_opt, w.coast_opt))
        L.append("    → 实际SOI穿越 %s UTC（北京 %s），距猪排图 departure %+.2f d"
                 % (fmt(t_soi_act), fmt_bj(t_soi_act),
                    (t_soi_act - r["departure"]).total_seconds() / 86400.0))
    L.append("")
    L.append("[定时闭合解 tmi_hit（顺行，滑行<1圈，出SOI=departure）]")
    if not r["hits"]:
        L.append("  无")
        L.append("  [诊断] Δλ(t_TMI) = %+.2f°。k=0 切片要求 Δλ 落入射向走廊"
                 % r["dl_tmi"])
        L.append("  对应区间内；本日相位不匹配属正常。")
    for g in r["hits"]:
        inw = any(w.t_open <= g.t_launch <= w.t_close for w in r["wins"])
        gap_min = min(abs((g.t_launch - w.t_opt).total_seconds()) / 60.0
                      for w in r["wins"]) if r["wins"] else float("nan")
        L.append("  %s UTC i=%.4f° A=%.2f° 滑行 %.2f min  %s  距t_opt %.1f min"
                 % (fmt(g.t_launch), g.inc_deg, g.az_deg, g.coast_min,
                    "[在窗口内]" if inw else "[窗口外]", gap_min))
    if asc:
        L.append("")
        L.append("[上升段闭合链 ★ 离台→入轨→近地点]"
                 "（上升 %.0f min / 航程 %.0f km）" % (asc["t_a"], asc["D"]))
        L.append("  联立求解：共面 det[r_s,r_ins,s]=0（解射向）+ "
                 "计时 t_ins+coast=t_ign（解离台）")
        L.append("  离台   %s UTC（北京 %s）"
                 % (fmt(asc["t_lift"]), fmt_bj(asc["t_lift"])))
        L.append("  射向   A0 = %.3f°（地面航向%s）"
                 % (asc["A0"],
                    "，走廊内 ✓" if AZ_LO <= asc["A0"] <= AZ_HI else "！"))
        L.append("  入轨   %g km @ %.2f°N %.2f°E（海上空）"
                 % (r["alt"], asc["lat_ins"], asc["lon_ins"]))
        L.append("  平面   i = %.4f°（渐近线 s 精确保留）" % asc["i_deg"])
        L.append("  滑行   入轨→近地点 du=%.2f° → %.2f min（<1圈 ✓）"
                 % (asc["du"], asc["coast"]))
        L.append("  残差   共面 %.2e km³ / 计时 %+.2e min"
                 % (abs(asc["res1"]), asc["res2"]))
        # 守卫：✓ 只在真闭合时打印
        if asc.get("closed"):
            L.append("  链条   离台 →(%.0f min)→ 入轨 →(%.2f min)→ 近地点 %s ✓"
                     % (asc["t_a"], asc["coast"], fmt(asc["t_anchor"])))
        else:
            t_pg = asc["t_ins"] + timedelta(minutes=asc["coast"])
            L.append("  链条   实际到近地点 %s，距锚 %s 差 %+.1f min ✗ 未闭合"
                     % (fmt(t_pg), fmt(asc["t_anchor"]), asc["res2"]))
            L.append("         （A1c 状态已按守卫从 GMAT 块剔除，勿手工挪用）")
        L.append("  对比   天真回拨(t_opt−t_a)=%s → 自洽解晚 %+.1f min"
                 "（downrange 相位修正，不能拍脑袋减）"
                 % (fmt(asc["naive"])[11:], asc["delta_lift_min"]))
    if r["gmat_states"]:
        L.append("")
        L.append("=" * 70)
        L.append("[GMAT 初始状态 · TMI 点火（近地点，点火后，停泊 %d km）]"
                 % r["alt"])
        L.append("  CoordinateSystem: EarthICRF    StateType: Cartesian")
        for label, t_l, ep, g, rv, vv in r["gmat_states"]:
            L.append("  ── %s ──" % label)
            L.append("    Epoch: %s  (UTCGregorian)" % fmt_gmat(ep))
            L.append("    X  = %14.4f km     VX = %12.6f km/s"
                     % (rv[0], vv[0]))
            L.append("    Y  = %14.4f km     VY = %12.6f km/s"
                     % (rv[1], vv[1]))
            L.append("    Z  = %14.4f km     VZ = %12.6f km/s"
                     % (rv[2], vv[2]))
            L.append("    校验: |r|=%.1f km  |v|=%.4f km/s  r·v=%+.3e"
                     % (float(np.linalg.norm(rv)),
                        float(np.linalg.norm(vv)),
                        float(rv @ vv)))
            t_soi = ep + timedelta(hours=hyp.t_soi_h)
            L.append("    → 出SOI %s UTC（距 departure %+.3f d）"
                     % (fmt(t_soi),
                        (t_soi - r["departure"]).total_seconds() / 86400.0))
        L.append("  注：① 点火后状态，GMAT 中不要再叠加 TMI 脉冲；")
        L.append("     ② 初始 200 km 但能量为正，请关闭大气阻力；")
        L.append("     ③ GMST 直写 J2000 省略岁差（~0.3°），由 dihuo2.py")
        L.append("        差分修正吸收；")
        L.append("     ④ A1/A1c=发射钟（出SOI偏离departure属正常，dihuo2")
        L.append("        初值建议 departure 带实际出SOI时刻，或直接以该时刻")
        L.append("        作 departure 重新生成本报告）；A2/B=网格钟")
        L.append("        （出SOI=departure）；")
        L.append("     ⑤ A1c 为全链闭合状态（离台→入轨→滑行→点火逐项验算，")
        L.append("        未闭合时自动不打印），与 A1 差 ~0.05° 面转角")
        L.append("        （km / m·s⁻¹ 级），两者均可作 dihuo2 修正初值；")
        L.append("        上升段参数以实际运载时序为准（--ascent/--downrange）；")
        L.append("     ⑥ 定时闭合/准点出SOI 场景请用方案B的状态。")
    L.append("=" * 70)
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="文昌当日发射窗口 + 上升段闭合链 + GMAT 状态"
                    "（猪排图 time=SOI, vinf1=SOI速率）")
    ap.add_argument("departure", nargs="?", default=None,
                    help="猪排图出发日期(SOI)，如 2020-07-25 或带时刻")
    ap.add_argument("arrival", nargs="?", default=None,
                    help="可选：到达日期；省略则自动扫该列 |V∞| 最小")
    ap.add_argument("--alt", type=float, default=200.0, help="停泊高度 km")
    ap.add_argument("--ascent", type=float, default=_ASC_T_DEF,
                    help="上升段时长 min（0=跳过闭合链；以实际运载时序为准）")
    ap.add_argument("--downrange", type=float, default=_ASC_D_DEF,
                    help="上升段航程 km")
    ap.add_argument("--golden", nargs=2, metavar=("DEP_LO", "DEP_HI"),
                    help="扫出发日期区间找'滑行短'的日期，"
                         "如 --golden 2020-07-15 2020-08-03")
    ap.add_argument("--max-coast", type=float, default=None,
                    help="golden：只保留滑行≤该分钟数的命中解（如 30 / 20）")
    a = ap.parse_args(argv)

    if a.golden:
        try:
            d0, d1 = parse_dt(a.golden[0]), parse_dt(a.golden[1])
        except ValueError as e:
            sys.exit(str(e))
        if d1 < d0:
            sys.exit("区间起止颠倒")
        golden_scan(d0, d1, alt=a.alt, max_coast=a.max_coast)
        return 0

    if not a.departure:
        ap.error("需要 departure，或使用 --golden 扫描出发期")
    try:
        dep = parse_dt(a.departure)
    except ValueError as e:
        sys.exit(str(e))
    arr = None
    if a.arrival:
        try:
            arr = parse_dt(a.arrival)
        except ValueError as e:
            sys.exit(str(e))
        if arr <= dep:
            sys.exit("到达日期必须晚于出发日期")

    try:
        r = compute(dep, alt=a.alt, arrival=arr,
                    ascent=a.ascent, downrange=a.downrange)
    except ValueError as e:
        sys.exit("错误：%s" % e)
    print(report(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
