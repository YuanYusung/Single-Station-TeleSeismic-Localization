# -*- coding: utf-8 -*-
"""
Compute P-wave polarization (incidence angle & azimuth)
using eigen-decomposition of 3-component waveform covariance matrix.
"""

import numpy as np
from obspy import read
from obspy.taup import TauPyModel
import matplotlib.pyplot as plt
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees

# ========== 用户输入部分 ==========
stla = 34.94591  # 台站纬度
stlo = -106.4572  # 台站经度

# 目录位置，用作结果对比
evla = 52.498  # 震中纬度
evlo = 160.2637  # 震中经度
evdp = 35  # 震源深度（km）

mseed_file = "kcka_ANMO_ZNE.mseed"  # 你的数据文件
p_pick = 20.0  # P波到时，可以是粗略的，此处仅用于窗口截取
win_len = 10.0  # 截取窗口长度（秒）
# =================================

def polarization_from_traces(z, n, e):
    """特征值分解求偏振方向"""
    Z = np.vstack([z - np.mean(z), n - np.mean(n), e - np.mean(e)])
    M = Z.shape[1]
    C = (Z @ Z.T) / M
    lam, vecs = np.linalg.eigh(C)
    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    vecs = vecs[:, idx]
    v1 = vecs[:, 0]
    vz, vn, ve = v1
    # 入射角：从竖直方向算（0°=垂直入射，90°=水平）
    theta = np.degrees(np.arctan2(np.sqrt(vn**2 + ve**2), vz))
    # 方位角：北为0°，向东正方向
    phi = (np.degrees(np.arctan2(ve, vn)) + 180) % 360 
    if theta > 90:
        theta = 180 - theta
        phi = (phi + 180) % 360
    pol_degree = (lam[0] - lam[1]) / lam[0]
    theta_true = np.rad2deg(np.arcsin(1.732 * np.sin(np.deg2rad(theta / 2.))))
    return theta_true, phi, pol_degree, lam, v1

# ========== 读取与预处理 ==========
st = read(mseed_file)

# 按通道名匹配分量
try:
    trZ = st.select(component="Z")[0]
    trN = st.select(component="N")[0]
    trE = st.select(component="E")[0]
except IndexError:
    raise ValueError("未找到 Z/N/E 三个分量，请检查通道名（例如 BHZ, BHN, BHE）")

fs = trZ.stats.sampling_rate
t = np.arange(0, trZ.stats.npts) / fs

# 截取窗口
start = p_pick
end = p_pick + win_len
i1, i2 = int(start * fs), int(end * fs)
z = trZ.data[i1:i2]
n = trN.data[i1:i2]
e = trE.data[i1:i2]

# 检查长度
if len(z) < 10:
    raise ValueError("选取窗口太短或超出波形长度，请调整 p_pick 或 win_len。")

# ========== 特征分解与计算 ==========
theta, phi, pol_degree, lam, v1 = polarization_from_traces(z, n, e)

# ========== 输出结果 ==========
print("\n===== P波偏振分析结果 =====")
print(f"窗口起止时间: {start:.2f} s ~ {end:.2f} s")
print(f"入射角 θ (°): {theta:.2f}")
print(f"反方位角 φ (°): {phi:.2f}")
print(f"极化度: {pol_degree:.3f}")
print(f"特征值: {lam}")
print(f"主特征向量 v1: {v1}\n")

# ========== 3. 理论计算（TauPyModel） ==========
model = TauPyModel(model="prem")

# 计算台站到震中的球面距离与方位角
dist_m, azimuth_deg, back_azimuth_deg = gps2dist_azimuth(evla, evlo, stla, stlo)
distance_deg = kilometers2degrees(dist_m / 1000.0)

# 计算理论P波入射角（台站处）
arrivals = model.get_ray_paths(source_depth_in_km=evdp, distance_in_degree=distance_deg, phase_list=["P"])
incident_angle_theoretical = arrivals[0].incident_angle

print("\n===== 理论射线路径 (TauPyModel) =====")
print(f"震中距: {distance_deg:.2f}° ({dist_m/1000:.1f} km)")
print(f"反方位角: {back_azimuth_deg:.2f}°")
print(f"理论P波入射角: {incident_angle_theoretical:.2f}°")

# ========== 4. 对比结果 ==========
baz_diff = abs(phi - back_azimuth_deg)
baz_diff = baz_diff if baz_diff <= 180 else 360 - baz_diff
inc_diff = abs(theta - incident_angle_theoretical)

print("\n===== 对比分析 =====")
print(f"观测 vs 理论 反方位角差值: {baz_diff:.2f}°")
print(f"观测 vs 理论 入射角差值: {inc_diff:.2f}°")

# ========== 绘图 ==========
fig, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
ax[0].plot(t, trZ.data, 'k')
ax[1].plot(t, trN.data, 'k')
ax[2].plot(t, trE.data, 'k')
for a in ax:
    a.axvspan(start, end, color='orange', alpha=0.3, label='P波分析窗口')
ax[0].set_ylabel("Z")
ax[1].set_ylabel("N")
ax[2].set_ylabel("E")
ax[2].set_xlabel("Time (s)")
ax[0].legend()
plt.suptitle(f"P波偏振分析\n入射角={theta:.1f}°, 方位角={phi:.1f}°, 极化度={pol_degree:.2f}")
plt.tight_layout()
plt.show()
