import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from obspy import UTCDateTime
import matplotlib.pyplot as plt

# -----------------------------
# 初始化地球模型与观测信息
# -----------------------------
model = TauPyModel(model="prem")

sta_lat = 34.94591
sta_lon = -106.4572

P_arrival = UTCDateTime("2025-07-29T23:35:27")
S_arrival = UTCDateTime("2025-07-29T23:44:05")
ref_time = P_arrival

tP_obs = (P_arrival - ref_time)
tS_obs = (S_arrival - ref_time)
inc_obs = 23.8
baz_obs = 314

# -----------------------------
# 正演函数
# -----------------------------
def forward_times(src_lat, src_lon, src_depth_km):
    dist_m, az, baz = gps2dist_azimuth(src_lat, src_lon, sta_lat, sta_lon)
    dist_deg = dist_m / (6371e3 * np.pi / 180.0)
    arrivals_P = model.get_travel_times(src_depth_km, dist_deg, phase_list=["P"])
    arrivals_S = model.get_travel_times(src_depth_km, dist_deg, phase_list=["S"])
    if not arrivals_P or not arrivals_S:
        return np.nan, np.nan, np.nan, np.nan
    tP = arrivals_P[0].time
    tS = arrivals_S[0].time
    inc_P = arrivals_P[0].incident_angle
    baz_mod = baz % 360
    return tP, tS, inc_P, baz_mod

# -----------------------------
# 似然函数（高斯噪声假设）
# -----------------------------
def log_likelihood(params):
    src_lat, src_lon, src_depth_km, origin_time_offset = params
    tP, tS, inc_P, baz_P = forward_times(src_lat, src_lon, src_depth_km)

    resP = tP_obs - (origin_time_offset + tP)
    resS = tS_obs - (origin_time_offset + tS)
    resTheta = inc_obs - inc_P
    d_baz = (baz_obs - baz_P + 180) % 360 - 180
    resPhi = d_baz

    # 观测误差标准差
    sigma_t_P = 0.3
    sigma_t_S = 1.
    sigma_inc = 5.
    sigma_baz = 5.

    logL = (
        -0.5 * (resP / sigma_t_P) ** 2
        -0.5 * (resS / sigma_t_S) ** 2
        -0.5 * (resTheta / sigma_inc) ** 2
        -0.5 * (resPhi / sigma_baz) ** 2
    )
    return logL

# -----------------------------
# 先验分布（均匀）
# -----------------------------
def log_prior(params):
    src_lat, src_lon, src_depth_km, origin_time_offset = params
    if -90 <= src_lat <= 90 and -180 <= src_lon <= 180 and 0.1 <= src_depth_km <= 100 and -1000 <= origin_time_offset <= tP_obs:
        return 0.0  # 均匀先验的log值
    return -np.inf

# -----------------------------
# 后验（log）
# -----------------------------
def log_posterior(params):
    lp = log_prior(params)
    if np.isinf(lp):
        return -np.inf
    return lp + log_likelihood(params)

# -----------------------------
# MCMC 采样 (Metropolis-Hastings)
# -----------------------------
np.random.seed(42)
n_samples = 20000
chain = np.zeros((n_samples, 4))
logp = np.zeros(n_samples)

# 初始点
chain[0] = [45.0, 165.0, 20.0, 0.0]
logp[0] = log_posterior(chain[0])

step_sizes = [0.3, 0.3, 0.3, 0.5]  # 步长调节参数

for i in range(1, n_samples):
    proposal = chain[i - 1] + np.random.normal(0, step_sizes)
    logp_prop = log_posterior(proposal)
    # 接受概率
    if np.log(np.random.rand()) < (logp_prop - logp[i - 1]):
        chain[i] = proposal
        logp[i] = logp_prop
    else:
        chain[i] = chain[i - 1]
        logp[i] = logp[i - 1]

print("MCMC完成 ✅")

# -----------------------------
# 去除前期“燃烧期”（burn-in）
# -----------------------------
burn = int(0.7 * n_samples)
chain_burned = chain[burn:]

# 计算均值与标准差
mean_params = np.mean(chain_burned, axis=0)
std_params = np.std(chain_burned, axis=0)
print("\n=== 贝叶斯反演结果（参数均值 ± 标准差） ===")
print(f"纬度: {mean_params[0]:.3f} ± {std_params[0]:.3f}")
print(f"经度: {mean_params[1]:.3f} ± {std_params[1]:.3f}")
print(f"深度: {mean_params[2]:.3f} ± {std_params[2]:.3f} km")
print(f"发震时刻偏移: {mean_params[3]:.2f} ± {std_params[3]:.2f} s")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.patches import Ellipse

# --- residuals: 以 posterior mean 计算预测到时差并画残差分布 ---
tP_m, tS_m, inc_P_m, baz_P_m = forward_times(mean_params[0], mean_params[1], mean_params[2])
origin_offset_m = mean_params[3]
resP = (tP_obs - (origin_offset_m + tP_m))
resS = (tS_obs - (origin_offset_m + tS_m))
print(f"Residuals at posterior-mean: P_res={resP:.3f} s, S_res={resS:.3f} s, inc_res={inc_obs-inc_P_m:.2f} deg, baz_res={((baz_obs-baz_P_m+180)%360-180):.2f} deg")

# -----------------------------
# 图 A: trace plot (改进)
# -----------------------------
burn = int(0.7 * n_samples)
fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
labels = ["Latitude (deg)", "Longitude (deg)", "Depth (km)", "Origin time offset (s)"]
for i in range(4):
    ax = axs[i]
    ax.plot(chain[:, i], lw=0.5, alpha=0.6)
    ax.axvspan(0, burn, color='gray', alpha=0.2)  # burn shading
    mean_val = mean_params[i]
    std_val = std_params[i]
    ax.hlines(mean_val, burn, n_samples, colors='C1', linestyles='-', label='posterior mean (post-burn)')
    ax.hlines([mean_val - std_val, mean_val + std_val], burn, n_samples, colors='C1', linestyles='--', alpha=0.7, label='±1σ' if i==0 else None)
    ax.set_ylabel(labels[i])
    if i == 0:
        ax.legend(loc='upper right', fontsize=8)
axs[-1].set_xlabel("MCMC step")
plt.suptitle("Trace plots (gray area = burn-in). Posterior mean ±1σ shown on right part.")
plt.tight_layout(rect=[0, 0, 1, 0.97])


# -----------------------------
# 图 B: 经纬度 KDE + scatter + 目录震中
# -----------------------------
catalog_lat = float(52.498)    # 从 USGS 目录读出的纬度
catalog_lon = float(160.2637)
catalog_depth = 35.0

lons = chain_burned[:, 1]
lats = chain_burned[:, 0]
depths = chain_burned[:, 2]

xy = np.vstack([lons, lats])
kde = gaussian_kde(xy)
# grid
lon_min, lon_max = lons.min(), lons.max()
lat_min, lat_max = lats.min(), lats.max()
lon_pad = (lon_max - lon_min) * 0.15 + 1e-6
lat_pad = (lat_max - lat_min) * 0.15 + 1e-6
xi, yi = np.mgrid[lon_min-lon_pad:lon_max+lon_pad:200j, 
                  lat_min-lat_pad:lat_max+lat_pad:200j]
zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

fig, ax = plt.subplots(figsize=(7, 6))

mean_lon = np.mean(lons)
mean_lat = np.mean(lats)

# 协方差矩阵
cov = np.cov(lons, lats)
vals, vecs = np.linalg.eigh(cov)
order = vals.argsort()[::-1]
vals = vals[order]
vecs = vecs[:, order]

# 椭圆方向角
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

# 1-sigma 68% 区域（近似）
scale = np.sqrt(2.30)  # 2D 高斯68%置信区约为 sqrt(2.30) * σ
width, height = 2 * scale * np.sqrt(vals)  # 2*表示直径
ell = Ellipse(xy=(mean_lon, mean_lat), width=width, height=height, angle=theta,
              edgecolor='red', facecolor='none', lw=2, linestyle='--', label='~68% CI')
ax.add_patch(ell)

# 后验样本散点（以深度着色）
sc = ax.scatter(lons, lats, c=depths, s=8, cmap='viridis', alpha=0.6, label='posterior samples')
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Depth (km)')

# 标注后验均值位置
ax.scatter(mean_params[1], mean_params[0], marker='*', s=150, c='red', edgecolor='k', zorder=5, label='posterior mean')

# 标注目录震中位置
ax.scatter(catalog_lon, catalog_lat, marker='^', s=120, c='orange', edgecolor='k', zorder=5, label='catalog location')

# 设置坐标轴与标题
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Posterior samples with catalog location')

ax.legend(loc='best', fontsize=9)
plt.tight_layout()


# -----------------------------
# 残差直方图 + KDE 可视化
# -----------------------------
from scipy.stats import gaussian_kde

# 计算各步残差
tP_chain = np.zeros(chain_burned.shape[0])
tS_chain = np.zeros(chain_burned.shape[0])
inc_chain = np.zeros(chain_burned.shape[0])
baz_chain = np.zeros(chain_burned.shape[0])

for i, params in enumerate(chain_burned):
    tP_i, tS_i, inc_i, baz_i = forward_times(params[0], params[1], params[2])
    origin_offset_i = params[3]
    tP_chain[i] = tP_obs - (origin_offset_i + tP_i)
    tS_chain[i] = tS_obs - (origin_offset_i + tS_i)
    inc_chain[i] = inc_obs - inc_i
    baz_chain[i] = (baz_obs - baz_i + 180) % 360 - 180

# 绘图
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
res_labels = ['P residual (s)', 'S residual (s)', 'Inc residual (°)', 'Baz residual (°)']
res_data = [tP_chain, tS_chain, inc_chain, baz_chain]

for ax, data, label in zip(axs.flat, res_data, res_labels):
    # 直方图
    ax.hist(data, bins=30, density=True, alpha=0.6, color='skyblue', label='Histogram')
    # KDE
    kde = gaussian_kde(data)
    x_grid = np.linspace(min(data), max(data), 200)
    ax.plot(x_grid, kde(x_grid), color='red', lw=1.5, label='KDE')
    ax.axvline(np.mean(data), color='k', linestyle='--', lw=1, label='Mean')
    ax.set_xlabel(label)
    ax.legend(fontsize=8)

plt.suptitle("Residual distributions of posterior samples")
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()