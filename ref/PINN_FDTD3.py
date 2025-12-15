##########################################################################
#  PINN用FDTDシミュレーションコード
#  cupy版

import numpy as np
import cupy as cp
import sourse_new
import os

from sampling_utils import SpatioTemporalSampler


x_length = 0.02  # x方向の長さ m
y_length = 0.04  # y方向の長さ m
mesh_length = 1.0e-5  # m
nx = int(x_length / mesh_length)  # how many mesh
ny = int(y_length / mesh_length)

dx = x_length / nx  # mesh length m
dy = y_length / ny  # m

rho = 7840  # density kg/m^3
E = 206 * 1e9  # young percentage kg/ms^2
G = 80 * 1e9  # stiffness
V = 0.27  # poisson ratio

cl = np.sqrt(E / rho * (1 - V) / (1 + V) / (1 - 2 * V))  # P wave
ct = np.sqrt(G / rho)  # S wave
c11 = E * (1 - V) / (1 + V) / (1 - 2 * V)
c13 = E * V / (1 + V) / (1 - 2 * V)
c55 = (c11 - c13) / 2

dt = dx / cl / np.sqrt(6)  # time mesh
f = 4.7e6  # frequency
T = 1 / f  # period
lam = cl / f  # lambda
k = 1 / lam  # wave number
n = T / dt  # 波が離散点上で何点か

# サンプリング設定
nx_sample = 20  # x方向サンプル数
ny_sample = 40  # y方向サンプル数
nt_sample = 100  # 時刻サンプル数

# サンプラー初期化
sampler = SpatioTemporalSampler(
    x_range=(0, x_length),
    y_range=(0, y_length),
    t_range=(3.5e-6, 10e-6),
    seed=42,  # 固定シード
)

# サンプリング座標を生成
x_samples, y_samples, t_samples = sampler.sample_time_slices(nx=nx_sample, ny=ny_sample, nt=nt_sample)

# FDTDインデックスに変換
i_x_all = []
i_y_all = []
for i in range(nt_sample):
    i_x, i_y = sampler.to_fdtd_indices(x_samples[i], y_samples[i], dx, dy)
    i_x_all.append(i_x)
    i_y_all.append(i_y)

i_x_all = np.array(i_x_all)
i_y_all = np.array(i_y_all)

# サンプリング点のデータを記録
T1_sampled = np.zeros((nt_sample, nx_sample * ny_sample))
T3_sampled = np.zeros((nt_sample, nx_sample * ny_sample))
Ux_sampled = np.zeros((nt_sample, nx_sample * ny_sample))
Uy_sampled = np.zeros((nt_sample, nx_sample * ny_sample))


def isfree(nx, ny, f_width, f_pitch, f_depth, mesh_length):
    T13_isfree = np.ones((nx + 1, ny))
    T5_isfree = np.ones((nx + 1, ny + 1))
    mn_p = int(f_pitch / mesh_length)  # 1ピッチの離散点数
    mn_nf = int((f_pitch - f_width) / mesh_length)  # きずのない部分の離散点数
    mn_d = int(f_depth / mesh_length)  # きずの深さ方向の離散点数
    num_f = int(ny * mesh_length / f_pitch)  # きずの数
    T13_isfree[0, 0:ny] = 0
    T13_isfree[nx, 0:ny] = 0
    T5_isfree[0:nx, 0] = 0
    T5_isfree[0:nx, ny] = 0
    T5_isfree[nx, 0 : ny + 1] = 0

    for i in range(num_f):
        if (i + 1) * mn_p >= ny:
            break
        #  きず部分ゆえに消すとこ
        T5_isfree[nx - mn_d : nx, mn_nf + i * mn_p : (i + 1) * mn_p] = 0
        T13_isfree[nx - mn_d : nx + 1, mn_nf + i * mn_p : (i + 1) * mn_p - 1] = 0
    return T13_isfree, T5_isfree


def around_free():
    Ux_free_count = np.zeros((nx, ny), dtype=float)
    Uy_free_count = np.zeros((nx + 1, ny + 1), dtype=float)

    for i in range(nx):
        for j in range(ny):
            if T13_isfree[i, j] == 0:
                Ux_free_count[i, j] += 1
            if T13_isfree[i + 1, j] == 0:
                Ux_free_count[i, j] += 1
            if T5_isfree[i, j] == 0:
                Ux_free_count[i, j] += 1
            if T5_isfree[i, j + 1] == 0:
                Ux_free_count[i, j] += 1

    for i in range(nx + 1):
        for j in range(ny + 1):
            if i == 0 or (i == nx and j == 0) or (i == nx and j == ny):
                Uy_free_count[i, j] += 1
            if j == 0 or j == ny:
                Uy_free_count[i, j] += 1
            elif 0 < i and 0 < j:
                if T13_isfree[i, j - 1] == 0:
                    Uy_free_count[i, j] += 1
                if T13_isfree[i, j] == 0:
                    Uy_free_count[i, j] += 1
                if T5_isfree[i - 1, j] == 0:
                    Uy_free_count[i, j] += 1
                if T5_isfree[i, j] == 0:
                    Uy_free_count[i, j] += 1
    return Ux_free_count, Uy_free_count


# =========================================
# 入射波の設定
wn = 2.5  # 波数
wave4 = np.zeros(int(wn * n), dtype=float)
for ms in range(len(wave4)):
    wave2 = (1 - np.cos(2 * np.pi * f * dt * ms / wn)) / 2
    wave3 = np.sin(2 * np.pi * f * dt * ms)
    wave4[ms] = wave2 * wave3
# ==========================================

# 音源の位置
sy = int(ny / 2)
sx = 0
# 探触子の直径 m
probe_d = 0.007
sy_l = sy - int(probe_d / mesh_length / 2)
sy_r = sy + int(probe_d / mesh_length / 2)

t_max = 10e-6 / dt

#  =====================================================================================
# 複数パラメータの設定
#  =====================================================================================
# 傷の幅 m (固定)
f_width = 0.25e-3  # m

# シミュレーションするpitchとdepthのリスト
pitch_list = [1.25e-3, 1.50e-3, 1.75e-3, 2.00e-3]  # 例: 1.25mm, 1.50mm, 1.75mm
depth_list = [0.10e-3, 0.20e-3, 0.30e-3]  # 例: 0.10mm, 0.20mm, 0.30mm

# 複数パラメータでシミュレーション実行
for f_pitch in pitch_list:
    for f_depth in depth_list:
        print(f"\n{'=' * 60}")
        print(f"Starting simulation: pitch={f_pitch * 1e3:.2f}mm, depth={f_depth * 1e3:.2f}mm")
        print(f"{'=' * 60}\n")

        # サンプリングデータをリセット
        T1_sampled = np.zeros((nt_sample, nx_sample * ny_sample))
        T3_sampled = np.zeros((nt_sample, nx_sample * ny_sample))
        Ux_sampled = np.zeros((nt_sample, nx_sample * ny_sample))
        Uy_sampled = np.zeros((nt_sample, nx_sample * ny_sample))
        wave = np.zeros(int(t_max))
        print(f"f_depth = {f_depth}")
        #  傷の数
        num_f = int(y_length / f_pitch)
        #  傷の幅の離散点数
        mn_w = int(f_width / mesh_length)
        #  1ピッチの離散点数
        mn_p = int(f_pitch / mesh_length)
        # 傷のない部分の離散点数
        mn_nf = int((f_pitch - f_width) / mesh_length)
        # 傷の深さ方向の離散点数
        mn_d = int(f_depth / mesh_length)
        #  =========================================================================================
        #  =======FDTD本体=======
        # 　T1,T3は垂直応力 T5はせん断応力　Ux,Uyは固体粒子速度
        T1 = cp.zeros((nx + 1, ny), dtype=float)
        T3 = cp.zeros((nx + 1, ny), dtype=float)
        T5 = cp.zeros((nx, ny + 1), dtype=float)
        Ux = cp.zeros((nx, ny), dtype=float)
        Uy = cp.zeros((nx + 1, ny + 1), dtype=float)

        dtx = dt / dx
        dty = dt / dy

        T13_isfree, T5_isfree = isfree(nx, ny, f_width, f_pitch, f_depth, mesh_length)
        Ux_free_count, Uy_free_count = around_free()
        time_idx = 0

        for t in range(int(t_max)):
            print(t / t_max)
            #  入射波
            if t < int(len(wave4)):
                T1[0, sy_l:sy_r] = wave4[t]

            if t >= int(len(wave4)):
                Uy[0, sy_l:sy_r] = 0
                Ux[0, sy_l:sy_r] = 0
                T1[0, 0:ny] = 0
                T5[0, 0:ny] = 0

            #  応力の境界条件
            T5[0:nx, 0] = 0
            T5[0:nx, ny] = 0
            T3[0, 0:ny] = 0
            T1[nx, 0:ny] = 0
            T3[nx, 0:ny] = 0
            T1[0, 0] = 0
            T3[0, 0] = 0
            T5[0, 0] = 0
            #  横粒子速度の境界条件
            Uy[1:nx, 0] = Uy[1:nx, 0] - (4 / rho) * dtx * (T3[1:nx, 0])
            Uy[1:nx, ny] = Uy[1:nx, ny] - (4 / rho) * dtx * (-1 * T3[1:nx, ny - 1])
            Uy[nx, 1:ny] = Uy[nx, 1:ny] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:ny])
            #  きず部分の応力境界条件
            for i in range(num_f):
                if (i + 1) * mn_p >= ny:
                    break
                #  きず部分ゆえに消すとこ
                T5[nx - mn_d : nx, mn_nf + i * mn_p : (i + 1) * mn_p] = 0
                T1[nx - mn_d : nx + 1, mn_nf + i * mn_p : (i + 1) * mn_p - 1] = 0
                T3[nx - mn_d : nx + 1, mn_nf + i * mn_p : (i + 1) * mn_p - 1] = 0

            #  横粒子速度の境界条件
            Uy[1:nx, 0] = Uy[1:nx, 0] - (4 / rho) * dtx * (T3[1:nx, 0])
            Uy[1:nx, ny] = Uy[1:nx, ny] - (4 / rho) * dtx * (-1 * T3[1:nx, ny - 1])
            Uy[nx, 1:ny] = Uy[nx, 1:ny] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:ny])

            #  応力の更新
            T1[1:nx, 0:ny] = T1[1:nx, 0:ny] - dtx * (
                (c11 * (Ux[1:nx, 0:ny] - Ux[0 : nx - 1, 0:ny])) + (c13 * (Uy[1:nx, 1 : ny + 1] - Uy[1:nx, 0:ny]))
            )

            T3[1:nx, 0:ny] = T3[1:nx, 0:ny] - dtx * (
                (c13 * (Ux[1:nx, 0:ny] - Ux[0 : nx - 1, 0:ny])) + (c11 * (Uy[1:nx, 1 : ny + 1] - Uy[1:nx, 0:ny]))
            )

            T5[0:nx, 1:ny] = T5[0:nx, 1:ny] - dtx * c55 * (
                Ux[0:nx, 1:ny] - Ux[0:nx, 0 : ny - 1] + Uy[1 : nx + 1, 1:ny] - Uy[0:nx, 1:ny]
            )
            #  きず部分の応力境界条件
            for i in range(num_f):
                if (i + 1) * mn_p >= ny:
                    break
                #  きず部分ゆえに消すとこ
                T5[nx - mn_d : nx, mn_nf + i * mn_p : (i + 1) * mn_p] = 0
                T1[nx - mn_d : nx + 1, mn_nf + i * mn_p : (i + 1) * mn_p - 1] = 0
                T3[nx - mn_d : nx + 1, mn_nf + i * mn_p : (i + 1) * mn_p - 1] = 0

            #  横粒子速度の境界条件
            Uy[1:nx, 0] = Uy[1:nx, 0] - (4 / rho) * dtx * (T3[1:nx, 0])
            Uy[1:nx, ny] = Uy[1:nx, ny] - (4 / rho) * dtx * (-1 * T3[1:nx, ny - 1])
            Uy[nx, 1:ny] = Uy[nx, 1:ny] - (4 / rho) * dtx * (-1 * T5[nx - 1, 1:ny])
            #  粒子速度の更新
            # 自由境界に接するノードと接しないノードを組み合わせて粒子速度を更新
            Ux[0:nx, 0:ny] = cp.where(
                cp.asarray(Ux_free_count[0:nx, 0:ny]) < 4,
                Ux[0:nx, 0:ny]
                - (4 / rho / (4 - cp.asarray(Ux_free_count[0:nx, 0:ny])))
                * dtx
                * (T1[1 : nx + 1, 0:ny] - T1[0:nx, 0:ny] + T5[0:nx, 1 : ny + 1] - T5[0:nx, 0:ny]),
                0,
            )

            Uy[1:nx, 1:ny] = cp.where(
                cp.asarray(Uy_free_count[1:nx, 1:ny]) < 4,
                Uy[1:nx, 1:ny]
                - (4 / rho / (4 - cp.asarray(Uy_free_count[1:nx, 1:ny])))
                * dty
                * (T3[1:nx, 1:ny] - T3[1:nx, 0 : ny - 1] + T5[1:nx, 1:ny] - T5[0 : nx - 1, 1:ny]),
                0,
            )
            cp.cuda.Device().synchronize()
            if t > int(3.5e-6 / dt) and t < int(10e-6 / dt):
                wave[t] = cp.mean(T1[1, sy_l:sy_r])
            if time_idx < nt_sample and t * dt >= t_samples[time_idx, 0]:
                for j in range(int(nx_sample * ny_sample)):
                    ix = i_x_all[time_idx, j]
                    iy = i_y_all[time_idx, j]

                    print(f"time_idx: {time_idx}, ix: {ix}, iy: {iy}")

                    T1_sampled[time_idx, j] = cp.asnumpy(T1[ix, iy])
                    T3_sampled[time_idx, j] = cp.asnumpy(T3[ix, iy])
                    Ux_sampled[time_idx, j] = cp.asnumpy(Ux[ix, iy])
                    Uy_sampled[time_idx, j] = cp.asnumpy(Uy[ix, iy])

                time_idx += 1

        wave = cp.asnumpy(wave)

        yf, freq = sourse_new.make_fftdata(
            sourse_new.kiritori2(sourse_new.interpolate_sim_one(wave)[8000:], sourse_new.left, sourse_new.right),
            sourse_new.exp_dt,
        )
        ex_freq, ex_yf = sourse_new.extract_frequency_band(freq, yf, 2e6, 8e6)
        #  """

        #  """
        print(f"f_pitch = {f_pitch}")
        print(f"f_depth = {f_depth}")

        # 出力ディレクトリを作成
        os.makedirs("tmp_output", exist_ok=True)
        os.makedirs(r"C:\Users\manat\project2\PINN_data2", exist_ok=True)

        name = f"tmp_output\\cupy_pitch{int(f_pitch * 100000)}_depth{int(f_depth * 100000)}.csv"
        np.savetxt(name, wave, delimiter=",")
        # データ保存
        np.savez(
            f"C:\\Users\\manat\\project2\\PINN_data\\p{int(f_pitch * 1e6)}_d{int(f_depth * 1e6)}.npz",
            # パラメータ
            p=f_pitch,
            d=f_depth,
            w=f_width,
            # サンプリング座標
            x=x_samples.flatten(),  # (nt*nx*ny,)
            y=y_samples.flatten(),
            t=t_samples.flatten(),
            # 波動場データ
            T1=T1_sampled.flatten(),  # (nt*nx*ny,)
            T3=T3_sampled.flatten(),
            Ux=Ux_sampled.flatten(),
            Uy=Uy_sampled.flatten(),
            # メタデータ
            nx_sample=nx_sample,
            ny_sample=ny_sample,
            nt_sample=nt_sample,
            # プローブデータ
            y_probe=sy,
            t_probe=np.arange(len(wave)) * sourse_new.dt,
            reflection=wave,
            fft_freq=ex_freq,  # 周波数軸 (2-8 MHz)
            fft_amplitude=ex_yf,  # 振幅スペクトル,
            # サンプリング設定
            seed=42,
        )

        print(f"Completed: pitch={f_pitch * 1e3:.2f}mm, depth={f_depth * 1e3:.2f}mm")

        print(f"\n{'=' * 60}")
        print("All simulations completed!")
        print(f"Total: {len(pitch_list)} pitches × {len(depth_list)} depths = {len(pitch_list) * len(depth_list)} simulations")
        print(f"{'=' * 60}")

        # """
