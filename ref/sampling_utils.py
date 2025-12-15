# sampling_utils.py

import numpy as np
from scipy.stats import qmc


class SpatioTemporalSampler:
    """
    空間・時間のサンプリング座標を生成
    PINNとFDTDで共有
    """

    def __init__(self, x_range, y_range, t_range, seed=42):
        """
        Args:
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            t_range: (t_min, t_max)
            seed: 乱数シード
        """
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.t_min, self.t_max = t_range
        self.seed = seed

    def sample_lhs_spatial(self, nx, ny, seed_offset=0):
        """
        空間座標をラテン超方格サンプリング

        Args:
            nx: x方向のサンプル数
            nz: z方向のサンプル数
            seed_offset: シードのオフセット（異なる時刻用）

        Returns:
            x_coords: shape (nx*nz,)
            z_coords: shape (nx*nz,)
            indices: (i_x, i_z) のインデックスペア
        """
        # ラテン超方格サンプラー
        sampler = qmc.LatinHypercube(d=2, seed=self.seed + seed_offset)
        samples = sampler.random(n=nx * ny)

        # [0,1] の範囲から実際の座標へスケーリング
        x_coords = samples[:, 0] * (self.x_max - self.x_min) + self.x_min
        y_coords = samples[:, 1] * (self.y_max - self.y_min) + self.y_min
        return x_coords, y_coords

    def sample_lhs_temporal(self, nt):
        """
        時間座標をラテン超方格サンプリング

        Args:
            nt: 時刻のサンプル数

        Returns:
            t_coords: shape (nt,)
        """
        sampler = qmc.LatinHypercube(d=1, seed=self.seed)
        samples = sampler.random(n=nt)

        t_coords = samples[:, 0] * (self.t_max - self.t_min) + self.t_min
        t_coords = np.sort(t_coords)  # 時刻は昇順にソート

        return t_coords

    def sample_time_slices(self, nx, ny, nt):
        """
        各時刻で空間サンプリング（時刻ごとに異なるパターン）

        Args:
            nx: x方向のサンプル数
            ny: y方向のサンプル数
            nt: 時刻のサンプル数

        Returns:
            x_all: shape (nt, nx*ny)
            y_all: shape (nt, nx*ny)
            t_all: shape (nt, nx*ny)
        """
        # 時刻座標
        t_coords = self.sample_lhs_temporal(nt)

        x_all = []
        y_all = []
        t_all = []

        for i, t in enumerate(t_coords):
            # 各時刻で異なる空間パターン
            x_spatial, y_spatial = self.sample_lhs_spatial(nx, ny, seed_offset=i)

            x_all.append(x_spatial)
            y_all.append(y_spatial)
            t_all.append(np.full_like(x_spatial, t))

        return (
            np.array(x_all),  # (nt, nx*ny)
            np.array(y_all),  # (nt, nx*ny)
            np.array(t_all),  # (nt, nx*ny)
        )

    def to_fdtd_indices(self, x_coords, y_coords, dx, dy):
        """
        実座標をFDTDメッシュのインデックスに変換

        Args:
            x_coords: 実座標 (m)
            y_coords: 実座標 (m)
            dx: FDTDのxメッシュ間隔 (m)
            dy: FDTDのyメッシュ間隔 (m)

        Returns:
            i_x: xインデックス
            i_y: yインデックス
        """
        # 座標の原点をFDTDの原点に合わせる
        x_offset = self.x_min
        y_offset = self.y_min

        # 配列サイズを計算（FDTDのUx, Uyのサイズに合わせる）
        nx_max = int((self.x_max - self.x_min) / dx)  # 2000
        ny_max = int((self.y_max - self.y_min) / dy)  # 4000
        i_x = np.round((x_coords - x_offset) / dx).astype(int)
        i_y = np.round((y_coords - y_offset) / dy).astype(int)

        # インデックスをクリップ（0 から nx-1, ny-1 の範囲に収める）
        i_x = np.clip(i_x, 0, nx_max - 1)
        i_y = np.clip(i_y, 0, ny_max - 1)
        return i_x, i_y
