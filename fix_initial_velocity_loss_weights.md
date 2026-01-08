# 初期速度問題の修正案

## 問題
PINNが初期速度条件 `u_t(x,0) = -c*f'(x)` を学習できず、波が両方向に分裂している

## 診断
- 解析解：右方向に単一の波（振幅1.0）
- PINN：両方向に分裂した波（振幅0.5）
- 相対誤差：48953.67%

## 根本原因
1. BC損失が高すぎて（4.39）、初期速度の学習を圧倒
2. 実効的な損失寄与度：BC (175.6) >> IC velocity (14.2)

## 修正案1: 損失重みの調整

```python
# 現在の設定（ノートブック セル7）
loss_weights = [
    config.training.loss_weights["bc"] * 2.0,  # 40.0
    config.training.loss_weights["ic_displacement"],  # 20.0
    config.training.loss_weights["ic_velocity"] * 2.0,  # 400.0
    config.training.loss_weights["pde"],  # 1.0
]

# 提案：初期速度を最優先にする
loss_weights = [
    5.0,   # BC - 大幅に削減
    20.0,  # IC displacement
    1000.0,  # IC velocity - さらに大幅に増加
    1.0,   # PDE
]
```

## 修正案2: 段階的学習

```python
# ステップ1: 初期条件だけを重視して学習
loss_weights_stage1 = [1.0, 50.0, 2000.0, 1.0]
model.compile("adam", lr=0.001, loss_weights=loss_weights_stage1)
model.train(iterations=5000)

# ステップ2: 全体をバランスよく学習
loss_weights_stage2 = [10.0, 20.0, 500.0, 1.0]
model.compile("adam", lr=0.0001, loss_weights=loss_weights_stage2)
model.train(iterations=5000)
```

## 修正案3: サンプリング点の大幅増加

```python
# 現在
data = dde.data.TimePDE(
    geomtime, pde_func, constraints,
    num_domain=5000,
    num_boundary=200,
    num_initial=1000,  # これを増やす
)

# 提案
data = dde.data.TimePDE(
    geomtime, pde_func, constraints,
    num_domain=5000,
    num_boundary=100,  # BCを減らす
    num_initial=3000,  # 初期条件を3倍に
)
```

## 修正案4: 因果的重み付きサンプリング

初期時刻に近い領域のPDE損失を重視：
- t=0付近のPDEサンプルを増やす
- causal_betaを調整

## 推奨アクション

1. まず**修正案1と3を組み合わせ**て試す
2. それでもダメなら**修正案2（段階的学習）**を試す
3. 学習後、初期速度 `u_t(x,0)` を明示的にプロットして検証

## 検証方法

```python
# 学習後、t=0でのu_tを確認
x_test = np.linspace(0, L, 200)
xt_test = np.column_stack([x_test, np.zeros_like(x_test)])
u_t_pred = model.predict(xt_test, operator=lambda x, y: dde.grad.jacobian(y, x, j=1))

# 解析解と比較
u_t_analytical = initial_velocity_func(x_test)
plt.plot(x_test, u_t_analytical, label='Analytical')
plt.plot(x_test, u_t_pred, '--', label='PINN')
plt.legend()
plt.title('Initial Velocity Check')
```
