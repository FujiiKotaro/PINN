# Task 5.3: 損失重み初期値設定と検証

## 目的

無次元化による損失スケール統一の効果を検証し、損失重み初期値（w_data=1.0, w_pde=1.0, w_bc=0.0）の妥当性を確認する。

## 背景

Phase 1では損失スケールの不均衡が問題となっていた：
- Data loss: O(1e-6) - 非常に小さい
- PDE loss: O(1e2) - 非常に大きい
- BC loss: O(1e-4) - 小さい

この不均衡により、適切な損失重みの設定が困難であった。

## 無次元化による解決策

Phase 2では、全変数を無次元化することで損失スケールをO(1)に統一：

### 無次元化変換

- **座標**: x̃ = x/L_ref, ỹ = y/L_ref, t̃ = t/T_ref
- **応力**: T̃1 = T1/σ_ref, T̃3 = T3/σ_ref
- **変位**: Ũx = Ux/U_ref, Ũy = Uy/U_ref

### 特性スケール（Aluminum 6061-T6）

- L_ref = 0.04 m (domain length)
- T_ref = 6.46e-6 s (L_ref / c_l)
- σ_ref = 1.03e11 Pa (ρ * c_l²)
- U_ref ≈ 1.12e-9 m (from FDTD data)

## 損失重み初期値

### 設定値（`configs/pinn_2d_example.yaml`）

```yaml
loss_weights:
  pde: 1.0    # PDE residual loss
  data: 1.0   # FDTD data fitting loss
```

### BC/IC重み

2D parametric PINNでは明示的なBC/ICを使用しない（FDTDデータ監視に依存）:
- w_bc = 0.0 (no explicit boundary conditions)
- w_ic = 0.0 (no explicit initial conditions)

## 検証手順

### Step 1: 訓練初期の損失オーダー確認

訓練開始後の最初の数イテレーション（0-100 iterations）で各損失項のオーダーを確認：

```python
# 期待される損失オーダー（無次元化後）
Expected:
  Data loss: O(1) ~ O(10)
  PDE loss: O(1) ~ O(10)
  Total loss: O(1) ~ O(10)
```

### Step 2: 損失ログ記録

DeepXDEの訓練ログから損失値を抽出：

```
Iteration 0:
  Train loss: [pde_loss, data_loss] = [X.XX, Y.YY]

Iteration 100:
  Train loss: [pde_loss, data_loss] = [X.XX, Y.YY]
```

### Step 3: 損失バランス検証

両方の損失項が同じオーダーであることを確認：

```python
assert 0.1 < (pde_loss / data_loss) < 10.0  # 同じオーダー
```

## 期待される結果

### ✓ 成功基準

1. **損失オーダー統一**: 全損失項がO(1)～O(10)範囲内
2. **損失バランス**: PDE loss / Data loss ≈ 1.0 (factor of 10以内)
3. **訓練安定性**: 損失が単調減少、発散なし
4. **重み調整不要**: 初期値（1.0, 1.0）で良好な訓練が可能

### ⚠ 失敗基準

1. **損失スケール不均衡**: いずれかの損失が極端に大きい/小さい（>100倍差）
2. **訓練発散**: 損失が増加または発散
3. **重み調整必要**: 初期値では訓練が収束せず、手動調整が必要

## 検証実施

### 実行コマンド

```bash
python pinn/training/train_2d.py configs/pinn_2d_example.yaml --data-dir /PINN_data
```

### ログ確認項目

1. **Iteration 0の損失値**:
   - Train loss: [pde_loss, data_loss]
   - 両方がO(1)～O(10)であることを確認

2. **Iteration 100の損失値**:
   - 損失が減少傾向にあることを確認
   - 両損失項のバランスが維持されていることを確認

3. **訓練中の損失推移**:
   - 単調減少または安定した振動
   - 発散や極端なスパイクがないこと

## 検証結果テンプレート

```markdown
### 検証結果 (YYYY-MM-DD)

#### 損失オーダー（Iteration 0）
- PDE loss: X.XXe±YY
- Data loss: X.XXe±YY
- Total loss: X.XXe±YY
- Ratio (PDE/Data): X.XX

#### 損失オーダー（Iteration 100）
- PDE loss: X.XXe±YY
- Data loss: X.XXe±YY
- Total loss: X.XXe±YY
- Ratio (PDE/Data): X.XX

#### 判定
- ✓ 損失オーダー統一: [OK/NG]
- ✓ 損失バランス: [OK/NG]
- ✓ 訓練安定性: [OK/NG]

#### 結論
[無次元化により損失スケール問題が解決されたか]
```

## トラブルシューティング

### 問題: 損失スケール不均衡

**症状**: PDE loss >> Data loss または Data loss >> PDE loss

**原因候補**:
1. 無次元化が正しく適用されていない
2. 特性スケール（σ_ref, U_ref）の選択が不適切
3. PDE係数の計算誤り

**対策**:
1. `train_2d.py`のStep 3で無次元化データ範囲を確認
2. CharacteristicScalesの計算ロジックを確認
3. PDEDefinition2DServiceのPDE係数を検証

### 問題: 訓練発散

**症状**: 損失が増加または NaN

**原因候補**:
1. 学習率が高すぎる
2. ネットワーク初期化の問題
3. データの異常値

**対策**:
1. 学習率を下げる（0.001 → 0.0001）
2. Glorot初期化を確認
3. データ範囲とNaN/Infを確認

## 参考

- Phase 1 Issue: 損失スケール不均衡（data: O(1e-6), PDE: O(1e2)）
- Phase 2 Solution: 無次元化により全損失をO(1)に統一
- Design: `.kiro/specs/pinn-2d-fdtd-integration/design.md`
- Requirements: `.kiro/specs/pinn-2d-fdtd-integration/requirements.md` (Req 1.1, 5.1)
