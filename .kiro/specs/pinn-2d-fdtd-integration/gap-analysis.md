# 実装ギャップ分析: pinn-2d-fdtd-integration

## 分析サマリー

- **スコープ**: Phase 1（pinn-1d-foundation）の基盤を活用し、2D弾性波PINN + FDTD統合を実装
- **既存資産**: 12個のService class、完全なGPU訓練パイプライン、FDTDデータローダー（既にマルチ出力対応）が利用可能
- **主要ギャップ**: (1) 2D geometry + 弾性波PDE定義、(2) マルチ出力ネットワーク対応、(3) conditional PINN（パラメトリック入力）、(4) R²スコア計算、(5) 時系列スナップショット可視化
- **推奨アプローチ**: Hybrid（既存コンポーネント拡張 + 新規モジュール追加）
- **複雑度**: Medium（3-7日）、Phase 1パターンを踏襲しつつ、DeepXDEの2D geometry + マルチ出力機能を活用

---

## 1. 現状調査

### 1.1 既存資産マップ

#### Phase 1で実装済みの再利用可能コンポーネント

| コンポーネント | ファイルパス | 再利用可否 | 備考 |
|--------------|------------|----------|------|
| **FDTDDataLoaderService** | `pinn/data/fdtd_loader.py` | ✅ そのまま使用可 | 既にT1, T3, Ux, Uyの4出力対応済み。80,000点のサンプルデータ読み込み検証済み |
| **TensorConverterService** | `pinn/data/tensor_converter.py` | ✅ そのまま使用可 | NumPy → PyTorch tensor変換をサポート |
| **ConfigLoaderService** | `pinn/utils/config_loader.py` | 🔧 拡張必要 | DomainConfig（x_max, y_max追加）、ElasticConfig（λ, μ, ρ）追加が必要 |
| **TrainingPipelineService** | `pinn/training/training_pipeline.py` | ✅ 継承可能 | GPU検出、AMP、callbacks統合が完成済み |
| **AMPWrapperService** | `pinn/training/amp_wrapper.py` | ✅ そのまま使用可 | GPU混合精度訓練をサポート |
| **Callbacks** | `pinn/training/callbacks.py` | ✅ そのまま使用可 | LossLogging, Checkpoint, Divergence Detection |
| **WeightTuningFramework** | `pinn/tuning/weight_tuning.py` | ✅ そのまま使用可 | Grid/Random searchによる損失重み最適化 |
| **ErrorMetricsService** | `pinn/validation/error_metrics.py` | 🔧 拡張必要 | L2/相対誤差は実装済み。R²スコア計算を追加 |
| **PlotGeneratorService** | `pinn/validation/plot_generator.py` | 🔧 拡張必要 | 訓練カーブは実装済み。時系列スナップショット可視化（2D heatmap）を追加 |
| **ExperimentManager** | `pinn/utils/experiment_manager.py` | ✅ そのまま使用可 | タイムスタンプ付きディレクトリ作成、設定保存 |
| **SeedManager** | `pinn/utils/seed_manager.py` | ✅ そのまま使用可 | NumPy/PyTorch/Python random seed統一管理 |
| **MetadataLogger** | `pinn/utils/metadata_logger.py` | 🔧 拡張必要 | 使用FDTDファイルリスト、パラメータ範囲の記録を追加 |

#### 新規実装が必要なコンポーネント

| コンポーネント | 理由 | 複雑度 |
|-------------|------|-------|
| **Elastic2DPDEDefinitionService** | Phase 1は1D波動方程式のみ。2D弾性波（縦波・横波）のPDE residual計算が必要 | Medium |
| **ConditionalPINNModelBuilder** | Phase 1はパラメータ条件付き学習未対応。入力次元を(x,y,t)→(x,y,t,pitch,depth)に拡張 | Medium |

### 1.2 既存コードの詳細分析

#### PINNModelBuilderService（`pinn/models/pinn_model_builder.py`）

**現在の実装**:
```python
def _create_geometry(self, domain: DomainConfig) -> dde.geometry.GeometryXTime:
    geom = dde.geometry.Interval(domain.x_min, domain.x_max)  # 1D spatial
    timedomain = dde.geometry.TimeDomain(domain.t_min, domain.t_max)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    return geomtime
```

**Phase 2への拡張戦略**:
- `dde.geometry.Interval` → `dde.geometry.Rectangle` (2D spatial domain)
- 入力: `[x_min, x_max] → [[x_min, y_min], [x_max, y_max]]`
- 出力: `dde.geometry.GeometryXTime(rectangle, timedomain)`

#### PDEDefinitionService（`pinn/models/pde_definition.py`）

**現在の実装**:
```python
def wave_equation_residual(x, u, c):
    du_xx = dde.grad.jacobian(dde.grad.jacobian(u, x, i=0, j=0), x, i=0, j=0)
    du_tt = dde.grad.jacobian(dde.grad.jacobian(u, x, i=0, j=1), x, i=0, j=1)
    residual = du_tt - (c ** 2) * du_xx  # 1D wave equation
    return residual
```

**Phase 2への拡張戦略**:
- 新規Service作成: `Elastic2DPDEDefinitionService`
- 4出力（T1, T3, Ux, Uy）に対する個別のPDE residual計算
- 弾性波方程式:
  - 縦波: `∂²u/∂t² = (λ+2μ)/ρ ∇²u`
  - 横波: `∂²v/∂t² = μ/ρ ∇²v`
- DeepXDEの`dde.grad.jacobian`でx, y方向の偏微分を計算

#### FDTDDataLoaderService（`pinn/data/fdtd_loader.py`）

**現状**:
- ✅ 既にT1, T3, Ux, Uyの4出力フィールド読み込み対応済み
- ✅ 80,000点のサンプルデータ（nx_sample × ny_sample × nt_sample）読み込み検証済み
- ✅ pitch/depth範囲フィルタリング機能実装済み（`load_multiple`メソッド）

**Phase 2での利用**:
- そのまま使用可能（変更不要）
- `load_multiple(pitch_range=(0.00125, 0.002), depth_range=(0.0001, 0.0003))`で12ファイル読み込み

#### ConfigLoaderService（`pinn/utils/config_loader.py`）

**現在の設定構造**:
```python
class DomainConfig(BaseModel):
    x_min, x_max, t_min, t_max, wave_speed
```

**Phase 2への拡張**:
```python
class DomainConfig(BaseModel):
    x_min, x_max, y_min, y_max  # 2D spatial domain
    t_min, t_max
    # wave_speed → elastic constants

class ElasticConfig(BaseModel):
    lame_lambda: float  # λ (Lamé's first parameter)
    lame_mu: float      # μ (shear modulus)
    density: float      # ρ
```

### 1.3 既存アーキテクチャパターン

**確認されたパターン**:
1. **Service-based design**: すべての機能が`*Service`クラスに実装
2. **Pydantic validation**: 設定ファイルの型安全性とバリデーション
3. **DeepXDE native integration**: `dde.Model`, `dde.data.TimePDE`を直接使用
4. **Callback pattern**: 訓練監視・チェックポイント保存をcallbackで実装
5. **Modular separation**: models/, training/, validation/, data/, tuning/の明確な責任分離

---

## 2. 要求事項の技術的実現可能性分析

### 要求1: 2D弾性波方程式PINNモデル

**技術的ニーズ**:
- 2D geometry定義（Rectangle + TimeDomain）
- 弾性波PDE residual計算（4出力: T1, T3, Ux, Uy）
- マルチ出力ニューラルネットワーク（input: 3次元 → output: 4次元）
- 弾性定数（λ, μ, ρ）の設定管理

**ギャップ**:
- ❌ **Missing**: 2D Rectangle geometryサポート（Phase 1は1D Intervalのみ）
- ❌ **Missing**: 弾性波PDE定義（Phase 1は1D波動方程式のみ）
- ❌ **Missing**: マルチ出力NN構成（Phase 1は単一出力）
- ✅ **Exists**: DeepXDEフレームワーク統合パターン、GPU加速

**実装アプローチ**:
- DeepXDEドキュメント参照: `dde.geometry.Rectangle([[x_min, y_min], [x_max, y_max]])`
- マルチ出力: `dde.nn.FNN([input_dim, ...hidden..., 4])`（最後の層を4出力に変更）
- 新規Service: `Elastic2DPDEDefinitionService`で弾性波方程式定義

**Research Needed**:
- DeepXDEでのマルチ出力PDE定義の具体的な構文（公式example確認）

---

### 要求2: FDTDデータローディングと訓練統合

**技術的ニーズ**:
- .npzファイル読み込み（x, y, t, T1, T3, Ux, Uy）
- pitch/depth範囲フィルタリング
- PyTorchテンソル変換
- 訓練/検証データ分割（80/20）

**ギャップ**:
- ✅ **Exists**: `FDTDDataLoaderService.load_multiple`で12ファイル読み込み可能
- ✅ **Exists**: `TensorConverterService`でPyTorchテンソル変換対応
- ✅ **Exists**: `SeedManager`でseed=42再現性保証
- ⚠️ **Constraint**: 80,000点/ファイルのサンプル点数（将来的により密なサンプリングが利用可能になる可能性）

**実装アプローチ**:
- Phase 1コンポーネントをそのまま使用（変更不要）
- 訓練/検証分割は`sklearn.model_selection.train_test_split`を使用（既存パターン）

---

### 要求3: パラメトリックPINN学習（クラック形状条件付き）

**技術的ニーズ**:
- 入力次元拡張: (x, y, t) → (x, y, t, pitch, depth)
- 複数パラメータ組み合わせの混合訓練
- 内挿予測性能評価

**ギャップ**:
- ❌ **Missing**: Conditional PINN実装（Phase 1は条件付き学習未対応）
- ❌ **Missing**: 5次元入力対応のネットワーク構成

**実装アプローチ**:
- 入力層を`[5, ...hidden..., 4]`に変更（5次元入力、4次元出力）
- DeepXDEの`dde.data.TimePDE`にFDTDデータを供給する際、座標配列に`[x, y, t, pitch, depth]`を結合
- 訓練時: 複数の.npzファイルから読み込んだデータを混合（`np.concatenate`）

**Research Needed**:
- DeepXDEでの追加入力次元の扱い方（公式exampleでparametric PINNを確認）

---

### 要求4: FDTDデータとの比較検証

**技術的ニーズ**:
- R²スコア計算（決定係数）
- 各出力場（T1, T3, Ux, Uy）の個別評価
- 時系列スナップショット可視化（2D heatmap）
- 誤差分布ヒートマップ

**ギャップ**:
- ✅ **Exists**: L2誤差、相対誤差計算（`ErrorMetricsService`）
- ❌ **Missing**: R²スコア計算機能
- ❌ **Missing**: 2D空間分布の時系列スナップショット可視化
- ✅ **Exists**: matplotlib/seabornによるプロット基盤

**実装アプローチ**:
- R²スコア追加: `ErrorMetricsService.r2_score()`メソッドを新規追加
  - 実装: `1 - np.sum((y_pred - y_true)**2) / np.sum((y_true - y_true.mean())**2)`
- 時系列スナップショット: `PlotGeneratorService.plot_field_snapshots()`メソッドを新規追加
  - 2D heatmap（`plt.imshow` または `plt.contourf`）で空間分布を可視化
  - 複数時刻（例: t=4e-6s, 5e-6s, 6e-6s）のサブプロットを生成

---

### 要求5: Phase 1基盤コンポーネントの再利用

**ギャップ**: なし（すべて再利用可能）

**再利用戦略**:
- `TrainingPipelineService`: そのまま継承（`TrainingPipelineService2D`）
- `AMPWrapperService`, `Callbacks`, `WeightTuningFramework`: 変更不要
- `ConfigLoaderService`: `ElasticConfig`を追加するのみ

---

### 要求6: 簡素化されたテスト戦略

**技術的ニーズ**:
- 重要コンポーネントの単体テスト（PDE residual、R²スコア）
- 1つのE2Eシナリオ（小規模データ訓練→検証）
- Notebook実行による整合性確認

**ギャップ**:
- ✅ **Exists**: pytestインフラ、テストディレクトリ構造
- ⚠️ **Constraint**: Phase 1で過剰テストによるnotebook不整合を経験

**実装アプローチ**:
- テストカバレッジ目標: 50%（Phase 1の70%から引き下げ）
- 統合テスト: `test_2d_forward_e2e.py`（小規模FDTDデータで訓練→R²スコア計算）
- Notebook実行: CI/CD外で手動検証（notebook内で実装APIを直接使用）

---

### 要求7: Jupyter Notebookデモンストレーション

**技術的ニーズ**:
- 順問題ワークフロー（データ読み込み→訓練→FDTD比較）
- 変位/応力分布の時系列スナップショット可視化（最低3時刻）
- API整合性保証

**ギャップ**:
- ✅ **Exists**: Phase 1の`wave_1d_demo.ipynb`テンプレート
- ❌ **Missing**: 2D空間分布可視化（Phase 1は1D lineplot）

**実装アプローチ**:
- `wave_1d_demo.ipynb`を複製して`wave_2d_forward_demo.ipynb`作成
- データ読み込み: `FDTDDataLoaderService.load_file('p1250_d100.npz')`
- 訓練: `PINNModelBuilder2D.build_model()`
- 可視化: `PlotGeneratorService.plot_field_snapshots()`で3時刻スナップショット

---

### 要求8: 再現性と実験管理

**ギャップ**: なし（Phase 1コンポーネント全て再利用可能）

**実装アプローチ**:
- `ExperimentManager`: そのまま使用
- `MetadataLogger`: `log_fdtd_files()`メソッド追加（使用した.npzファイルリストを記録）

---

## 3. 実装アプローチオプション

### オプションA: 既存コンポーネント拡張

**拡張対象**:
1. `PINNModelBuilderService` → `_create_geometry()`を2D対応に修正
2. `PDEDefinitionService` → 弾性波方程式メソッドを追加
3. `ConfigLoaderService` → `DomainConfig`にy_min/y_max追加、`ElasticConfig`追加
4. `ErrorMetricsService` → `r2_score()`メソッド追加
5. `PlotGeneratorService` → `plot_field_snapshots()`メソッド追加

**互換性評価**:
- ✅ `_create_geometry()`はprivateメソッドなので拡張しても影響なし
- ⚠️ `PDEDefinitionService`に弾性波メソッド追加は可能だが、1D/2Dが混在してクラスが肥大化

**複雑度・保守性**:
- 👍 最小限のファイル変更
- 👎 1D/2Dロジックが同一ファイルに混在（将来的な保守が困難）

**トレードオフ**:
- ✅ Phase 1との互換性維持（既存テストが壊れない）
- ✅ 高速な初期開発
- ❌ コードの複雑化（1D/2D分岐が増加）
- ❌ 単一責任原則の違反

---

### オプションB: 新規コンポーネント作成

**新規作成対象**:
1. `Elastic2DPDEDefinitionService` (新規ファイル: `pinn/models/pde_definition_2d.py`)
   - 弾性波方程式のPDE residual計算
2. `ConditionalPINNModelBuilder` (新規ファイル: `pinn/models/pinn_model_builder_2d.py`)
   - 2D geometry + マルチ出力 + conditional input対応
3. `FieldVisualizationService` (新規ファイル: `pinn/validation/field_visualization.py`)
   - 2D空間分布の時系列スナップショット可視化

**統合ポイント**:
- Phase 1の`TrainingPipelineService`, `AMPWrapper`, `Callbacks`をそのまま使用
- `ConfigLoaderService`を拡張（`ElasticConfig`追加）
- `FDTDDataLoaderService`をそのまま使用

**責任境界**:
- `Elastic2DPDEDefinitionService`: 2D弾性波PDE定義のみ
- `ConditionalPINNModelBuilder`: 2D PINN構築（geometry + PDE + BC + IC + conditional input）
- `FieldVisualizationService`: 空間分布可視化のみ

**トレードオフ**:
- ✅ 明確な責任分離（1D/2Dが分離）
- ✅ Phase 1コードへの影響ゼロ
- ✅ 将来的な保守性向上
- ❌ ファイル数増加（3ファイル追加）
- ❌ Phase 1との類似コード重複（geometry作成ロジックなど）

---

### オプションC: ハイブリッドアプローチ（推奨）

**戦略**:
1. **新規作成**: 2D専用コンポーネント
   - `Elastic2DPDEDefinitionService` (新規)
   - `ConditionalPINNModelBuilder2D` (新規)
   - `FieldVisualizationService` (新規)

2. **拡張**: 汎用コンポーネント
   - `ConfigLoaderService`: `ElasticConfig`追加（既存`DomainConfig`は変更しない）
   - `ErrorMetricsService`: `r2_score()`追加（既存メソッドは変更しない）
   - `MetadataLogger`: `log_fdtd_files()`追加

3. **そのまま使用**: 基盤コンポーネント
   - `FDTDDataLoaderService`, `TensorConverterService`, `TrainingPipelineService`, `AMPWrapperService`, `Callbacks`, `WeightTuningFramework`, `ExperimentManager`, `SeedManager`

**段階的実装**:
- **Phase 2.1（順問題検証）**:
  1. `Elastic2DPDEDefinitionService`実装（弾性波PDE）
  2. `ConditionalPINNModelBuilder2D`実装（2D geometry + マルチ出力）
  3. `FieldVisualizationService`実装（時系列スナップショット）
  4. `wave_2d_forward_demo.ipynb`作成（単一パラメータでの訓練→FDTD比較）

- **Phase 2.2（パラメトリック学習）**:
  1. `ConditionalPINNModelBuilder2D`にconditional input機能追加（pitch, depth）
  2. 複数パラメータ混合訓練のnotebook追加

**リスク軽減**:
- Phase 1コンポーネントは変更しないため、既存テストが壊れない
- 順問題検証後に逆問題へ進むため、基盤の正しさを段階的に確認

**トレードオフ**:
- ✅ Phase 1の安定性を維持
- ✅ 2D専用ロジックを明確に分離
- ✅ 段階的な機能追加が可能
- ❌ 計画が複雑（2段階に分割）
- ⚠️ 一部コードの重複（geometry作成ロジックなど）

---

## 4. 実装複雑度とリスク評価

### 全体的な複雑度: **Medium（3-7日）**

**理由**:
- ✅ Phase 1の安定した基盤が存在（GPU訓練、データローディング、設定管理）
- ✅ DeepXDEの2D geometryおよびマルチ出力NNは公式サポート済み（ドキュメント存在）
- ⚠️ 弾性波PDE定義は新規実装が必要だが、1D波動方程式パターンを踏襲可能
- ⚠️ Conditional PINN（パラメトリック入力）はDeepXDEのexampleを参照する必要あり

### コンポーネント別の複雑度・リスク

| コンポーネント | 複雑度 | リスク | 理由 |
|-------------|-------|-------|------|
| `Elastic2DPDEDefinitionService` | **M (3-5日)** | **Medium** | 弾性波方程式の偏微分計算は新規。DeepXDEの`grad.jacobian`パターンは既知 |
| `ConditionalPINNModelBuilder2D` | **M (2-4日)** | **Medium** | 2D geometryはDeepXDE標準機能。Conditional inputの実装パターン要調査 |
| `FieldVisualizationService` | **S (1-2日)** | **Low** | matplotlibのheatmap可視化は標準的。Phase 1のプロットパターン踏襲可能 |
| `ConfigLoaderService拡張` | **S (0.5日)** | **Low** | Pydantic modelへのフィールド追加のみ |
| `ErrorMetricsService拡張` | **S (0.5日)** | **Low** | R²スコア計算式は定型的 |
| `wave_2d_forward_demo.ipynb` | **M (2-3日)** | **Medium** | Phase 1 notebookテンプレート流用可能だが、API整合性確保が課題（Phase 1の反省） |
| 統合テスト | **S (1日)** | **Low** | テスト範囲を限定（50%カバレッジ目標） |

### 総合リスク評価: **Medium**

**High Riskエリア（要注意）**:
1. **Conditional PINN実装**: DeepXDEでのparametric input拡張パターンが不明
   - 軽減策: 公式example/documentationを事前調査（設計フェーズで実施）

2. **Notebook-実装整合性**: Phase 1でnotebook不整合を経験
   - 軽減策: notebook開発時に実装APIを直接使用（import pathを完全一致）、順問題検証後に逆問題へ進む

**Medium Riskエリア**:
1. **弾性波PDE residual計算**: 4出力（T1, T3, Ux, Uy）の個別PDE定義が必要
   - 軽減策: Phase 1の1D波動方程式パターンを参考に、段階的に実装

**Low Riskエリア**:
- Phase 1基盤コンポーネントの再利用（GPU訓練、データローダー、callbacks）
- R²スコア計算、2D可視化（標準的なNumPy/matplotlib処理）

---

## 5. 要求-資産マッピング

| 要求 | 既存資産 | ギャップ | 実装アプローチ |
|-----|---------|---------|--------------|
| **要求1: 2D弾性波PINNモデル** | `PDEDefinitionService`(1D), `PINNModelBuilderService`(1D) | ❌ 2D geometry未対応、❌ 弾性波PDE未定義、❌ マルチ出力NN未対応 | 新規Service: `Elastic2DPDEDefinitionService`, `ConditionalPINNModelBuilder2D` |
| **要求2: FDTDデータローディング** | `FDTDDataLoaderService`, `TensorConverterService` | ✅ 完全対応済み | そのまま使用 |
| **要求3: パラメトリック学習** | - | ❌ Conditional input未実装 | `ConditionalPINNModelBuilder2D`にpitch/depth入力機能追加 |
| **要求4: FDTD比較検証** | `ErrorMetricsService`(L2/相対誤差), `PlotGeneratorService`(訓練カーブ) | ❌ R²スコア未実装、❌ 時系列スナップショット可視化未実装 | `ErrorMetricsService.r2_score()`追加、`FieldVisualizationService`新規作成 |
| **要求5: Phase 1再利用** | 12個のService class（GPU訓練、設定管理、callbacks、tuning） | ✅ 完全再利用可能 | そのまま使用 |
| **要求6: 簡素化テスト** | pytestインフラ | ⚠️ Phase 1で過剰テスト経験 | テストカバレッジ50%、E2E 1シナリオのみ |
| **要求7: Notebook** | `wave_1d_demo.ipynb`テンプレート | ❌ 2D可視化未対応 | `wave_2d_forward_demo.ipynb`新規作成 |
| **要求8: 再現性** | `ExperimentManager`, `SeedManager`, `MetadataLogger` | ✅ 再利用可能 | `MetadataLogger.log_fdtd_files()`追加のみ |

---

## 6. 設計フェーズへの推奨事項

### 推奨実装アプローチ: **オプションC（ハイブリッド）**

**理由**:
- Phase 1の安定した基盤を保持しつつ、2D専用ロジックを明確に分離
- 順問題検証→逆問題の段階的実装により、リスクを軽減
- Phase 1との類似コード重複は許容範囲内（保守性とのトレードオフ）

### 重要な設計決定事項

1. **DeepXDEのConditional PINN実装パターン調査**
   - 公式example: [DeepXDE Parametric PINN examples](https://deepxde.readthedocs.io/)
   - 入力次元拡張の具体的な構文確認（設計フェーズで実施）

2. **2D Geometry + Multi-output NN構成**
   - `dde.geometry.Rectangle` + `dde.nn.FNN([5, ...hidden..., 4])`の動作確認
   - 弾性波PDEでの4出力（T1, T3, Ux, Uy）の個別residual計算方法

3. **Notebook-実装整合性の保証戦略**
   - 実装完了後にnotebook作成（実装→notebook順）
   - notebook内で直接`from pinn.models.xxx import xxx`を使用（中間ラッパー禁止）

### 設計フェーズで実施すべきResearch項目

1. **DeepXDE Parametric PINN Example調査** (Priority: High)
   - 公式documentationでconditional input（パラメータ条件付き学習）の実装パターンを確認
   - 入力次元拡張の構文: `dde.data.TimePDE`に追加座標を渡す方法

2. **Multi-output PDE Definition** (Priority: High)
   - DeepXDEで4出力ニューラルネットワークに対する個別PDE residual定義の方法
   - 各出力（T1, T3, Ux, Uy）に異なるPDEを適用する構文

3. **2D Boundary Conditions** (Priority: Medium)
   - `dde.geometry.Rectangle`での境界条件定義（on_boundary predicateの記述方法）

### 次のステップ

設計フェーズ（`/kiro:spec-design pinn-2d-fdtd-integration`）で以下を実施:

1. **アーキテクチャ設計**:
   - `Elastic2DPDEDefinitionService`, `ConditionalPINNModelBuilder2D`, `FieldVisualizationService`の詳細設計
   - Phase 1コンポーネントとの統合フロー図作成

2. **Research項目の解決**:
   - DeepXDE parametric PINN exampleの調査結果を反映
   - Multi-output PDE定義の具体的な実装方針決定

3. **段階的実装計画**:
   - Phase 2.1（順問題検証）とPhase 2.2（パラメトリック学習）のタスク分解
   - 各タスクの依存関係と実装順序の明確化

---

## 分析手法

本ギャップ分析は`.kiro/settings/rules/gap-analysis.md`のフレームワークに従い、以下の手法で実施しました:

1. **Current State Investigation**: Phase 1の12個のService classを調査（Grep, Read tool使用）
2. **Requirements Feasibility Analysis**: 8つの要求事項に対する技術的実現可能性を評価
3. **Implementation Approach Options**: 3つのオプション（Extend, New, Hybrid）を比較分析
4. **Complexity & Risk Assessment**: コンポーネント別の複雑度（S/M/L）とリスク（Low/Medium/High）を評価
5. **Requirement-to-Asset Mapping**: 要求と既存資産のマッピング表を作成

---

**分析完了日**: 2025-12-16
**次フェーズ**: 技術設計（`/kiro:spec-design pinn-2d-fdtd-integration`）
