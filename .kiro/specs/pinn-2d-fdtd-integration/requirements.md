# 要求仕様書

## はじめに

本文書は、Phase 1（pinn-1d-foundation）で構築したPINN基盤を活用し、2次元弾性波方程式とFDTDデータを統合した実用的なPINNシステムの要求仕様を定義します。Phase 1では解析解による検証に留めましたが、Phase 2では実際のFDTDシミュレーションデータ（`/PINN_data/`内の.npzファイル）を用いた訓練・評価を実施し、クラック形状パラメータ（pitch、depth）に対する超音波伝播の予測モデルを構築します。

**スコープ**: 2D弾性波PINN（T1, T3応力場およびUx, Uy変位場の予測）、FDTD訓練データ統合、パラメトリック学習（pitch: 1.25-2.0mm、depth: 0.1-0.3mm）、Phase 1で構築済みの基盤コンポーネント再利用

**スコープ外**: 3次元波動、リアルタイム推論最適化、クラック検出アルゴリズムの実装、本番環境デプロイ、過度なテストコード（Phase 1の反省を踏まえ、実装整合性に注力）

## 要求事項

### 要求1: 2D弾性波方程式PINNモデル

**目的:** 物理シミュレーション開発者として、2次元弾性波方程式を満たすPINNモデルが必要であり、T1/T3応力場とUx/Uy変位場を同時予測できるようにする。

#### 受入基準

1. The PINN Model shall 2次元弾性波方程式のPDE residualを計算する: ∂²u/∂t² = (λ+2μ)/ρ ∂²u（縦波方程式）および ∂²v/∂t² = μ/ρ ∂²v（横波方程式）
2. The PINN Model shall 空間領域[0, 40mm] × [0, 20mm]および時間領域[3.5e-6s, 6.5e-6s]をDeepXDEのGeometry制約として定義する
3. When モデルが初期化されると、the PINN Model shall 4出力（T1, T3, Ux, Uy）を持つマルチタスクニューラルネットワークを構築する
4. The PINN Model shall Phase 1で実装済みの`pinn.models.pinn_model_builder.PINNModelBuilderService`を拡張して2D geometryおよびマルチ出力対応を追加する
5. The PINN Model shall 弾性定数（λ, μ, ρ）を設定ファイルから読み込み可能な構成パラメータとして受け入れる

### 要求2: FDTDデータローディングと訓練統合

**目的:** 物理シミュレーション開発者として、`/PINN_data/`のFDTDシミュレーションデータを訓練に使用したいため、既存のデータローダーを活用してPINNモデルに供給する。

#### 受入基準

1. The Training Pipeline shall Phase 1で実装済みの`pinn.data.fdtd_loader.FDTDDataLoaderService`を使用して.npzファイルを読み込む
2. When 複数の.npzファイルが指定されると、the Data Loader shall pitch範囲[1.25mm, 2.0mm]およびdepth範囲[0.1mm, 0.3mm]でフィルタリングする
3. The Data Loader shall 空間座標(x, y)、時間座標(t)、および波動場データ(T1, T3, Ux, Uy)を抽出してPyTorchテンソルに変換する
4. The Training Pipeline shall 訓練データとして抽出されたFDTDサンプル点における(x, y, t) → (T1, T3, Ux, Uy)のマッピングを使用する
5. The Training Pipeline shall 訓練/検証データ分割（例: 80/20）を再現可能なseed=42で実施する
6. The Data Loader shall 現在のFDTDデータファイルのサンプリング点数をそのまま使用する（将来的により密なサンプリングデータが利用可能になった場合に対応可能な設計とする）
7. If .npzファイルが不正な形式の場合、then the Data Loader shall 具体的なエラーメッセージとともにValueErrorを発生させる

### 要求3: パラメトリックPINN学習（クラック形状条件付き）

**目的:** 物理シミュレーション開発者として、クラックパラメータ（pitch、depth）を条件とした学習を行い、訓練データ範囲内の未知パラメータ組み合わせに対して内挿予測が可能なモデルを構築したい。

#### 受入基準

1. The PINN Model shall クラックパラメータ（pitch, depth）を追加入力として受け入れるconditional PINN構成をサポートする
2. The PINN Model shall 入力次元を(x, y, t, pitch, depth)の5次元に拡張する
3. The Training Pipeline shall 複数のパラメータ組み合わせ（現在利用可能な12ファイル: 4 pitch × 3 depth）のデータを混合して訓練する
4. When 訓練済みパラメータ範囲内の未学習組み合わせ（例: pitch=1.625mm, depth=0.15mm）で推論すると、the PINN Model shall 内挿予測を行う
5. The Validation Module shall 1つ以上のholdoutパラメータ組み合わせを用いて内挿性能を評価する
6. The System shall 現在のデータ量では外挿（訓練範囲外のパラメータ予測）は困難であることを認識し、内挿性能の達成を優先目標とする（将来データ量が増加した場合の外挿対応は拡張目標）

### 要求4: FDTDデータとの比較検証

**目的:** 物理シミュレーション開発者として、PINNの予測精度をFDTD ground truthと比較して定量評価し、R²スコアで性能を測定したい。

#### 受入基準

1. The Validation Module shall Phase 1の`pinn.validation.error_metrics.ErrorMetricsService`を再利用してL2誤差および相対誤差を計算する
2. The Validation Module shall R²スコア（決定係数）を計算する機能を追加する: R² = 1 - Σ(y_pred - y_true)² / Σ(y_true - ȳ)²
3. When 訓練が完了すると、the Validation Module shall 各出力場（T1, T3, Ux, Uy）に対してR²スコアを個別に計算してログに記録する
4. The Validation Module shall FDTDデータとPINN予測を時系列で可視化するプロット（複数時刻スナップショット）を生成する
5. If R²スコアが0.9未満の場合、then the Validation Module shall 警告を発し、ハイパーパラメータ調整を推奨する
6. The Validation Module shall 空間領域全体での誤差分布ヒートマップを生成して誤差が集中する領域を特定する

### 要求5: Phase 1基盤コンポーネントの再利用

**目的:** 物理シミュレーション開発者として、Phase 1で検証済みのコンポーネント（GPU加速、設定管理、損失関数チューニング）を再利用し、開発効率を高めたい。

#### 受入基準

1. The Training Pipeline shall Phase 1の`pinn.training.training_pipeline.TrainingPipelineService`を継承または拡張して2D PINNに対応する
2. The Training Pipeline shall Phase 1の`pinn.training.amp_wrapper.AMPWrapperService`をそのまま使用してGPU混合精度訓練を有効化する
3. The Configuration Loader shall Phase 1の`pinn.utils.config_loader.ConfigLoaderService`を使用してYAML設定ファイルを読み込む
4. The Training Pipeline shall Phase 1の`pinn.training.callbacks`（LossLoggingCallback、CheckpointCallback、DivergenceDetectionCallback）を再利用する
5. The Tuning Framework shall Phase 1の`pinn.tuning.weight_tuning.WeightTuningFrameworkService`を使用して損失重み（w_data, w_pde, w_bc）の最適化を実施する
6. When 新規コンポーネントを追加する際、the System shall Phase 1のディレクトリ構造（`/pinn/models/`, `/pinn/training/`, `/pinn/validation/`）を維持する

### 要求6: 簡素化されたテスト戦略

**目的:** 物理シミュレーション開発者として、Phase 1で発生したテストと実装の不整合を回避し、実装整合性に注力した効率的なテスト戦略を採用したい。

#### 受入基準

1. The Testing Strategy shall 単体テストを重要なコンポーネント（PDE residual計算、FDTDデータローディング、R²スコア計算）に限定する
2. The Testing Strategy shall 統合テストを1つのエンドツーエンドシナリオ（小規模データセットでの訓練→検証）に絞る
3. The Testing Strategy shall 実装とnotebookの整合性を保証するため、notebook内で実際の訓練パイプラインを実行する
4. The Testing Strategy shall Phase 1の過剰なモックテストを削減し、実データを用いた実行可能なテストを優先する
5. When テストが失敗した場合、the Testing Strategy shall 実装コードとnotebookの両方を同時に修正して整合性を維持する
6. The System shall テストカバレッジ目標を70%から50%に引き下げ、重要機能のカバレッジに注力する

### 要求7: Jupyter Notebookデモンストレーション（段階的実装）

**目的:** 物理シミュレーション開発者として、まず順問題（forward problem: 既知パラメータでの波動場予測）のワークフローをnotebookで検証し、その後逆問題（inverse problem: パラメータ推定）に進みたい。

#### 受入基準

1. The Demonstration Notebook shall `/notebooks/wave_2d_forward_demo.ipynb`として順問題用notebookを先に作成する
2. The Forward Problem Notebook shall Phase 1の`/notebooks/wave_1d_demo.ipynb`のテンプレート構造を再利用する
3. The Forward Problem Notebook shall FDTDデータの読み込みから2D PINN訓練（単一パラメータ組み合わせ）、FDTD比較プロットまでの完全なワークフローを含む
4. The Forward Problem Notebook shall 変位分布（Ux, Uy）または応力分布（T1, T3）の複数時間ステップ（最低3つのスナップショット）における空間分布を可視化する
5. When Forward Problem Notebookが実行されると、the Notebook shall エラーなく全セルが完了し、時系列スナップショットおよびR²スコアが生成される
6. The Forward Problem Notebook shall 各ステップに日本語の説明セルを含み、PINN手法の概要とFDTD統合の意義を解説する
7. The Forward Problem Notebook shall 実装コードと完全に一致するAPIを使用し、import pathやメソッド名の不整合がないことを保証する
8. The System shall 順問題notebookで正しく波動場予測ができることを確認した後、逆問題（パラメータ推定）用notebook実装に進む（逆問題notebookは順問題検証後のフェーズで追加）

### 要求8: 再現性と実験管理

**目的:** 物理シミュレーション開発者として、Phase 1と同様の再現性保証を維持しつつ、パラメトリック学習の実験を追跡可能にしたい。

#### 受入基準

1. The Experiment Manager shall Phase 1の`pinn.utils.experiment_manager.ExperimentManager`を使用してタイムスタンプ付き実験ディレクトリを作成する
2. The Experiment Manager shall 訓練に使用したクラックパラメータ組み合わせ（pitch, depth）をメタデータとして記録する
3. The Seed Manager shall Phase 1の`pinn.utils.seed_manager.SeedManager`を使用してNumPy、PyTorch、Pythonのrandomシードをseed=42に設定する
4. The Metadata Logger shall Phase 1の`pinn.utils.metadata_logger.MetadataLogger`を拡張して、使用したFDTDファイルリストとパラメータ範囲を記録する
5. When 実験が再実行されると、the System shall 同じseedおよび設定で同一の結果を再現する
6. The System shall 実験結果（チェックポイント、ログ、プロット）を`/experiments/exp_{timestamp}/`に保存する

## 非機能要求

### パフォーマンス
- 2D PINN訓練（10k collocation points、5k epochs）は GPU上で30分以内に完了すること
- 複数FDTD .npzファイル（計12ファイル、約56MB）の読み込みは10秒以内に完了すること
- PINN推論（1000点の時空間グリッド）は1秒以内に完了すること

### 保守性
- Phase 1のディレクトリ構造（`/pinn/models/`, `/pinn/training/`, `/pinn/validation/`, `/pinn/data/`）を維持すること
- Phase 1のService命名規則（`*Service`クラス）およびsnake_caseファイル名規則を遵守すること

### 互換性
- Python 3.11+、PyTorch 2.4.0、DeepXDE 1.15.0、CUDA 12.4との互換性を維持すること
- Phase 1で実装済みのコンポーネント（config_loader、seed_manager、fdtd_loaderなど）をそのまま使用可能であること

### ドキュメント
- `/pinn/README.md`にPhase 2の使用例（2D PINN訓練スクリプト、FDTDデータ指定方法）を追加すること（日本語）
- Jupyter notebook（`/notebooks/wave_2d_forward_demo.ipynb`）が完全に動作し、順問題ワークフローを示すこと（説明セルは日本語）
- 各新規コンポーネントにNumPy docstring規則に従ったドキュメントを含めること（docstringは英語）
- すべてのPythonコードのインラインコメントは英語で記述すること（日本語コメント禁止）
