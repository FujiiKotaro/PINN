# 要求仕様書

## Project Description (Input)
pinn-2d-fdtd-integrationの実装が完了したらしいので，これらの実装を用いて実際に順問題を解いて検証するようなノートブックを制作して．

## はじめに

本文書は、pinn-2d-fdtd-integration（Phase 2）で実装された2D弾性波PINNコンポーネントを活用し、順問題（forward problem）の完全なワークフローを検証するJupyter Notebookの要求仕様を定義します。Phase 2では`wave_2d_forward_demo.ipynb`が初期実装されましたが、本ノートブックでは実際の2D PINN訓練を実行し、FDTDデータとの比較検証を通じて実装の正当性を確認します。

**スコープ**: 完全な2D PINN訓練パイプラインの実証（データ読み込み、無次元化、モデル構築、訓練、R²評価、可視化）、Phase 2実装コンポーネントの統合検証、順問題ワークフローのベストプラクティス提示

**スコープ外**: 逆問題（パラメータ推定）、ハイパーパラメータチューニング自動化、リアルタイム推論最適化、本番環境デプロイ

## 要求事項

### 要求1: FDTD訓練データの準備と検証

**目的:** PINN研究者として、Phase 2で実装されたFDTDDataLoaderServiceとDimensionlessScalerServiceを用いて訓練データを準備し、データ品質を確認したい。

#### 受入基準

1. The Notebook shall `/PINN_data/`ディレクトリから複数の.npzファイル（最低2ファイル、最大12ファイル）をロードする
2. When データ読み込みが完了すると、the Notebook shall データセットサイズ（サンプル数、パラメータ組み合わせ数）を表示する
3. The Notebook shall DimensionlessScalerServiceを使用してFDTDデータを無次元化する
4. The Notebook shall 無次元化後の各変数（x̃, ỹ, t̃, T̃1, T̃3, Ũx, Ũy）の範囲をプロットまたは表形式で表示する
5. The Notebook shall 訓練データとバリデーションデータを80:20に分割し、各データセットのサイズを明示する
6. The Notebook shall データの空間-時間分布（例: t vs x散布図）を可視化してサンプリングの均一性を確認する
7. If データファイルが見つからない場合、then the Notebook shall FileNotFoundErrorまたは明確なエラーメッセージを表示して実行を停止する

### 要求2: 2D PINN モデルの構築と訓練

**目的:** PINN研究者として、Phase 2のPINNModelBuilder2DServiceとPDEDefinition2DServiceを用いてモデルを構築し、実際の訓練を実行して収束性を確認したい。

#### 受入基準

1. The Notebook shall YAMLまたはPython辞書形式で設定パラメータ（layer_sizes, activation, epochs, learning_rate, loss_weights）を定義する
2. The Notebook shall PINNModelBuilder2DServiceを使用して5D入力（x̃, ỹ, t̃, pitch_norm, depth_norm）、4D出力（T̃1, T̃3, Ũx, Ũy）のPINNモデルを構築する
3. The Notebook shall PDEDefinition2DServiceの無次元化PDE関数を使用してモデルに物理制約を適用する
4. The Notebook shall TrainingPipelineServiceまたは同等の訓練ループを実行する（最低1000 epochs、推奨5000 epochs）
5. The Notebook shall LossLoggingCallbackを使用して訓練中の個別損失項（L_data, L_pde, L_bc）とtotal lossを定期的（例: 100 epochsごと）に表示する
6. When 訓練中にlossが表示されると、the Notebook shall 各損失項の値を明示的に表示する（例: 「Epoch 1000 | L_data: 0.123, L_pde: 0.045, L_bc: 0.000, Total: 0.168」）
7. The Notebook shall 訓練履歴（epoch vs loss）をプロットし、L_data、L_pde、L_bc、Total lossの4系列を異なる色で可視化する
8. If 訓練中にNaN lossが発生した場合、then the Notebook shall DivergenceDetectionCallbackが検出した警告を表示し、学習率削減またはloss weight調整を推奨する
9. The Notebook shall GPU利用可否を確認し、GPU使用時は訓練時間を記録する

### 要求3: R²スコアによる定量評価

**目的:** PINN研究者として、Phase 2のR2ScoreCalculatorを用いてPINN予測精度をFDTD ground truthと比較し、各出力場の性能を定量評価したい。

#### 受入基準

1. The Notebook shall 訓練完了後にバリデーションデータ上でPINN予測を実行する
2. The Notebook shall R2ScoreCalculatorを使用して各出力場（T1, T3, Ux, Uy）のR²スコアを個別計算する
3. The Notebook shall R²スコアを表形式（例: pandas DataFrame）で表示し、フィールドごとの性能を比較可能にする
4. The Notebook shall R²スコアのbar chartまたはheatmapを生成して視覚的に性能を提示する
5. When R²スコアが計算されると、the Notebook shall 各フィールドの予測精度を解釈するコメント（例: 「Uxは高精度（R²=0.95）、T1は要改善（R²=0.82）」）を含める
6. If いずれかのフィールドのR²が0.9未満の場合、then the Notebook shall 改善策（loss weight調整、epochs増加、network拡大）を提案する

### 要求4: 波動場の2D空間分布可視化

**目的:** PINN研究者として、Phase 2のPlotGeneratorServiceを用いて2D波動場の空間分布を可視化し、FDTD ground truthとPINN予測を視覚的に比較したい。

#### 受入基準

1. The Notebook shall 複数の時刻スナップショット（最低3時刻、推奨5時刻）における波動場分布を選択する
2. The Notebook shall PlotGeneratorService.plot_time_snapshots()を使用してFDTDデータとPINN予測の2D heatmap比較プロットを生成する
3. The Notebook shall 各時刻スナップショットに対してUxまたはUy変位場の空間分布（x-y平面のheatmap）を表示する
4. The Notebook shall FDTDとPINNのheatmapを並列配置（side-by-side）し、視覚的差異を識別可能にする
5. When 可視化が生成されると、the Notebook shall 波動伝播パターン（例: 波面の位置、振幅分布）がFDTDと一致しているかを解説するコメントを含める
6. The Notebook shall 少なくとも1つの出力場（推奨: Ux変位場）に対して可視化を実施する

### 要求5: 誤差分布の空間解析

**目的:** PINN研究者として、PlotGeneratorService.plot_spatial_heatmap()を用いて予測誤差の空間分布を解析し、PINN学習の弱点領域を特定したい。

#### 受入基準

1. The Notebook shall 特定の時刻（例: 中間時刻t_mid）における絶対誤差|PINN - FDTD|を計算する
2. The Notebook shall PlotGeneratorService.plot_spatial_heatmap()を使用して誤差の2D空間分布をheatmapで可視化する
3. The Notebook shall heatmapのcolorbarに誤差スケール（物理単位: m for displacement, Pa for stress）を明示する
4. When 誤差分布が可視化されると、the Notebook shall 誤差が集中する領域（例: ドメイン境界付近、波源近傍）を特定し、その物理的意味を解説する
5. The Notebook shall 平均誤差、最大誤差、相対誤差（誤差/FDTD標準偏差）を数値で表示する
6. The Notebook shall 誤差分布から得られる知見（例: 境界条件不足、collocation points不足）を要約する

### 要求6: パラメトリック予測の検証（オプション）

**目的:** PINN研究者として、訓練時に使用していないパラメータ組み合わせ（内挿）でPINN予測を実行し、パラメトリック学習の汎化性能を検証したい。

#### 受入基準

1. Where 複数のFDTDパラメータ組み合わせで訓練された場合、the Notebook shall holdoutパラメータ（例: pitch=1.625mm, depth=0.15mm）を選択する
2. The Notebook shall holdoutパラメータに対してPINN予測を実行し、対応するFDTDデータ（存在する場合）と比較する
3. The Notebook shall holdoutデータに対するR²スコアを計算し、訓練データR²と比較する
4. When holdout性能が訓練性能より低い場合、the Notebook shall 過学習またはパラメータ範囲不足の可能性を指摘する
5. The Notebook shall パラメータ内挿予測の波動場可視化を1つ以上生成する

### 要求7: ノートブック構成と再現性

**目的:** PINN研究者として、ノートブックを他の研究者が再実行可能な形式で提供し、Phase 2実装の使用例として活用できるようにしたい。

#### 受入基準

1. The Notebook shall 各セクションに明確な日本語見出し（Markdown cell）を含める（例: 「## Step 1: データ読み込み」）
2. The Notebook shall 各主要ステップの目的と期待される出力を説明するコメントセルを含める
3. The Notebook shall import文を冒頭にまとめ、必要なパッケージ（pinn.data, pinn.models, pinn.training, pinn.validation）を明示する
4. The Notebook shall SeedManagerを使用してrandom seed（推奨: 42）を設定し、再現性を保証する
5. When Notebookが実行されると、the Notebook shall エラーなく全セルが完了し、すべての可視化が生成される
6. The Notebook shall 実装コードと完全に一致するAPI（クラス名、メソッド名、引数）を使用する
7. The Notebook shall 計算時間のかかるセル（訓練ループ）に対して推定実行時間（例: 「GPU: 約5分」）を注記する
8. The Notebook shall 最終セクションに「まとめ」を含め、達成されたR²スコア、可視化結果、実装の課題点を要約する

### 要求8: 実装整合性の確保

**目的:** PINN研究者として、Phase 2で発生したimport pathやメソッド名の不整合を回避し、ノートブックが常に最新の実装と同期されるようにしたい。

#### 受入基準

1. The Notebook shall Phase 2実装のServiceクラス（FDTDDataLoaderService, PINNModelBuilder2DService, TrainingPipelineService等）を直接importし、正確なパスを使用する
2. The Notebook shall Phase 2のconfig_loader.ConfigLoaderServiceでYAML設定を読み込む場合、実在する設定ファイルパス（例: `configs/pinn_2d_example.yaml`）を指定する
3. The Notebook shall Phase 2のメソッドシグネチャ（引数名、順序、デフォルト値）と完全一致するAPI呼び出しを行う
4. When Phase 2実装が更新された場合、the Notebook shall 対応する変更（例: メソッド名変更、引数追加）を同時反映する
5. The Notebook shall 非推奨API（deprecated methods）を使用せず、Phase 2の最新実装パターンに従う
6. If 実装とNotebookの不整合が検出された場合、then the Notebook shall エラーメッセージとともに正しいAPI使用例を提示する

## 非機能要求

### パフォーマンス
- ノートブック全体の実行時間はGPU上で30分以内に完了すること（5000 epochs訓練含む）
- データ読み込み（12ファイル）は30秒以内に完了すること
- 可視化生成（全プロット）は1分以内に完了すること

### 保守性
- Phase 2のディレクトリ構造（`/pinn/models/`, `/pinn/training/`, `/pinn/validation/`, `/pinn/data/`）からのimportパスを正確に維持すること
- コードセルとMarkdownセルの比率を1:1程度に保ち、可読性を最大化すること

### 互換性
- Python 3.11+、PyTorch 2.4.0、DeepXDE 1.15.0、CUDA 12.4との互換性を維持すること
- Phase 2実装コンポーネント（ConfigLoaderService、SeedManager、ErrorMetricsService等）をそのまま使用可能であること

### ドキュメント
- すべてのMarkdownセル（見出し、説明）は日本語で記述すること
- コード内のインラインコメントは英語で記述すること（Phase 2規約準拠）
- ノートブック冒頭に「概要」セクションを含め、本ノートブックの目的と前提条件（Phase 2実装完了）を明記すること
