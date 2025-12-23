# 実装タスク

## タスク概要

本実装計画は、Phase 2（pinn-2d-fdtd-integration）で実装された2D弾性波PINNコンポーネントを統合し、順問題の完全なワークフローを検証するJupyter Notebookを作成する。全タスクは既存Phase 2 Serviceを再利用するセル構成作業であり、新規コンポーネント開発は不要。

**実装スコープ**: Jupyter Notebook（.ipynb）ファイル1つ、9セルグループ（概要、Setup、データ準備、モデル構築、訓練、R²評価、2D可視化、誤差解析、まとめ）

**並列実行可能性**: 本機能はJupyter Notebook単一ファイルの順次セル構成のため、セル間に依存関係があり並列タスク実行は不可。全タスクは順次実行となる。

## 実装タスク

### 1. Notebookセットアップと概要セル作成

- [x] 1.1 Notebook初期化と概要Markdownセル作成
  - `/notebooks/`ディレクトリに`pinn_2d_forward_validation.ipynb`を新規作成
  - Cell 0（Markdown）: Notebookタイトル、概要、目的、前提条件、実行環境を日本語で記述
  - 概要セクションに「Phase 2実装完了が前提」「/PINN_data/に.npzファイル配置」「GPU環境推奨」を明記
  - _Requirements: 7.1, 7.2, 7.8_

- [x] 1.2 Import文とセットアップセル実装
  - Cell 1（Code）: 標準ライブラリ、Phase 2 Service、可視化ライブラリをimport
  - Phase 2 Serviceの正確なimport pathを使用（`from pinn.data.fdtd_loader import FDTDDataLoaderService`等）
  - `SeedManager.set_seed(42)`でrandom seed固定
  - `torch.cuda.is_available()`でGPU利用可否確認、デバイス名表示
  - matplotlib/seabornの可視化スタイル設定（`sns.set_style("whitegrid")`）
  - _Requirements: 7.3, 7.4, 2.9, 8.1_

### 2. FDTDデータ読み込みと前処理セル実装

- [x] 2.1 FDTDデータファイル読み込みセル実装
  - Cell 2（Code + Markdown説明）: `/PINN_data/`ディレクトリから.npzファイル検索・読み込み
  - データディレクトリ存在確認（`Path("/PINN_data").exists()`）、不在時はFileNotFoundErrorで停止
  - .npzファイル数確認（最低2ファイル）、不足時はValueErrorで停止
  - `FDTDDataLoaderService.load_multiple_files()`で複数ファイル結合
  - データセットサイズ（サンプル数、パラメータ組み合わせ数）を表示
  - _Requirements: 1.1, 1.2, 1.7, 8.2_

- [x] 2.2 無次元化と変数範囲表示セル実装
  - Cell 2（続き）: `CharacteristicScales.from_physics()`で特性スケール計算
  - `DimensionlessScalerService`でFDTDデータを無次元化
  - 無次元化後の各変数（x̃, ỹ, t̃, T̃1, T̃3, Ũx, Ũy）の範囲を表形式またはprint出力で表示
  - 各変数が[0, 1]近傍であることを確認（無次元化正当性検証）
  - _Requirements: 1.3, 1.4_

- [x] 2.3 Train/Val分割とデータ分布可視化セル実装
  - Cell 3（Code + Markdown説明）: `FDTDDataLoaderService.train_val_split(dataset, train_ratio=0.8, seed=42)`で分割
  - 訓練データとバリデーションデータのサイズを明示（print出力）
  - データの空間-時間分布をscatter plot可視化（例: t vs x散布図）
  - サンプリングの均一性を視覚的に確認（空間-時間ドメイン全体のカバレッジ）
  - _Requirements: 1.5, 1.6_

### 3. 2D PINNモデル構築セル実装

- [x] 3.1 設定パラメータ定義とモデル構築セル実装
  - Cell 4（Code + Markdown説明）: Python辞書形式で設定パラメータ定義（layer_sizes, activation, epochs, learning_rate, loss_weights）
  - 推奨設定を例示（`layer_sizes: [5, 64, 64, 64, 4]`, `activation: "tanh"`, `epochs: 5000`, `learning_rate: 0.001`）
  - `PINNModelBuilder2DService.build_model(config)`で5D入力、4D出力の2D PINNモデル構築
  - `PDEDefinition2DService.create_pde_function()`の無次元化PDE関数を使用してモデルに物理制約適用
  - モデル構築成功確認（`assert model is not None`）
  - _Requirements: 2.1, 2.2, 2.3, 8.3_

### 4. 訓練実行と損失監視セル実装

- [x] 4.1 訓練実行と損失ロギングセル実装
  - Cell 5（Code + Markdown説明）: `LossLoggingCallback`, `R2ValidationCallback`, `DivergenceDetectionCallback`をcallbacksリストに追加
  - `TrainingPipelineService.train(model, config, callbacks=[...])`で訓練ループ実行（最低1000 epochs、推奨5000 epochs）
  - 訓練開始・終了時刻を`time.time()`で記録、訓練時間を表示（GPU使用時）
  - LossLoggingCallbackが100 epochsごとに個別損失項（L_data, L_pde, L_bc）とtotal lossを表示
  - 各損失項の値を明示的に表示（例: "Epoch 1000 | L_data: 0.123, L_pde: 0.045, L_bc: 0.000, Total: 0.168"）
  - _Requirements: 2.4, 2.5, 2.6, 2.9_

- [x] 4.2 訓練履歴プロットとNaN検出セル実装
  - Cell 5（続き）: 訓練完了後、`PlotGeneratorService.plot_training_curves(history)`で損失履歴プロット
  - L_data（青点線）、L_pde（赤点線）、L_bc（緑点線）、Total loss（黒実線）の4系列を異なる色で可視化
  - DivergenceDetectionCallbackがNaN loss検出時の警告を表示（学習率削減またはloss weight調整を推奨）
  - 訓練収束性を視覚的に確認（loss減少トレンド）
  - _Requirements: 2.7, 2.8_

### 5. R²スコア評価セル実装

- [x] 5.1 R²スコア計算と表形式表示セル実装
  - Cell 6（Code + Markdown説明）: 訓練済みモデルで`model.predict(val_x)`を実行し、バリデーションデータ予測
  - `R2ScoreCalculator.compute_r2_multi_output(y_true, y_pred)`で各出力場（T1, T3, Ux, Uy）のR²スコアを個別計算
  - pandas.DataFrameでR²スコアを表形式表示（`pd.DataFrame([r2_scores]).T`で転置）
  - 各フィールドのR²値を明示的に表示
  - _Requirements: 3.1, 3.2, 3.3_

- [x] 5.2 R²スコアbar chart可視化と解釈コメント実装
  - Cell 6（続き）: matplotlib `ax.bar()`でR²スコアのbar chart生成
  - x軸: フィールド名（T1, T3, Ux, Uy）、y軸: R²値（0-1範囲）
  - 各フィールドの予測精度を解釈するコメント追加（例: "Uxは高精度（R²=0.95）、T1は要改善（R²=0.82）"）
  - R² < 0.9のフィールドに対して改善策を提案（loss weight調整、epochs増加、network拡大）
  - _Requirements: 3.4, 3.5, 3.6_

### 6. 2D波動場可視化セル実装

- [x] 6.1 時刻スナップショット可視化セル実装
  - Cell 7（Code + Markdown説明）: 時系列から3-5時刻を選択（例: t_unique[len//4], t_unique[len//2], t_unique[3*len//4]）
  - `PlotGeneratorService.plot_time_snapshots(x, y, t_list, fdtd_data, pinn_pred, output_field='Ux')`でFDTD vs PINN比較プロット
  - 各時刻スナップショットに対してUx変位場の空間分布（x-y平面heatmap）を表示
  - FDTDとPINNのheatmapを並列配置（side-by-side）し、視覚的差異を識別可能にする
  - 波動伝播パターン（波面位置、振幅分布）がFDTDと一致しているかを解説するMarkdownコメント追加
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

### 7. 誤差分布解析セル実装

- [x] 7.1 誤差分布heatmap可視化セル実装
  - Cell 8（Code + Markdown説明）: 特定時刻（例: 中間時刻t_mid）における絶対誤差|PINN - FDTD|を計算
  - `PlotGeneratorService.plot_spatial_heatmap(x, y, error, output_field)`で誤差の2D空間分布をheatmap可視化
  - heatmapのcolorbarに誤差スケール（物理単位: m for displacement, Pa for stress）を明示
  - 誤差が集中する領域（例: ドメイン境界付近、波源近傍）を特定し、その物理的意味を解説するMarkdownコメント追加
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 7.2 誤差統計表示と知見要約セル実装
  - Cell 8（続き）: 平均誤差、最大誤差、相対誤差（誤差/FDTD標準偏差）を数値で表示
  - 誤差分布から得られる知見を要約（例: "境界条件不足によりドメイン端で誤差増大"、"collocation points不足により波源近傍で誤差集中"）
  - 改善策を提案（境界条件強化、collocation points増加、network拡大等）
  - _Requirements: 5.5, 5.6_

### 8. まとめセルとNotebook完成

- [x] 8.1 まとめセル実装とNotebook検証
  - Cell 9（Markdown + Code）: 達成されたR²スコアの要約（各フィールドのR²値を再掲）
  - 可視化結果のまとめ（時刻スナップショット、誤差分布の主要知見）
  - 実装の課題点を記載（低R²フィールド、訓練時間、誤差集中領域等）
  - 次のステップ提案（逆問題、ハイパーパラメータチューニング、パラメトリック予測検証等）
  - Notebook全体の一貫性確認（全セル実行でエラーなし、すべての可視化生成）
  - _Requirements: 7.8, 7.5, 7.6_

- [x] 8.2 Notebook実行時間注記とAPI整合性確認
  - 各セルに推定実行時間を注記（例: Cell 5に"GPU: 約10-15分（5000 epochs）"を追記）
  - Phase 2実装APIとの完全一致を確認（import path、メソッドシグネチャ、引数順序）
  - 実装とNotebookの不整合検出時のエラーメッセージ明確化（正しいAPI使用例を提示）
  - Notebookのコード内インラインコメントは英語で記述（Phase 2規約準拠）
  - _Requirements: 7.7, 8.3, 8.4, 8.5, 8.6_

### 9. テストと検証

- [x] 9.1 Notebook実行テスト実装
  - `tests/test_notebook_execution.py`を強化し、nbconvertでNotebook全セル実行テスト
  - `ExecutePreprocessor`で各セル順次実行、エラー出力なしを確認
  - 最終セル実行後のkernel state検証（`r2_scores`変数存在、`trained_model`変数存在）
  - Integration testとして実装（@pytest.mark.integration, @pytest.mark.slow）
  - _Requirements: 7.5_
  - **完了**: 2テストメソッド追加（`test_notebook_executes_without_errors`, `test_notebook_final_kernel_state`）

- [x] 9.2 出力形式検証テスト実装
  - `tests/test_notebook_output_format.py`を作成
  - R²スコアDataFrame形状確認（4行×1列、index=['T1', 'T3', 'Ux', 'Uy']）
  - 損失履歴プロット生成確認（PlotGeneratorService.plot_training_curves()使用確認）
  - 時刻スナップショットheatmap生成確認（最低3時刻）
  - 誤差分布heatmap生成確認（colorbar表示）
  - _Requirements: 3.3, 3.4, 2.7, 4.1, 5.2_
  - **完了**: 14テスト実装、13個成功（1個は/PINN_data不在でスキップ）

- [ ] 9.3* E2Eパフォーマンステスト実装（オプション）
  - `tests/test_notebook_performance.py`を作成（デフォーカス可能）
  - データ読み込み時間測定（12ファイル: 30秒以内）
  - 可視化生成時間測定（全プロット: 1分以内）
  - R²スコア閾値確認（≥ 0.7達成、理想値0.9）
  - パフォーマンス要求未達時のwarning表示
  - _Requirements: 非機能要求（パフォーマンス）_

## タスク依存関係

本実装は単一Jupyter Notebookの順次セル構成のため、全タスクは厳密に順次実行となる。各タスクは前タスクの出力（セル実装）に依存する。

**実行順序**:
1. タスク1（Setup）→ 2（データ準備）→ 3（モデル構築）→ 4（訓練）→ 5（R²評価）→ 6（2D可視化）→ 7（誤差解析）→ 8（まとめ）→ 9（テスト）

**並列実行不可理由**: Jupyter Notebookの各セルが前セルのkernel stateに依存（例: Cell 3がCell 2の`dataset`変数を使用）

## 要求カバレッジ検証

全要求（要求1.1-1.7, 2.1-2.9, 3.1-3.6, 4.1-4.6, 5.1-5.6, 6.1-6.5, 7.1-7.8, 8.1-8.6, 非機能要求）が上記タスクでカバーされていることを確認済み。

**要求カバレッジマップ**:
- 要求1（FDTDデータ準備）: タスク2.1-2.3
- 要求2（モデル構築・訓練）: タスク3.1, 4.1-4.2
- 要求3（R²評価）: タスク5.1-5.2
- 要求4（波動場可視化）: タスク6.1
- 要求5（誤差解析）: タスク7.1-7.2
- 要求6（パラメトリック検証）: スコープ外（オプション要求、Phase 3で実装）
- 要求7（Notebook構成）: タスク1.1-1.2, 8.1-8.2
- 要求8（実装整合性）: タスク1.2, 2.1, 3.1, 8.2
- 非機能要求: タスク9.1-9.3

**意図的除外**: 要求6（パラメトリック予測検証）はオプション要求のため、Phase 3に延期。
