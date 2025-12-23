# 調査・設計決定ログ

## サマリー
- **Feature**: `pinn-2d-forward-validation`
- **Discovery Scope**: Extension（Phase 2実装コンポーネントを統合した検証ノートブック作成）
- **Key Findings**:
  - Phase 2実装（pinn-2d-fdtd-integration）のすべての必要コンポーネントが利用可能
  - Jupyter Notebook形式での実装により、段階的実行と可視化が可能
  - 既存のLossLoggingCallback、R2ValidationCallback、PlotGeneratorServiceが要求仕様を完全にカバー

## Research Log

### Phase 2実装コンポーネントの可用性調査

- **Context**: 要求仕様で参照されているPhase 2コンポーネントが実装済みであることを確認
- **Sources Consulted**:
  - `/home/manat/project2/pinn/` ディレクトリ構造
  - `.kiro/specs/pinn-2d-fdtd-integration/design.md`
  - Phase 2実装ファイル（callbacks.py, r2_score.py, plot_generator.py等）
- **Findings**:
  - **Data Layer**: `FDTDDataLoaderService`, `DimensionlessScalerService`, `ParameterNormalizer`が実装済み
  - **Model Layer**: `PINNModelBuilder2DService`, `PDEDefinition2DService`が実装済み
  - **Training Layer**: `TrainingPipelineService`, `LossLoggingCallback`, `R2ValidationCallback`, `DivergenceDetectionCallback`が実装済み
  - **Validation Layer**: `R2ScoreCalculator`, `PlotGeneratorService`（`plot_time_snapshots`, `plot_spatial_heatmap`メソッド含む）が実装済み
  - **Utils**: `ConfigLoaderService`, `SeedManager`, `ExperimentManager`, `MetadataLogger`が実装済み
- **Implications**:
  - 新規コンポーネント開発は不要、既存APIの統合のみ実施
  - Notebookセル構成は既存Serviceメソッド呼び出しの順序付けとなる

### Jupyter Notebook vs Python Script

- **Context**: 検証ワークフローの実装形式を決定
- **Alternatives Considered**:
  1. **Jupyter Notebook (.ipynb)**: セル単位の実行、インライン可視化、Markdown説明挿入可能
  2. **Python Script (.py)**: 単一実行フロー、バージョン管理容易、自動化向き
- **Findings**:
  - 要求仕様7.1で明確に"各セクションに明確な日本語見出し（Markdown cell）"を要求
  - 要求仕様7.2で"各主要ステップの目的と期待される出力を説明するコメントセル"を要求
  - Phase 1, Phase 2ともに`/notebooks/wave_1d_demo.ipynb`, `/notebooks/wave_2d_forward_demo.ipynb`を使用
- **Selected Approach**: Jupyter Notebook形式
- **Rationale**:
  - 要求仕様と既存プロジェクトパターンに整合
  - PINN研究者がステップごとに実行結果を確認しながら学習可能
  - 可視化がインラインで表示され、研究ワークフローに最適
- **Trade-offs**:
  - Benefits: 対話的実行、教育的価値、段階的デバッグ容易
  - Compromises: 差分管理がPython scriptより複雑、CI/CD統合にはnbconvertが必要

### Loss項表示の詳細化（L_data, L_pde, L_bc）

- **Context**: ユーザーフィードバックで各損失項の明示的表示を要求
- **Sources Consulted**:
  - `pinn/training/callbacks.py:10-68` (LossLoggingCallback)
  - DeepXDE model.train_state.loss_train構造
- **Findings**:
  - `LossLoggingCallback`がhistory辞書に`L_data`, `L_pde`, `L_bc`, `total_loss`を記録
  - DeepXDE `model.train_state.loss_train`は`[L_data, L_pde, L_bc]`の3要素ndarray
  - 要求2.6で"Epoch 1000 | L_data: 0.123, L_pde: 0.045, L_bc: 0.000, Total: 0.168"形式を指定
- **Implications**:
  - Notebookで`LossLoggingCallback`のhistory辞書を直接参照
  - 訓練後にmatplotlibで4系列（L_data, L_pde, L_bc, Total）をプロット
  - 各損失項の値を数値表示するセルを追加

### R²スコア可視化形式

- **Context**: 要求3.3, 3.4でR²スコアの表形式およびbar chart表示を要求
- **Sources Consulted**:
  - `pinn/validation/r2_score.py` (R2ScoreCalculator)
  - pandas DataFrame APIドキュメント
  - matplotlib bar chart例
- **Findings**:
  - `R2ScoreCalculator.compute_r2_multi_output()`が`{'T1': r2, 'T3': r2, 'Ux': r2, 'Uy': r2}`の辞書を返却
  - pandas.DataFrameで表形式変換が容易
  - matplotlib `ax.bar()`でbar chart生成可能
- **Selected Approach**:
  1. R²スコア辞書をpandas DataFrameに変換（`pd.DataFrame([r2_scores])`）
  2. `DataFrame.T`で転置し、フィールド名を行indexに
  3. matplotlib bar chartで可視化（x軸: フィールド名、y軸: R²値）
- **Rationale**:
  - pandasは科学計算環境で標準的、Phase 2依存関係に既存
  - bar chartはR²比較に視覚的に適切（0-1範囲の比較）
- **Trade-offs**:
  - heatmap代替案も要求仕様に記載されているが、4フィールドのみのためbar chartが効率的

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Sequential Notebook Cells | 各要求をNotebookセルの順序で実装、セル間の実行順序に依存 | 実行フロー明確、デバッグ容易、教育的 | セル実行順序違反でエラー、再現性がkernel restartに依存 | Jupyter標準パターン、Phase 1/2の既存notebookと整合 |
| Modular Functions in Notebook | 各ステップをヘルパー関数として定義し、mainセルから呼び出し | 再利用可能、テスト容易、コード重複削減 | Notebook利点（段階的実行）が薄れる、複雑化 | Python scriptに近づくため不採用 |
| Hybrid: Cells + Utils Module | 主要ロジックを`/pinn/utils/notebook_helpers.py`に分離、Notebookは呼び出しのみ | コード整理、バージョン管理容易 | Notebookの自己完結性が失われる、Phase 2コンポーネントで十分 | 既存Serviceで網羅されているため不要 |

**Selected**: Sequential Notebook Cells (Jupyter標準パターン)

## Design Decisions

### Decision: Notebookセル構成の順序設計

- **Context**: 要求1-8を満たすNotebookセル構成を決定
- **Alternatives Considered**:
  1. 要求番号順（要求1→要求2→...→要求8）
  2. ワークフロー時系列順（データ準備→訓練→評価→可視化）
  3. ハイブリッド（時系列主軸、要求カバレッジを副次的に確保）
- **Selected Approach**: ワークフロー時系列順
- **Rationale**:
  - PINN研究者の典型的なワークフロー（データロード→モデル構築→訓練→検証→可視化）に整合
  - 要求7.1-7.2で"各セクションに明確な日本語見出し"を要求、時系列見出しが直感的
  - 要求1-6が自然な時系列に整列（要求1: データ準備、要求2: 訓練、要求3-5: 評価・可視化、要求6: オプション検証）
- **Trade-offs**:
  - Benefits: 実行フロー直感的、研究者の学習曲線フラット、エラー時のデバッグ容易
  - Compromises: 要求番号と見出し順序が一致しない（要求トレーサビリティで対応）
- **Follow-up**: セル見出しに要求番号を併記（例: "## Step 1: データ読み込み（要求1）"）

### Decision: 損失項プロットの系列数

- **Context**: 要求2.7で"L_data、L_pde、L_bc、Total lossの4系列を異なる色で可視化"を明示
- **Selected Approach**: 4系列プロット（L_data: 青点線、L_pde: 赤点線、L_bc: 緑点線、Total: 黒実線）
- **Rationale**:
  - 要求仕様に明示的指定あり
  - Phase 2の`PlotGeneratorService.plot_training_curves()`が既にこのパターンを実装
  - 各損失項の寄与度を視覚的に比較可能
- **Trade-offs**:
  - Benefits: 訓練診断に有用、PDE loss vs data lossのバランス確認容易
  - Compromises: プロット複雑化（4系列）、色盲対応は線種で補完
- **Follow-up**: 実装時にPlotGeneratorServiceの既存メソッドをそのまま使用

### Decision: エラーハンドリング戦略（データファイル不在時）

- **Context**: ユーザーフィードバックで「データファイルが見つからないときはエラーを返すだけでよい」
- **Alternatives Considered**:
  1. 合成データ生成フォールバック（当初の要求1.7）
  2. FileNotFoundErrorを発生させ実行停止（ユーザー要求）
  3. 警告表示後に一部機能のみ実行
- **Selected Approach**: FileNotFoundError発生、実行停止
- **Rationale**:
  - ユーザー明示的要求
  - 実データでの検証がNotebookの目的、合成データは本質的でない
  - エラーメッセージで`/PINN_data/`のデータ配置方法を案内
- **Trade-offs**:
  - Benefits: データ準備の明確化、不完全実行の回避
  - Compromises: デモ実行時にデータ準備ステップが必須
- **Follow-up**: セル実行前の前提条件チェック（`assert data_dir.exists()`）を追加

## Risks & Mitigations

- **Risk 1**: Phase 2実装APIの変更によるNotebook破損 — **Mitigation**: 要求8でAPI整合性チェックを明示、Phase 2更新時にNotebookも同時更新
- **Risk 2**: GPU OOMエラー（5000 epochs訓練時） — **Mitigation**: 要求2.1で設定パラメータをNotebook冒頭で定義、epochs/batch_size調整可能に
- **Risk 3**: R²スコアが低い場合のユーザー混乱 — **Mitigation**: 要求3.6で改善策を自動提案、要求7.8で実装の課題点を要約セクションに記載
- **Risk 4**: Notebook実行時間が30分超過 — **Mitigation**: 要求2.1で最低1000 epochs（推奨5000 epochs）とし、時間制約時は小epochs数で実行可能

## References

- [DeepXDE Documentation](https://deepxde.readthedocs.io/) — PINN framework API reference
- [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/) — Notebook構成ガイドライン
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html) — 可視化パターン例
- Phase 2 Design Document: `.kiro/specs/pinn-2d-fdtd-integration/design.md` — 再利用コンポーネント仕様
