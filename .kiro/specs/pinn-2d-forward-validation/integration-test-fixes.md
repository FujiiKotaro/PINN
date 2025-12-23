# Integration Test Fixes - Task 9.1

## 問題の概要

Integration test (`test_notebook_executes_without_errors`) が以下の2つのエラーで失敗していました:

### エラー 1: ModuleNotFoundError: No module named 'pinn'
- **原因**: Notebook実行時にPYTHONPATHが設定されていないため、`pinn`モジュールがimportできない
- **影響範囲**: Notebookの最初のセルでインポートが失敗

### エラー 2: Notebook JSON is invalid
- **原因**: Markdown cellに`execution_count`と`outputs`プロパティが存在（code cellにのみ許可されるプロパティ）
- **影響範囲**: nbformatバリデーション時にエラー

### エラー 3: FileNotFoundError: PINN_data not found
- **原因**: Notebookがtmpディレクトリで実行され、`Path.cwd().parent / "PINN_data"`が誤った場所を指す
- **影響範囲**: データ読み込みセルで失敗

## 実施した修正

### 修正 1: PYTHONPATH設定の追加
**ファイル**: `tests/test_notebook_execution.py`

```python
# Set PYTHONPATH to include project root
old_pythonpath = os.environ.get("PYTHONPATH", "")
project_root = str(Path(__file__).parent.parent)
os.environ["PYTHONPATH"] = f"{project_root}:{old_pythonpath}"

# ... (test execution) ...

finally:
    # Restore PYTHONPATH
    os.environ["PYTHONPATH"] = old_pythonpath
```

**変更内容**:
- Notebook実行前にPYTHONPATHにproject rootを追加
- テスト終了後に元のPYTHONPATHを復元
- `test_notebook_executes_without_errors`と`test_notebook_final_kernel_state`の両方に適用

### 修正 2: Notebook metadataのクリーンアップ
**ファイル**: `scripts/clean_notebook_metadata.py` (新規作成)

```python
# Remove invalid properties from markdown cells
if cell['cell_type'] == 'markdown':
    if 'execution_count' in cell:
        del cell['execution_count']
    if 'outputs' in cell:
        del cell['outputs']
```

**実行結果**:
```
Cleaned cell eomwto9fe6p: removed execution_count, outputs
✓ Cleaned 1 markdown cell(s)
```

### 修正 3: Notebook実行ディレクトリの修正
**ファイル**: `tests/test_notebook_execution.py`

```python
# Execute notebook from notebooks directory so Path.cwd() is correct
notebooks_dir = notebook_path.parent

# Execute all cells (use notebooks directory as working directory)
nb_executed, resources = ep.preprocess(nb, {"metadata": {"path": str(notebooks_dir)}})
```

**変更内容**:
- `tmp_path`から`notebooks_dir`に変更
- Notebookが`notebooks/`ディレクトリで実行されるため、`Path.cwd().parent`が正しくproject rootを指す
- PINN_dataディレクトリが正しく見つかるようになる

## 検証方法

### 非統合テストの実行 (PINN_dataなしでも実行可能)
```bash
poetry run pytest tests/test_notebook_execution.py::TestNotebookStructure -v
poetry run pytest tests/test_notebook_execution.py::TestNotebookImports -v
poetry run pytest tests/test_notebook_execution.py::TestTask2DataPreparation -v
```

### 統合テストの実行 (PINN_dataが必要)
```bash
PYTHONPATH=/home/manat/project2:$PYTHONPATH poetry run pytest tests/test_notebook_execution.py::TestNotebookExecution::test_notebook_executes_without_errors -m integration -v
```

**注意**: 統合テストは約10-30分かかります（GPU使用時は約10-15分、CPU使用時は約60-90分）

## 期待される結果

### 成功時の出力
```
tests/test_notebook_execution.py::TestNotebookExecution::test_notebook_executes_without_errors PASSED [100%]
```

### エラー時のトラブルシューティング

#### 1. PINN_dataディレクトリが見つからない
```
pytest.skip: Skipping integration test: /home/manat/project2/PINN_data not found
```
**解決策**: PINN_dataディレクトリを作成し、.npzファイルを配置

#### 2. GPU not available warning
```
WARNING: GPU not available, training will be slow on CPU
```
**影響**: テストは実行されるが、訓練時間が大幅に増加（60-90分）

#### 3. Training timeout
```
TimeoutError: Cell execution timed out after 1800 seconds
```
**解決策**: `timeout=1800`を`timeout=5400`に増加（90分に延長）

## 関連ファイル

### 修正されたファイル
- `/home/manat/project2/tests/test_notebook_execution.py` (PYTHONPATH設定と実行ディレクトリ修正)
- `/home/manat/project2/notebooks/pinn_2d_forward_validation.ipynb` (metadata cleaned)

### 新規作成されたスクリプト
- `/home/manat/project2/scripts/clean_notebook_metadata.py` (metadata cleanup script)

### 影響を受けるテスト
- `TestNotebookExecution::test_notebook_executes_without_errors` (Task 9.1)
- `TestNotebookExecution::test_notebook_final_kernel_state` (Task 9.1)

## 次のステップ

1. **統合テストの実行**: GPU環境で統合テストを実行し、全セルが正常に実行されることを確認
2. **パフォーマンス測定**: 訓練時間が30分以内（要求2.9）を満たすか測定
3. **R²スコア検証**: R² ≥ 0.9（要求3.6）を達成しているか確認
4. **出力フォーマットテスト**: Task 9.2の出力フォーマットテストを実行
5. **Gap analysis更新**: integration test結果に基づきgap-analysis.mdを更新

## 参考情報

### Task 9.1 Requirements
- **Requirement 7.5**: Notebook実行テスト実装
- **Requirement 2.9**: GPU使用時の訓練時間30分以内
- **Requirement 3.6**: R² ≥ 0.9の達成

### 関連するSpec文書
- `.kiro/specs/pinn-2d-forward-validation/tasks.md` (Task 9.1, 9.2, 9.3)
- `.kiro/specs/pinn-2d-forward-validation/gap-analysis.md` (実装状況)
- `.kiro/specs/pinn-2d-forward-validation/requirements.md` (要求仕様)
