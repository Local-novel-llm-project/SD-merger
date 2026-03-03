# SD-Merger TODO

> **凡例**: 各タスクにはコードベース調査に基づく **[考察]** を付記。
> 既存コードの具体的な課題・実装方針のヒントを記載している。

---

## 🔴 優先度: 高

### 1. マージコア機能

#### 1.1 レシピ機能 — ヒストリからレシピとして再利用可能にする

- [x] ヒストリエントリから YAML 設定を再生成する API
  - [x] `history.py` の `save_history(entry)` で保存される JSON に、元の YAML 設定全体を含める
  - [x] `entry` → YAML 逆変換関数 `history_to_yaml(entry) -> str` の実装
- [x] UI 上でワンクリック再実行ボタン
  - [x] `ui/components/history.py` にボタン追加、選択したエントリから YAML 再生成→ `main()` 呼出
- [x] レシピのインポート / エクスポート機能
  - [x] YAML ファイルとしてダウンロード / アップロード
  - [x] レシピにバージョン情報とメタデータ（使用モデル名、日時、ストラテジー）を付加

> **[考察]** 現在 `history.py` は 30 行の最小実装。`save_history(entry)` は辞書を丸ごと JSON に追記するだけで、
> YAML 設定の構造とは疎結合。レシピ化するには `main.py` の `config` 辞書を丸ごとヒストリに保存し、
> そこから YAML を再構築する「往復変換」が核心になる。
> また最大 100 件のハードリミットがあるため、レシピは別ストレージ（`recipes/` ディレクトリ）に保存する設計が望ましい。

#### 1.2 キューイングリスト — マージタスクのキュー管理

- [x] キューへの追加 / 削除 / 並び替え UI
  - [x] リストコンポーネント（Gradio `gr.Dataframe` 使用）
  - [x] 各エントリにステータス表示（pending / running / completed / error）
- [x] 逐次実行エンジン（進捗表示付き）
  - [x] `threading` ベースのワーカー
  - [x] 各タスクの状態を UI で更新可能（Refresh / Pause / Resume）
  - [x] キャンセル / 一時停止機能
- [x] キュー状態の永続化（アプリ再起動後も復元）
  - [x] JSON ファイル (`queue.json`) による保存
  - [x] アプリ起動時に未完了タスクの復元・ワーカー起動ロジック

> **[考察]** タスクは非同期で `queue_manager.py` によって動作するように改善されました。Gradio での中断のない状態遷移を実現し、永続化しています。

---

### 2. マージ戦略の拡張

#### 2.1 毒薬マージ — LoRA で過学習→薄くマージを繰り返す手法

- [ ] LoRA 適用 → マージ → 繰り返しのパイプライン構築
  - [ ] パイプライン定義（反復ステップ、減衰率のスケジュール）
  - [ ] 既存の `lora_ops` 拡張の `apply / merge` 機能をループ内で呼び出す統合
  - [ ] 中間モデルの一時保存（ディスク or メモリ、VRAM とのバランス）
- [ ] 反復回数・減衰率のパラメータ化
  - [ ] 減衰カーブの選択（線形, 指数, コサインアニーリング）
  - [ ] 反復ごとのパラメータオーバーライド設定
- [ ] XY Plot 風の比較画像グリッド生成（X軸: 反復回数, Y軸: 減衰率 等）
  - [ ] 既存の `xyz_plot.py` のグリッド生成ロジックを共通化して再利用
- [ ] 各反復ステップのサンプル画像を並べて変化を可視化

> **[考察]** 毒薬マージは複数ステップの複合処理。現在の `main.py` は 1 回のマージが完了すると関数終了するため、
> ループ制御レイヤーが必要。`extensions/lora_ops/` には LoRA の抽出・マージ機能はあるが、
> 「LoRA 適用 → モデル保存 → 再マージ」のイテレーションを制御するオーケストレータがない。
> 新規モジュール `module/pipeline.py` を作成し、パイプライン定義と実行を分離する設計を推奨。

#### 2.2 針山マージ — Base と Alpha の微調整による精密マージ

- [ ] Alpha 値のブロック別グラデーション設定
  - [ ] ブロックごとの weight を連続関数（線形, シグモイド, カスタムカーブ）で自動生成
  - [ ] IN/MID/OUT 各セクション独立のカーブ設定
- [ ] プレビュー付きのインタラクティブ調整 UI
  - [ ] Alpha カーブの可視化グラフ（Plotly or matplotlib）
  - [ ] スライダー操作で即座にグラフ更新
  - [ ] 調整結果を MBW 重み文字列として出力（既存の `mbw_each` にそのまま渡せる形）

> **[考察]** MBW Each にはブロック別の重みを手入力するが「直感的でない」。
> 針山マージは「カーブを視覚的に調整→26 値の MBW 配列を自動生成」するワークフロー。
> `ui/components/mbw_each.py` の既存 UI を拡張し、プリセットカーブ + カスタムカーブエディタを追加する。

#### 2.3 A/B テストマージ — モデル入替比較

- [ ] XY Plot 風の比較画像グリッド生成（A→B vs B→A を同一条件で並列表示）
  - [ ] 既存 `xyz_plot.py` の `run_xy` 関数を汎用化してパラメータ軸を拡張
- [ ] 同一 Seed・同一プロンプトでの Side-by-Side 比較ビュー
  - [ ] `generation.py` で `seed` の固定が既に実装済み → これを比較 UI と連携
  - [ ] 左右並置、スライダーによるフェード比較
- [ ] メトリクス（CLIP Score, FID 等）による定量比較
  - [ ] CLIP Score: `transformers` の CLIP モデルで計算
  - [ ] FID: `torchmetrics` or `clean-fid` ライブラリで計算
  - [ ] IS (Inception Score): オプション対応
  - [ ] 比較結果をレーダーチャートで可視化

> **[考察]** XY Plot (`xyz_plot.py`) は既に velocity / strategy / CFG / Steps の軸を持つが、
> 「モデル A/B の入れ替え」自体を軸に取る機能がない。`x_type` の選択肢に `Model Order` を追加すれば拡張可能。
> CLIP Score は `transformers` の `CLIPProcessor` + `CLIPModel` で画像-テキスト類似度を算出できるが、
> 大量画像に対しては GPU メモリとの兼ね合いが必要。

---

## 🟡 優先度: 中

### 3. 自動化・最適化

#### 3.1 BW/Each の自動化 — bayesian-merger 連携

> **リファレンス実装**: `refrence/sd-webui-bayesian-merger/` に実際のコードを追加済み。
> 以下のタスクはリファレンスの設計を SD-merger アーキテクチャに移植する方針。

- [ ] **最適化エンジンの移植** (`module/bayesian_optimizer/`)
  - [ ] 抽象基底クラス `Optimiser` の移植
    - リファレンスの `optimiser.py` を参考に、最適化ループ（探索→活用）を実装
    - `sd_target_function()`: マージ → 画像生成 → スコアリング → 平均スコア返却
  - [ ] 3 種のオプティマイザ実装
    - `BayesOptimiser`: `bayesian-optimization` ライブラリ（Gaussian Process）
      - Latin Hypercube Sampling (LHS) による初期探索オプション
      - `SequentialDomainReductionTransformer` による探索空間の逐次縮小
    - `TPEOptimiser`: `hyperopt` ライブラリ（Tree-structured Parzen Estimator）
    - `ATPEOptimiser`: `hyperopt.atpe`（Adaptive TPE）
  - [ ] UI からのオプティマイザ選択ドロップダウン
- [ ] **パラメータ空間管理** (`module/bayesian_optimizer/bounds.py`)
  - [ ] 探索空間の定義（各ブロック重み 0.0〜1.0 + `base_alpha`）
    - リファレンスの `Bounds` クラスを参考に、26 パラメータ（IN12 + MID1 + OUT12 + base_alpha）の空間定義
  - [ ] **Freeze 機能**: 特定ブロックの重みを固定値に凍結して探索対象外にする
  - [ ] **Group 機能**: 複数ブロックをグループ化し、同一の重みとして探索する（探索次元の削減）
  - [ ] **Custom Range 機能**: ブロックごとに探索範囲（上下限）をカスタマイズ
  - [ ] UI 上でのパラメータ空間設定（freeze/group/range の視覚的な設定）
- [ ] **スコアリングシステム** (`module/bayesian_optimizer/scorer.py`)
  - [ ] CLIP Aesthetic Score（`ViT-L/14` + `AestheticPredictor` ニューラルネット）
    - リファレンスの `laion` / `chad` スコアラーを参考に実装
    - スコアラーモデルの自動ダウンロード機能
  - [ ] Manual Scoring（画像を表示してユーザーが手動スコア入力）
  - [ ] ペイロードごとの `score_weight`（重み付きスコア平均）
  - [ ] **将来拡張**: FID / IS / ユーザー定義 Python 関数
- [ ] **目的関数のコールバック設計**
  - [ ] `sd_target_function()` の実装
    - SD-merger の `main()` を呼び出してマージ → `generation.py` で画像生成 → スコアリング
    - リファレンスは WebUI API (`requests.post`) 依存だが、SD-merger では直接 Python API を呼び出す
  - [ ] 複数プロンプト（ペイロード）での一括生成・スコアリング
    - リファレンスの `Prompter` を参考に、YAML ペイロードテンプレート管理
  - [ ] イテレーション情報のリアルタイム表示（warmup / optimisation フェーズ表示）
- [ ] **結果の可視化・保存**
  - [ ] 収束プロット（スコア推移グラフ）— リファレンスの `artist.py` `convergence_plot()` 参考
  - [ ] 最適 UNet 重み分布の可視化 — リファレンスの `draw_unet()` 参考
  - [ ] ベストパラメータの自動保存（`best.log` + レシピ形式への変換）
  - [ ] 最適パラメータでの最終マージモデル保存オプション

> **[考察]** リファレンスの `sd-webui-bayesian-merger` はアーキテクチャ的に以下の特徴がある:
>
> 1. **WebUI API 依存**: `merger.py` は `requests.post(url + "/bbwm/merge-models")` で sd-webui にマージを委譲。
>    SD-merger では直接 `sd_mecha.merge()` を呼び出す形に書き換える必要がある。
> 2. **Hydra 設定管理**: `omegaconf.DictConfig` ベース。SD-merger の YAML 設定体系に合わせて変換が必要。
> 3. **3 種のオプティマイザ**: Bayes（`bayesian-optimization`）、TPE/ATPE（`hyperopt`）が実装済み。
>    いずれも `optimise()` + `postprocess()` の統一インターフェースを持つ。
> 4. **パラメータ空間の柔軟性**: `Bounds` クラスが freeze / group / custom_range を組み合わせた
>    高度なパラメータ空間管理を提供。これは MBW Each の手動調整を自動化する上で重要。
> 5. **依存ライブラリ**: `bayesian-optimization`, `hyperopt`, `scipy` (LHS), `clip`, `safetensors` が必要。
>    `requirements.txt` への追加が必要。

#### 3.2 SuperAutoMerger 連携 — 外部自動マージツールとの統合

- [ ] パラメータ空間の自動探索
  - [ ] Optuna ベースのハイパーパラメータ最適化
  - [ ] 探索結果の可視化（Optuna Dashboard 連携 or 独自プロット）
- [ ] 探索結果のレシピへのフィードバック
  - [ ] 最良パラメータをレシピ形式（1.1 のレシピ機能）で保存

---

### 4. ストラテジー選択の統一（ベース機能追加）

> **現状の問題**: メインのマージ機能では **Strategy**（計算戦略）と **Target Strategy**（ターゲット計算戦略）を
> 自由に選択できるが、**MBW Each / Multi Merge / LoRA Ops** にはその選択肢が存在しない。

#### 4.1 MBW Each にストラテジー / ターゲットストラテジー選択を追加

- [ ] Strategy ドロップダウンの追加
- [ ] Target Strategy ドロップダウンの追加
- [ ] 既存のブロック重み設定との連携確認
  - [ ] `mbw_each` 拡張の `pre_config_hook` でストラテジー上書きが正しく伝播するか検証

> **[考察]** `ui/components/mbw_each.py` では `strategy` を `"mbw_each"` にハードコードしている (L73)。
> ここにドロップダウンを追加し、`config["models"][0]["strategy"]` を動的に設定する必要がある。
> ただし `mbw_each` 拡張の `pre_config_hook` がストラテジーを上書きする可能性があるため、
> フック適用順序のテストが必要。

#### 4.2 リストマージ（Multi Merge）にストラテジー / ターゲットストラテジー選択を追加

- [ ] バッチコマンド構文に `strategy` / `target_strategy` パラメータ追加
  - [ ] コマンドパーサー `parse_multi_merge_command()` に新変数を追加
- [ ] UI 上での選択肢追加（デフォルト値、行ごとのオーバーライド）

> **[考察]** `multi_merge.py` の `parse_multi_merge_command()` は `key=value` のカンマ区切りを解析。
> `S=mix, TS=addition` のような構文を追加すれば自然に拡張できる。
> ただし既に `base_alpha` パラメータが `strategy` を `"mix"` にハードコードしており (L63)、
> 明示的な `S=` パラメータとの競合ロジックを設計する必要がある。

#### 4.3 LoRA 操作にストラテジー / ターゲットストラテジー選択を追加

- [ ] LoRA マージ時の計算戦略選択
- [ ] LoRA 抽出時のターゲット戦略選択
- [ ] `lora_ops` 拡張の内部ロジックとの整合性確認

> **[考察]** `ui/components/lora_ops.py` は LoRA 抽出・マージに専用の Config を組み立てて `main()` に渡す。
> ここにストラテジー選択を追加するのは直接的だが、LoRA 操作のセマンティクス上
> すべてのストラテジーが意味を持つわけではない（例: `replace` は LoRA 抽出に不適）。
> 有効なストラテジーのフィルタリングが必要。

---

### 5. マージ後処理

#### 5.1 マージ後の自動生成 — マージ完了後にサンプル画像生成

- [ ] 生成設定のプリセット管理
  - [ ] 現在の `presets.py` を拡張し、マージ後生成用のプリセットカテゴリを追加
  - [ ] プロンプト / ネガティブ / 解像度 / ステップ数 / CFG のセット保存
- [ ] 複数プロンプトでの一括生成
  - [ ] プロンプトリスト（テキストファイル or UI 入力）からバッチ生成
  - [ ] 生成結果のサムネイルギャラリー表示
- [ ] 生成結果をヒストリに自動紐付け
  - [ ] `post_merge_hook` で画像生成を自動的にトリガー
  - [ ] ヒストリの `entry` に生成画像パスを格納

> **[考察]** `post_merge_hook` は `extension_manager.py` に既に仕組みがある。
> マージ完了後に自動生成する拡張機能を `extensions/auto_generate/` として作成すれば、
> 既存アーキテクチャの範囲内で実現可能。`generation.py` の `generate_image()` は
> 毎回パイプラインをロードするため、複数プロンプト生成時にはパイプラインの再利用が必要。

#### 5.2 モデルキャッシュ — ロード済みモデルのメモリキャッシュ

- [ ] LRU キャッシュによる VRAM 管理
  - [ ] `functools.lru_cache` or カスタム LRU（VRAM 容量ベースの eviction）
  - [ ] キャッシュヒット率のモニタリング
- [ ] キャッシュサイズ上限の設定
  - [ ] 設定ファイル or UI でバイト単位の上限指定
  - [ ] VRAM / RAM の自動検出と推奨値の提示
- [ ] モデルリロードボタン（UI）
  - [ ] キャッシュの手動クリア機能
  - [ ] 現在のキャッシュ状態（モデル名、サイズ）の表示

> **[考察]** `generation.py` の `generate_image()` は毎回 `from_single_file()` でモデルをロードし、
> 関数終了時に `del pipe` でメモリ解放している。これは毎回 数GB のロードが発生する。
> モジュールレベルの辞書 `_MODEL_CACHE: Dict[str, StableDiffusionPipeline]` を用意し、
> パスをキーにキャッシュすれば大幅な高速化が見込める。
> ただし VRAM 管理が重要で、SD1.5 (~2GB) と SDXL (~6.5GB) で必要量が大きく異なる。

---

## 🟢 優先度: 低 / 改善

### 6. UI/UX 改善

- [ ] マージ進捗のリアルタイムプログレスバー
  - [ ] `sd_mecha` のコールバック or ログパースによる進捗取得
  - [ ] Gradio `gr.Progress` コンポーネントとの連携
- [ ] XYZ Plot の Z 軸対応（3 次元グリッド生成）
  - [ ] Z 軸パラメータ追加（タブ切替 or アニメーション GIF で 3 次元表現）
  - [ ] 生成画像数の爆発に対するガードレール（上限設定、推定時間の事前表示）
- [ ] モデル分析の可視化強化
  - [ ] ヒートマップ: 全ブロック × 全メトリクスの2D可視化
  - [ ] レーダーチャート: IN/MID/OUT の特性バランスを直感的に把握
  - [ ] **SDXL 対応**: 現在 `analysis.py` は SD1.5 の `input_blocks/output_blocks` キー構造のみ対応。SDXL の `conditioner.embedders.1` 等の SDXL 固有キーへの対応が必要
- [ ] ダークモード / テーマ切り替え
  - [ ] Gradio の `theme` パラメータで対応（`gr.themes.Soft()` 等）
- [ ] キーボードショートカット対応
  - [ ] Gradio の JavaScript カスタマイズで実装
- [ ] **モデル一覧の動的リフレッシュ**
  - [ ] 現在 `get_model_list()` はアプリ起動時に1回だけ呼ばれる → ファイル追加/削除時に自動更新
  - [ ] Gradio Dropdown の `choices` を動的更新

> **[考察・新規追加]** `ui/utils.py` の `get_model_list()` は静的なリストを返す。
> モデルフォルダの監視（`watchdog` ライブラリ）or ボタン押下でリフレッシュする仕組みが最低限必要。

---

### 7. コード品質・保守性

#### 7.1 テスト・CI/CD

- [ ] ユニットテストの追加（`module/` 配下）
  - [ ] `calc_method.py`: 各ストラテジーの入出力テスト（小さなテンソルで検証）
  - [ ] `calc_target.py`: ターゲットストラテジーのテスト
  - [ ] `history.py`: ファイル I/O のテスト（一時ディレクトリ使用）
  - [ ] `presets.py`: プリセット保存/読込のテスト
  - [ ] `utility.py`: ファイル名生成、モデルパス正規化のテスト
  - [ ] `extension_manager.py`: フック登録・実行のテスト
- [ ] CI/CD パイプライン構築（lint, test, build）
  - [ ] GitHub Actions: `ruff check`, `pytest`, `mypy`
  - [ ] pre-commit hooks の設定

> **[考察]** 現在テストコードは完全に存在しない。`.ruff_cache/` はあるため ruff は導入済みだが、
> CI での自動実行設定はない。`pytest` + `pytest-cov` でカバレッジ計測を推奨。
> モデルファイル（数 GB）を伴うテストはモックが必須。`@sd_mecha.merge_method` 付きの関数は
> 通常の PyTorch テンソルで単体テスト可能。

#### 7.2 エラーハンドリングの統一

- [ ] カスタム例外クラスの定義
  - [ ] `module/exceptions.py` を新規作成
  - [ ] `ConfigError`, `ModelLoadError`, `MergeError`, `ExtensionError` 等
- [ ] 裸の `except` / `except Exception` の置き換え
  - [ ] `presets.py` L27: `except:` → 適切なエラーハンドリング
  - [ ] `generation.py` L55: `except Exception` → `except OSError`
  - [ ] `extension_manager.py` のフック実行部: 例外時のフォールバック戦略を明確化
- [ ] ユーザー向けエラーメッセージの改善
  - [ ] 技術的なスタックトレースではなく、対処法を含むメッセージ

> **[考察]** `presets.py` の `load_preset()` は裸の `except:` で空辞書を返しており、
> 設定ファイルの破損を完全に無視する。`generation.py` の SDXL 判定のフォールバック（ファイルサイズ推定）も
> サイレントフォールバックで、ログ出力がない箇所がある。

#### 7.3 ロギング・ドキュメンテーション

- [ ] ロギングの構造化（JSON ログ出力オプション）
  - [ ] `python-json-logger` or `structlog` の導入
  - [ ] ログレベルの細分化（現在は INFO/ERROR のみが大半）
  - [ ] ファイルログとコンソールログの分離
- [ ] ドキュメント整備
  - [ ] API リファレンス（`module/` 配下の全 public 関数）
  - [ ] 拡張機能開発ガイド（`extensions/sample_extension/` を参考にしたテンプレート）
  - [ ] YAML 設定ファイルのスキーマ定義（JSON Schema or Pydantic モデル）

> **[考察]** 現在 `main.py`, `ui/app.py` ともに `logging.basicConfig()` を個別に呼んでおり、
> ロガー設定が散在している。ルートロガーの設定を `module/logging_config.py` に統一すべき。
> `RichHandler` は開発時には便利だが、JSON ログ出力との切替可能にしたい。

#### 7.4 型安全性と YAML 設定のバリデーション

- [ ] Pydantic モデルによる設定バリデーション
  - [ ] `module/config_schema.py` を新規作成
  - [ ] `MergeConfig`, `ModelConfig`, `GenerationConfig` 等のデータクラス定義
  - [ ] YAML 読み込み時に自動バリデーション
  - [ ] 無効な設定時の明確なエラーメッセージ
- [ ] 型注釈の補完
  - [ ] `history.py`, `presets.py` 等の型注釈が欠如しているモジュール
  - [ ] `mypy --strict` でのチェック

> **[考察・新規追加]** `main.py` の `config` は生の `dict`。`models` リスト内の各要素は
> `required_fields = ["left", "right", "velocity", "strategy"]` でバリデーションしているが、
> `velocity` の型チェック（float であるべき）や `strategy` の有効値チェックはランタイムまで遅延する。
> Pydantic v2 を使えば YAML → 型安全なオブジェクトへの変換とバリデーションを一括で行える。

---

### 8. 拡張機能エコシステム

- [ ] 拡張機能のバージョン管理・依存関係解決
  - [ ] 各拡張に `manifest.json`（名前、バージョン、依存する他拡張、最低 SD-merger バージョン）
  - [ ] 起動時に依存関係チェック、不足時は警告表示
- [ ] 拡張機能のホットリロード対応
  - [ ] `importlib.reload()` による差分リロード
  - [ ] フック登録のクリア→再登録の安全な処理
  - [ ] UI からの有効/無効トグル

> **[考察]** `extension_manager.py` の `load_extensions()` は起動時に 1 回だけ実行。
> フックリスト (`_HOOKS_*`) はモジュールレベルのグローバル変数のため、
> ホットリロード時には先に `_HOOKS_PRE_CONFIG.clear()` 等でクリアしてから再読込する必要がある。
> また `_EXTENSION_STRATEGIES` のキー衝突時に `logging.warning` を出すが処理は続行するため、
> 拡張同士の互換性問題がサイレントに起こりうる。

---

### 9. アーキテクチャ改善（新規セクション）

> **[考察・新規追加]** コードベース全体を通じて発見した構造的課題。

#### 9.1 モジュールの責務分離

- [ ] `main.py` の分割
  - [ ] マージレシピ組み立てロジックを `module/recipe_builder.py` に抽出
  - [ ] コマンドライン処理を `cli.py` に分離
  - [ ] `main.py` は薄いエントリーポイントのみに
- [ ] UI コンポーネントの共通パターン抽出
  - [ ] 複数コンポーネント (`mbw_each.py`, `multi_merge.py`, `lora_ops.py`) で
    `sys.path.insert` → `from main import main` を繰り返している → 共通ヘルパー化
  - [ ] 一時 YAML ファイル生成パターンの共通ユーティリティ化

> **[考察]** `ui/components/mbw_each.py` L80-84, `multi_merge.py` L97-100,
> `lora_ops.py` L186-212 にほぼ同一の「Config → 一時 YAML → main() 呼出」パターンが重複。
> `ui/utils.py` に `run_merge_from_config(config: dict, output_dir: str)` を追加すれば
> 約 20 行のコード重複を排除できる。

#### 9.2 SD 3.x / Flux 対応への準備

- [ ] モデルアーキテクチャの抽象化
  - [ ] 現在は SD1.5 と SDXL の 2 種類のみ対応
  - [ ] `const.py` の `SDKeyWrapper` を拡張してモデルタイプのレジストリ化
  - [ ] `analysis.py` の `categorize_key()` を DiT 系アーキテクチャ向けに拡張
- [ ] `diffusers` パイプラインの動的選択
  - [ ] `generation.py` の `_is_sdxl_checkpoint()` を一般化 → `detect_model_architecture()`
  - [ ] `SD3Pipeline`, `FluxPipeline` 等への対応準備

> **[考察]** `categorize_key()` は `input_blocks`, `middle_block`, `output_blocks` の正規表現で
> SD1.5/SDXL の UNet ブロックを分類しているが、SD3/Flux の DiT (Diffusion Transformer) は
> 全く異なるキー構造 (`joint_blocks`, `single_blocks` 等) を持つ。
> 今のうちにモデル判定・キー分類を `Strategy パターン` で差し替え可能にしておくと将来の拡張が容易。

#### 9.3 設定管理の一元化

- [ ] アプリケーション設定ファイル (`settings.yaml` or `.env`)
  - [ ] モデルディレクトリ、出力ディレクトリ、VRAM 上限、デフォルト生成設定等
  - [ ] 現在は各モジュールがハードコードされたパスを持っている
    - `presets.py`: `PRESET_DIR` がプロジェクトルートからの相対パス
    - `history.py`: `HISTORY_FILE` がプロジェクトルートからの相対パス
    - `multi_merge.py`: `out_dir` がハードコード

> **[考察]** パス管理が `os.path.join(os.path.dirname(__file__), "..", "..")` の連鎖で
> 散在しており、インストール先やディレクトリ構造の変更に脆弱。
> 中央設定ファイルとクラス (`module/settings.py`) を作成し、パス解決を一元化すべき。

---

## ✅ 実装済み

- [x] 基本マージ戦略（subtraction, addition, multiplication, average, mix, replace）
- [x] MBW（Merge Block Weight）サポート
- [x] 拡張機能システム（`extension_manager.py`）— フック: pre_config / pre_merge / post_merge
- [x] LoRA 抽出・マージ（`lora_ops` 拡張）
- [x] LoRA リサイズ（`resize_lora` 拡張）
- [x] 画像生成モジュール（diffusers バックエンド）— SD1.5 / SDXL 自動判定
- [x] ヒストリ管理（JSON 永続化、最大 100 件）
- [x] プリセット管理（JSON 形式で保存 / 読込）
- [x] マルチマージ（バッチ実行）— カンマ区切りコマンド構文
- [x] XY Plot 生成（Velocity / Strategy / CFG / Steps 軸）
- [x] Dice Roll マージ
- [x] Elemental Merge
- [x] モデル分析・比較（Cosine Similarity, Euclidean Distance, Mean Absolute Difference）
- [x] SD 1.5 / SDXL 自動判定
- [x] Gradio WebUI
- [x] ターゲット計算戦略（addition, subtraction, multiplication, mix）
- [x] 正規化戦略（std/mean マッチング）
- [x] Weight Matching 前処理（ハンガリアンアルゴリズム）
- [x] sd-mecha ベースのストリーミングマージ

---

## 📊 技術的負債サマリー

| カテゴリ | 該当ファイル | 内容 |
|---------|------------|------|
| 裸の `except` | `presets.py` L27 | `except:` で例外を握りつぶし |
| コード重複 | `mbw_each.py`, `multi_merge.py`, `lora_ops.py` | YAML 生成 → `main()` 呼出パターンが 3 箇所で重複 |
| ハードコードパス | `presets.py`, `history.py`, `multi_merge.py` | `os.path.join(dirname, "..", "..")` の連鎖 |
| SDXL 未対応 | `analysis.py` | `categorize_key()` が SD1.5 構造のみ前提 |
| テスト不在 | 全モジュール | ユニットテスト 0 件 |
| 型注釈不足 | `history.py`, `presets.py` | 入出力の型が不明瞭 |
| メモリ管理 | `generation.py` | 毎回フルロード、キャッシュなし |
| ロガー設定散在 | `main.py`, `ui/app.py`, `preprocess_method.py` | `basicConfig()` が複数箇所で呼ばれている |
