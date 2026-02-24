# SD-merger TODO 機能の実装計画

本計画は、`TODO.md` に記載されている未実装機能の追加および最適化タスクを遂行するための実装設計となります。

## User Review Required

> [!IMPORTANT]
> - `extensions/lora_ops/kohyas/model_util.py` における `text_encoder` の `dtype` 指定に関して、動作確認が必要となります。修正後、提供のスクリプト等でテストを依頼する想定です。
> - `sai_model_spec.py` のハッシュ計算において、テンソルをメモリに一度に載せず、1次元 uint8 のバイトストリームとして個別にハッシュ化する手法を取ります。これによるハッシュ値が safetensors の仕様と合致するよう調整します。
> - `train_util.py` の VAE による latents キャッシュ処理については、バッチ化 (DataLoader 或いは Bucket 内のチャンク処理) を導入し、GPU/CPU コール回数を減らすことで高速化を図ります。

## Proposed Changes

---

### UI Component

UI の `render_lora_ops_tab()` に関連する実装は、既に `ui/components/lora_ops.py` などでカバーされて完了済みです。
`TODO.md` の関連チェックボックスを `[x]` に更新します。

---

### Core Model & LoRA Extractor Optimizations

#### [MODIFY] extensions/lora_ops/kohyas/model_util.py
- **メモリ最適化**: `save_diffusers_checkpoint` メソッドにて `UNet2DConditionModel.from_pretrained` で事前学習モデルを丸ごとロードして破棄する実装 (`TODO this consumes a lot of memory`) があります。これを回避し、HuggingFace Diffusers の `load_config` と `from_config` を用いて、初期化時のメモリ消費を大幅に削減（ディスクからのメタデータの読み込みやダミーのウェイトの確保に留める）し、そこに `state_dict` を読み込ませる構成に変更します。
- **バグ改善 & Testing**: `load_models_from_stable_diffusion_checkpoint` 等で `dtype` (例えば `torch.float16`) が明示された場合に、`CLIPTextModel._from_config` の呼び出し直後、もしくは内包される Linear レイヤー等に対して確実に指定 dtype で構成されるよう `.to(dtype)` 等のキャストをテストし、適切に反映されるようにロジックを確定させます。

#### [MODIFY] extensions/lora_ops/kohyas/train_util.py
- **高速化**: `cache_latents` にて、1画像ずつunsqueezeしエンコードしているループ処理（`TODO ここを高速化したい`）において、同一の bucket (解像度) を持つ画像をミニバッチ化 (例えばバッチサイズ 4～8) し、`vae.encode` をバッチ単位で処理させることで高速化します。

#### [MODIFY] extensions/lora_ops/kohyas/sai_model_spec.py
- **メモリ最適化**: `precalculate_safetensors_hashes` および `build_metadata` 内の `TODO メモリを消費せずかつ正しいハッシュ計算の方法がわかったら実装する` に対処します。Safetensors のバイトレイアウトにおいて、ヘッダーに記載される順序（英数字順）に従い、テンソルを1つずつ `.view(torch.uint8).numpy().tobytes()` のように直列化して `hashlib.sha256().update()` に流し込むことで、OOM を防ぎながら完全な tensor contents の SHA256 を生成できるようにします。

---

### Feature Implementation

#### [MODIFY] extensions/lora_ops/kohyas/original_unet.py
#### [MODIFY] extensions/lora_ops/kohyas/sdxl_original_unet.py
- **Hypernetworks サポート**: `TODO support Hypernetworks` の箇所について、Stable Diffusion Web UI の仕様に準拠する形で、中間の transformer blocks 等に Hypernetwork 用の処理層（もし引数に含まれる場合）をフック、またはネットワークの定義として適用できるようにパスファインディングを実装します。（具体的な構成として外からネットワークオブジェクトが渡せるようなインターフェイスか、単純に呼び出し時の kwargs で受け取るようにします）

#### [MODIFY] extensions/lora_ops/kohyas/merge_lora.py
#### [MODIFY] extensions/lora_ops/kohyas/svd_merge_lora.py
- **Feature**: `TODO read sai modelspec` において、マージ対象のベースモデル（またはLoRA）に付属する Safetensors メタデータを `load_metadata_from_safetensors` 等で読み取り、`modelspec.prediction_type` などのパラメーターが `v` (v-prediction) であるかを確認し、`args.v2` フラグ等と照合・自動設定させる処理を実装します。

## Verification Plan

### Automated Tests
1. **Hash Verification**: `sai_model_spec.py` 内に簡易なスクリプトまたはインライン定義を渡し、従来通りにテンソル全体を `safetensors.torch.save` して得られた完全なハッシュ値と、最適化ロジックによるハッシュ値が正確に合致するかをユニットレベルで検証します。
2. **LoRA Extraction/Merge Trial**: 最適化対応を行った後、UI 介在で適当なチェックポイント 2 種を用いて LoRA Extract を回し、メモリ使用率が従来よりも下がっていることを確認できるようスクリプトを実行します。

### Manual Verification
- コマンドラインや UI で `cache_latents` （マージや学習時に呼ばれる場合）の速度が向上しているかをコンソールの進捗ログ (it/s) で確認します。
- `dtype` の適用状況については、デバッグログを差し込む等して `text_encoder.dtype` が渡された引数に準拠しているか確認します。
