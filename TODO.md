# SD-merger TODO List

このファイルは、プロジェクト内の未実装機能(`TODO`, `FIXME`, `NotImplemented`)を整理したタスクリストです。

## 🔴 High Priority (高優先度)

- [x] **UI:** LoRA Operations (Extract & Merge LoRAs) の UI 実装 (`ui/app.py` にある `*UI implementation pending...*` の解消)
- [x] **Optimization:** モデル操作時のメモリ消費量の最適化 (`extensions/lora_ops/kohyas/model_util.py` の `TODO this consumes a lot of memory`)

## 🟡 Medium Priority (中優先度)

- [x] **Feature:** UNet および SDXL UNet における Hypernetworks サポートの実装 (`original_unet.py`, `sdxl_original_unet.py`)
- [x] **Feature:** LoRA マージ時における `sai_model_spec` の読み込みサポート (`svd_merge_lora.py`, `merge_lora.py`)
- [x] **Optimization:** `sai_model_spec.py` のハッシュ計算におけるメモリ最適化
- [x] **Optimization:** `train_util.py` 内の特定の処理の高速化 (`TODO ここを高速化したい`)

## 🟢 Low Priority (低優先度)

- [x] **Feature:** Attention Slicing のサポート追加 (`original_unet.py`, `sdxl_original_unet.py` で `NotImplementedError` が発生するモデル向け)
- [x] **Refactoring:** LoRA 関連の処理を `apply_to` と共通関数にリファクタリング (`lora.py`)
- [x] **Testing:** `text_encoder` を指定形式の `dtype` で正しく作成できるかの検証 (`model_util.py`)
- [x] **Refactoring:** PyTorch 側で Issue が修正された後、UNet 内の workaround（キャスト処理）を削除 (`original_unet.py`, `sdxl_original_unet.py`)
