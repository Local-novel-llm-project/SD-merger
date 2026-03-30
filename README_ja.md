# sd-merger

[English](README.md) | *日本語*

Stable Diffusion モデルのマージを行うツールキットです。CLI のコア処理をそのまま再利用しつつ、UI は Reflex ベースへ移行しています。

## 特徴

- `main.py` の既存マージパイプラインをそのまま使う Reflex UI
- キュー実行、履歴保存、履歴からの再実行に対応
- UI 固有の責務を `ui/services`、`ui/state`、`ui/pages` へ再編
- 拡張機構と Arthemy Tuner フックを継続利用
- Stable Diffusion 1.5 / SDXL をサポート

## インストール

```bash
git clone https://github.com/Local-novel-llm-project/SD-merger
cd SD-merger
pip install -r requirements.txt
```

## 使い方

Reflex UI を起動:

```bash
reflex run
```

従来どおり CLI ランチャー経由でも起動できます:

```bash
python main.py ui --port 3000
```

YAML から直接マージ:

```bash
python main.py merge -c example/example.yaml
```

## UI 構成

- `ui/services`: UI 用の config 組み立てと queue/history 連携
- `ui/state`: Reflex の状態管理とイベント処理
- `ui/pages`: Merge / Queue / History / Arthemy Tuner の画面定義

コア処理は引き続き `main.py`、`module/*`、`extensions/*` を使います。
Reflex の正面入口はトップレベルの `app` パッケージに寄せ、UI 本体の実装は `ui/*` に残します。

## CLI

```bash
python main.py merge -c example/example.yaml
python main.py tune --model models/example.safetensors
python main.py ui --host 0.0.0.0 --port 3000
```

## 補足

- Reflex のフロントエンド既定ポートは `3000`、バックエンドは `3001` です。
- 旧 Gradio UI は削除済みです。追加機能は Reflex の pages/services を拡張して移植します。

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
