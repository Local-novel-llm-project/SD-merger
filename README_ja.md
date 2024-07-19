# sd-merger
[English](README.md) | *日本語*


sd-merger はLLMをベクトルマージするための小さなツールです。


## インストール

1. リポジトリのクローン
   ```bash
   git clone https://github.com/Local-novel-llm-project/SD-merger
   cd SD-merger
   ```

1. (オプション、推奨) Python仮想環境の作成とアクティベート
   ```bash
   # for example, we use venv
   python -m venv venv
   ```

1. pipを使って依存関係のインストール
   ```bash
   pip install -r requirements.txt
   ```


## 使い方

```bash
python main.py -c <your yaml config>.yaml
```

## 設定

sd-merger はマージ方法の設定にYAMLフォーマットを使用しています。
設定ファイルの例は `example` フォルダ以下にあります。

各設定の詳細は設定ファイル例の中にコメントで書いています。


## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
