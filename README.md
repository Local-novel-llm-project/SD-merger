# sd-merger
*English* | [日本語](README_ja.md)


sd-merger is a minimal tool for SD vector merge.


## Installation

1. Clone repository
   ```bash
   git clone https://github.com/Local-novel-llm-project/SD-merger
   cd SD-merger
   ```

1. (Optional but recommended) Create and activate your Python environment
   ```bash
   # for example, we use venv
   python -m venv venv
   ```

1. Install dependencies using pip
   ```bash
   pip install -r requirements.txt
   ```


## Usage

```bash
python main.py -c <your yaml config>.yaml
```

## Configs

sd-merger uses the YAML format for configure merge method.
Examples of configuration files can be found in the `example` folder.

Details of the settings are written in the comments of example configuration file.


## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
