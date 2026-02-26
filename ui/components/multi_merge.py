import os
import gradio as gr
import tempfile
import yaml
import logging


def parse_multi_merge_command(cmd_text: str):
    """
    複数行のバッチコマンドをパースしてYAMLのリスト形式に変換する。
    例:
    O=merged_1.safetensors, IN_A_00=0.75, Model_A=model1.safetensors, Model_B=model2.safetensors
    """
    operations = []
    lines = cmd_text.strip().split("\n")

    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue

        op_dict = {"strategy": "mbw_each"}
        mbw_a = [1.0] * 26
        mbw_b = [0.0] * 26  # 互換性のため一旦26で固定

        parts = line.split(",")
        for part in parts:
            if "=" not in part:
                continue

            key, val = part.split("=", 1)
            key = key.strip()
            val = val.strip()

            if key == "O":
                op_dict["output_name"] = val
            elif key == "Model_A":
                op_dict["left"] = val
            elif key == "Model_B":
                op_dict["right"] = val
            elif key.startswith("IN_A_"):
                idx = int(key.split("_")[-1])
                mbw_a[1 + idx] = float(val)
            elif key.startswith("IN_B_"):
                idx = int(key.split("_")[-1])
                mbw_b[1 + idx] = float(val)
            elif key == "M_00":
                mbw_a[13] = float(val)
                mbw_b[13] = 1.0 - float(
                    val
                )  # M_00 はAの比率を入れ、Bを補完する簡易実装例
            elif key.startswith("OUT_A_"):
                idx = int(key.split("_")[-1])
                mbw_a[14 + idx] = float(val)
            elif key.startswith("OUT_B_"):
                idx = int(key.split("_")[-1])
                mbw_b[14 + idx] = float(val)
            elif key == "base_alpha":
                op_dict["velocity"] = float(val)
                op_dict["strategy"] = "mix"  # ブロック指定がない全体アルファとみなす

        # もし MBW の上書きがあったら文字列化して追加
        if op_dict["strategy"] == "mbw_each":
            op_dict["mbw_a"] = ",".join(map(str, mbw_a))
            op_dict["mbw_b"] = ",".join(map(str, mbw_b))

        operations.append(op_dict)

    return operations


def render_multi_merge_tab():
    """Multi-merge 画面のレンダリング"""
    gr.Markdown("### Multi Merge Command Window (バッチ実行)")
    gr.Markdown(
        "1行に1つのマージ処理を記述し、複数のパラメータでの一括マージを行います。変数同士はカンマ(`,`)で区切ります。"
    )
    gr.Markdown(
        "**利用可能な変数:** `O` (出力ファイル名), `Model_A`, `Model_B`, `IN_A_00` ~ `IN_A_11`, `OUT_A_00` ~ `OUT_A_11`, `M_00` 等"
    )

    cmd_text = gr.Textbox(
        label="Commands",
        lines=10,
        placeholder="O=out1.safetensors, IN_A_00=0.75, Model_A=model_a.safetensors, Model_B=model_b.safetensors\nO=out2.safetensors, OUT_B_11=0.8, Model_A=model_a.safetensors, Model_B=model_b.safetensors",
    )

    run_btn = gr.Button("Run Batch Merge", variant="primary")
    output_log = gr.Textbox(label="Batch Log", lines=5)

    def run_batch_merge(cmd):
        ops = parse_multi_merge_command(cmd)
        if not ops:
            return "No valid commands parsed."

        import sys

        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        )
        from main import main as merger_main
        from module.extension_manager import load_extensions
        load_extensions()

        log_msgs = []
        for i, op in enumerate(ops):
            config = {"target_model": op.get("left", ""), "models": [op]}
            out_name = op.pop("output_name", f"batch_merged_{i}.safetensors")

            try:
                with tempfile.NamedTemporaryFile(
                    "w", delete=False, suffix=".yaml"
                ) as f:
                    yaml.dump(config, f)
                    tmp_cfg = f.name

                out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "output"))
                # ここではファイル名を直接指定する仕組みが main.py にはないため、一時的に generate_filename をフックするか出力後にリネームする必要があるが、一旦標準の流れを実行。
                merger_main(tmp_cfg, out_dir)
                log_msgs.append(
                    f"Completed operation {i + 1}: Output expected around {out_dir}"
                )
            except Exception as e:
                log_msgs.append(f"Error in operation {i + 1}: {str(e)}")

        return "\n".join(log_msgs)

    run_btn.click(run_batch_merge, inputs=[cmd_text], outputs=[output_log])
