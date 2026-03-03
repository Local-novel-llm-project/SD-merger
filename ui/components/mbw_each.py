import gradio as gr
import numpy as np
import pandas as pd
from ui.utils import get_model_list, get_model_path


def generate_curve(curve_type: str, start_val: float, end_val: float, length: int) -> list[float]:
    """指定されたカーブタイプで長さ length の補間配列を生成する"""
    if length <= 0:
        return []
    if length == 1:
        return [start_val]

    t = np.linspace(0, 1, length)

    if curve_type == "Flat":
        # Flatの場合はStart値を全体に適用 (Endは無視)
        y = np.full(length, start_val)
    elif curve_type == "Linear":
        y = start_val + (end_val - start_val) * t
    elif curve_type == "Sigmoid":
        # -6 to 6 range for a standard sigmoid shape
        x = np.linspace(-6, 6, length)
        sig = 1 / (1 + np.exp(-x))
        y = start_val + (end_val - start_val) * sig
    elif curve_type == "Cosine":
        # 0 to pi
        cos_val = (1 - np.cos(t * np.pi)) / 2
        y = start_val + (end_val - start_val) * cos_val
    else:
        y = np.full(length, start_val)

    return [round(float(v), 3) for v in y]


def generate_mbw_array(
    arch: str,
    base_val: float,
    in_curve: str,
    in_start: float,
    in_end: float,
    mid_val: float,
    out_curve: str,
    out_start: float,
    out_end: float,
) -> list[float]:
    """アーキテクチャに応じた一連のMBW配列を生成する"""
    if arch == "SD1.5 (26 Blocks)":
        in_len, out_len = 12, 12
    else:  # SDXL (20 Blocks)
        in_len, out_len = 9, 9

    in_blocks = generate_curve(in_curve, in_start, in_end, in_len)
    out_blocks = generate_curve(out_curve, out_start, out_end, out_len)

    return [base_val] + in_blocks + [mid_val] + out_blocks


def build_plot_data(mbw_array: list[float], arch: str) -> pd.DataFrame:
    """gr.LinePlot 用のデータフレームを構築する"""
    blocks = []
    blocks.append("BASE")

    if arch == "SD1.5 (26 Blocks)":
        blocks.extend([f"IN{i:02d}" for i in range(12)])
        blocks.append("MID")
        blocks.extend([f"OUT{i:02d}" for i in range(12)])
    else:
        blocks.extend([f"IN{i:02d}" for i in range(9)])
        blocks.append("MID")
        blocks.extend([f"OUT{i:02d}" for i in range(9)])

    df = pd.DataFrame({"Block": blocks, "Index": range(len(mbw_array)), "Alpha": mbw_array})
    return df


def render_mbw_each_tab():
    """MBW Each 画面のレンダリングを行う"""
    gr.Markdown("### MBW Each (AとBで個別の乗数を持つ階層マージ)")
    gr.Markdown(
        "A のブロックごとの重みと、B のブロックごとの重みを別々に指定してマージします。(通常は A=1-B となりますが、これなら A=1, B=1 のような特殊な加算も可能です)"
    )

    with gr.Row():
        model_list = get_model_list()
        model_a = gr.Dropdown(label="Model A (Left)", choices=model_list)
        model_b = gr.Dropdown(label="Model B (Right)", choices=model_list)
        model_c = gr.Dropdown(
            label="Model C (Base/Target, optional)",
            choices=model_list,
        )

    # --- Pincushion Merge (Auto Curve Generator) UI ---
    with gr.Accordion("Pincushion Merge (Auto Curve Generator)", open=False):
        gr.Markdown("連続的な関数を用いてブロックごとの重み（Alpha）配列を自動生成します。")

        with gr.Row():
            arch_radio = gr.Radio(
                choices=["SD1.5 (26 Blocks)", "SDXL (20 Blocks)"],
                value="SD1.5 (26 Blocks)",
                label="Target Architecture",
            )
            base_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="BASE Block Value")
            mid_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="MID Block Value")

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### IN Blocks")
                in_curve = gr.Dropdown(
                    choices=["Flat", "Linear", "Sigmoid", "Cosine"], value="Linear", label="Curve Type"
                )
                with gr.Row():
                    in_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Start")
                    in_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="End")
            with gr.Column():
                gr.Markdown("#### OUT Blocks")
                out_curve = gr.Dropdown(
                    choices=["Flat", "Linear", "Sigmoid", "Cosine"], value="Linear", label="Curve Type"
                )
                with gr.Row():
                    out_start = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Start")
                    out_end = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=1.0, label="End")

        with gr.Row():
            plot_output = gr.LinePlot(
                x="Index", y="Alpha", tooltip=["Block", "Alpha"], title="Alpha Weight Curve", width=800, height=300
            )

        with gr.Row():
            generated_mbw_text = gr.Textbox(label="Generated MBW Array", interactive=False)

        with gr.Row():
            apply_to_a_btn = gr.Button("Apply to Model A")
            apply_to_b_btn = gr.Button("Apply to Model B")

        def update_curve(arch, base_v, in_c, in_s, in_e, mid_v, out_c, out_s, out_e):
            arr = generate_mbw_array(arch, base_v, in_c, in_s, in_e, mid_v, out_c, out_s, out_e)
            df = build_plot_data(arr, arch)
            text_val = ",".join(map(str, arr))
            return df, text_val

        inputs_list = [arch_radio, base_slider, in_curve, in_start, in_end, mid_slider, out_curve, out_start, out_end]

        # イベントバインディング
        for ctrl in inputs_list:
            ctrl.change(fn=update_curve, inputs=inputs_list, outputs=[plot_output, generated_mbw_text])

        # 初期表示用の更新（ダミーイベント）
        # gradioでは通常loadイベント等で初期化するが、ここでは表示時に関数を呼んで初期値を入れる
        arch_radio.change(fn=update_curve, inputs=inputs_list, outputs=[plot_output, generated_mbw_text])

    # --- 既存の MBW 入力 UI ---
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Model A Weights")
            mbw_a = gr.Textbox(
                label="MBW for Model A (カンマ区切り)",
                placeholder="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
                lines=2,
                value="1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
            )

        with gr.Column():
            gr.Markdown("#### Model B Weights")
            mbw_b = gr.Textbox(
                label="MBW for Model B (カンマ区切り)",
                placeholder="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
                lines=2,
                value="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
            )

    # Applyボタンのアクション
    apply_to_a_btn.click(lambda x: x, inputs=[generated_mbw_text], outputs=[mbw_a])
    apply_to_b_btn.click(lambda x: x, inputs=[generated_mbw_text], outputs=[mbw_b])

    with gr.Row():
        with gr.Accordion("Advanced Options", open=False):
            use_advanced_options = gr.Checkbox(label="Enable Advanced Options", value=False)
            output_name = gr.Textbox(label="Output Filename", value="mbw_each_merged.safetensors")

    merge_btn = gr.Button("Run MBW Each Merge", variant="primary")
    output_log = gr.Textbox(label="Output Log", lines=3)

    def run_mbw_each_merge(a, b, c, mbw_a_val, mbw_b_val, use_adv, out):
        if not a or not b:
            return "Error: Model A and Model B are required."

        try:
            # 入力チェック
            len_a = len([x for x in mbw_a_val.split(",") if x.strip()])
            len_b = len([x for x in mbw_b_val.split(",") if x.strip()])

            if len_a != len_b:
                return f"Error: Length mismatch. A={len_a}, B={len_b}"
            if len_a not in (26, 20):
                return f"Error: Length must be 26 (SD1.5) or 20 (SDXL). Found {len_a}."

            target_model_path = get_model_path(c) if c else get_model_path(a)
            config = {
                "target_model": target_model_path,
                "models": [
                    {
                        "left": get_model_path(a),
                        "right": get_model_path(b),
                        "strategy": "mbw_each",
                        "mbw_a": mbw_a_val,
                        "mbw_b": mbw_b_val,
                    }
                ],
            }
            if use_adv and out:
                config["output_name"] = out

            from module.queue_manager import queue_manager

            task_id = queue_manager.add_task(config, out, task_name="MBW Each")

            return f"MBW Each merge task '{task_id}' added to queue.\nOutput will be: {out}"

        except Exception as e:
            return f"Error: {str(e)}"

    merge_btn.click(
        run_mbw_each_merge,
        inputs=[model_a, model_b, model_c, mbw_a, mbw_b, use_advanced_options, output_name],
        outputs=[output_log],
    )
