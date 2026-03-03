import gradio as gr
from ui.utils import get_model_list, get_model_path


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
