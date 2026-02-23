import gradio as gr


def render_elemental_merge_tab():
    gr.Markdown("### Visual Elemental Merge (MBW Sliders)")
    gr.Markdown(
        "Fine-tune Merge Block Weights visually instead of using text strings. The generated MBW string can be copied into the main Merge Models tab."
    )

    with gr.Row():
        with gr.Column(scale=1):
            base_alpha = gr.Slider(
                label="Base (BASE)", minimum=0.0, maximum=1.0, step=0.05, value=0.5
            )

            gr.Markdown("#### Input Blocks (IN)")
            in_sliders = []
            for i in range(12):
                slider = gr.Slider(
                    label=f"IN{i:02d}", minimum=0.0, maximum=1.0, step=0.05, value=0.5
                )
                in_sliders.append(slider)

        with gr.Column(scale=1):
            gr.Markdown("#### Middle Block (MID)")
            mid_slider = gr.Slider(
                label="MID00", minimum=0.0, maximum=1.0, step=0.05, value=0.5
            )

            gr.Markdown("#### Output Blocks (OUT)")
            out_sliders = []
            for i in range(12):
                slider = gr.Slider(
                    label=f"OUT{i:02d}", minimum=0.0, maximum=1.0, step=0.05, value=0.5
                )
                out_sliders.append(slider)

        with gr.Column(scale=1):
            output_str = gr.Textbox(
                label="Generated MBW String", lines=4, interactive=False
            )
            copy_btn = gr.Button(
                "Copy to Clipboard"
            )  # Currently Gradio doesn't easily copy to clipboard without JS, but user can manually copy

            preset_dropdown = gr.Dropdown(
                label="Apply Preset",
                choices=[
                    "All 0.5",
                    "All 1.0",
                    "All 0.0",
                    "Gradual IN-MID-OUT",
                    "U-Net Only",
                ],
                value="All 0.5",
            )
            apply_preset_btn = gr.Button("Apply Preset")

    all_sliders = [base_alpha] + in_sliders + [mid_slider] + out_sliders

    def update_mbw_string(*vals):
        return ",".join(f"{v:g}" for v in vals)

    def apply_preset(preset):
        if preset == "All 0.5":
            return [0.5] * 26
        elif preset == "All 1.0":
            return [1.0] * 26
        elif preset == "All 0.0":
            return [0.0] * 26
        elif preset == "Gradual IN-MID-OUT":
            # Just an example pattern
            res = [0.5]
            for i in range(12):
                res.append(i / 11.0)
            res.append(1.0)
            for i in range(12):
                res.append(1.0 - (i / 11.0))
            return res
        elif preset == "U-Net Only":
            # Base=0, rest=1
            return [0.0] + [1.0] * 25
        return [0.5] * 26

    # Update string whenever any slider changes
    for slider in all_sliders:
        slider.change(update_mbw_string, inputs=all_sliders, outputs=[output_str])

    apply_preset_btn.click(
        apply_preset, inputs=[preset_dropdown], outputs=all_sliders
    ).then(update_mbw_string, inputs=all_sliders, outputs=[output_str])
