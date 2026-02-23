import gradio as gr
from module.analysis import analyze_models


def render_analysis_tab():
    gr.Markdown("### Model Difference Analysis")
    gr.Markdown("Compare two Safetensors models block by block.")

    with gr.Row():
        with gr.Column(scale=1):
            model_a = gr.File(label="Model A", file_types=[".safetensors"])
            model_b = gr.File(label="Model B", file_types=[".safetensors"])

            metric = gr.Dropdown(
                label="Comparison Metric",
                choices=[
                    "Cosine Similarity",
                    "Euclidean Distance",
                    "Mean Absolute Difference",
                ],
                value="Cosine Similarity",
            )

            analyze_btn = gr.Button("Analyze Differences", variant="primary")
            output_log = gr.Textbox(label="Log", interactive=False)

        with gr.Column(scale=2):
            output_plot = gr.Plot(label="Difference Bar Chart")

    def run_analysis(ma, mb, met):
        if not ma or not mb:
            return None, "Please upload both Model A and Model B."

        try:
            fig = analyze_models(ma.name, mb.name, metric=met)
            return fig, "Analysis complete."
        except Exception as e:
            return None, f"Error during analysis: {e}"

    analyze_btn.click(
        run_analysis,
        inputs=[model_a, model_b, metric],
        outputs=[output_plot, output_log],
    )
