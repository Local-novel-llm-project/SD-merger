import gradio as gr
import pandas as pd
from module.queue_manager import queue_manager


def get_queue_df() -> pd.DataFrame:
    queue = queue_manager.get_queue()
    if not queue:
        return pd.DataFrame(
            columns=["ID", "Task Name", "Output Name", "Status", "Progress", "Description", "Added At", "Error"]
        )

    data = []
    for t in queue:
        data.append(
            {
                "ID": t["id"],
                "Task Name": t.get("name", "Unknown"),
                "Output Name": t.get("output_name", ""),
                "Status": t["status"].upper(),
                "Progress": f"{t.get('progress', 0.0) * 100:.1f}%",
                "Description": t.get("progress_desc", ""),
                "Added At": t.get("added_at", ""),
                "Error": t.get("error", "") or "",
            }
        )
    return pd.DataFrame(data)


def render_queue_tab():
    gr.Markdown("### Tasks Queue Management")
    gr.Markdown("View and manage background merge tasks. Tasks are processed sequentially. ")

    # State
    selected_row = gr.State(-1)

    with gr.Row():
        queue_status_md = gr.Markdown(value="**Queue Status**: " + ("PAUSED" if queue_manager.is_paused else "RUNNING"))

        refresh_btn = gr.Button("🔄 Refresh Queue", variant="secondary")
        pause_btn = gr.Button("⏸ Pause Queue", variant="secondary")
        resume_btn = gr.Button("▶ Resume Queue", variant="primary")

    with gr.Row():
        remove_btn = gr.Button("🗑 Remove Selected", variant="stop", interactive=False)
        clear_done_btn = gr.Button("Clear Completed/Error", variant="secondary")

    message = gr.Textbox(label="Message", interactive=False)

    # Use gr.Timer to auto-refresh the queue table every 2 seconds
    timer = gr.Timer(value=2)

    queue_table = gr.Dataframe(
        value=get_queue_df(),
        headers=["ID", "Task Name", "Output Name", "Status", "Progress", "Description", "Added At", "Error"],
        interactive=False,
        wrap=True,
    )

    def update_ui_wrapper(msg=""):
        df = get_queue_df()
        q_status = "**Queue Status**: " + ("PAUSED" if queue_manager.is_paused else "RUNNING")
        return [df, q_status, msg]

    def on_refresh():
        df, q_status, msg = update_ui_wrapper()
        return df, q_status

    refresh_btn.click(on_refresh, inputs=[], outputs=[queue_table, queue_status_md])
    timer.tick(on_refresh, inputs=[], outputs=[queue_table, queue_status_md])

    def on_pause():
        queue_manager.pause()
        return update_ui_wrapper("Queue paused.")

    pause_btn.click(on_pause, inputs=[], outputs=[queue_table, queue_status_md, message])

    def on_resume():
        queue_manager.resume()
        return update_ui_wrapper("Queue resumed.")

    resume_btn.click(on_resume, inputs=[], outputs=[queue_table, queue_status_md, message])

    def on_select(evt: gr.SelectData):
        if evt:
            # evt.index is a tuple [row, col]
            return gr.update(interactive=True), evt.index[0]
        return gr.update(interactive=False), -1

    queue_table.select(on_select, inputs=[], outputs=[remove_btn, selected_row])

    def on_remove(idx: int):
        if idx < 0:
            return update_ui_wrapper("No row selected.")

        df = get_queue_df()
        if idx >= len(df):
            return update_ui_wrapper("Invalid row selected.")

        task_id = df.iloc[idx]["ID"]
        success = queue_manager.remove_task(task_id)
        if success:
            df, q_status, _ = update_ui_wrapper()
            return df, q_status, f"Removed task {task_id}."
        else:
            df, q_status, _ = update_ui_wrapper()
            return df, q_status, f"Cannot remove running task: {task_id}"

    remove_btn.click(on_remove, inputs=[selected_row], outputs=[queue_table, queue_status_md, message])

    def on_clear_done():
        queue_manager.clear_completed()
        return update_ui_wrapper("Completed tasks cleared.")

    clear_done_btn.click(on_clear_done, inputs=[], outputs=[queue_table, queue_status_md, message])

    return refresh_btn, queue_table
