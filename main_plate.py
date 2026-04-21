import gradio as gr
from detection import detect_license

with gr.Blocks() as demo:
    gr.Markdown("# License Plate Detection System")
    gr.Markdown("## Detect a license plate from the car.")
    gr.Markdown("Upload an image of a plate product to detect any license.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload car image here!")
            predict_btn = gr.Button("DETECT!")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Detection result...")
            croped_image = gr.Image(type="pil", label="cropped plate")
            status_box = gr.HTML(label="Status message")

    predict_btn.click(
        fn=detect_license,
        inputs=input_image,
        outputs=[output_image, croped_image, status_box]
    )

demo.launch()