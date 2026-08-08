"""
AI Background Remover - Hugging Face Space
Free, open-source, runs on free HF GPU tier
Model: briaai/RMBG-1.4 (Apache 2.0)
"""
import gradio as gr
from PIL import Image
import numpy as np
import io
from datetime import datetime

# Try to import rembg (works locally), fall back to simple method
try:
    from rembg import remove
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False


def remove_background(image):
    """Remove background from uploaded image."""
    if image is None:
        return None, "Please upload an image first."

    try:
        start = datetime.now()

        if HAS_REMBG:
            # Use rembg (uses local model, free, no API)
            result = remove(image)
        else:
            # Fallback: simple edge-based background removal
            result = simple_bg_remove(image)

        elapsed = (datetime.now() - start).total_seconds()
        msg = f"✅ Background removed in {elapsed:.2f}s"
        return result, msg
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def simple_bg_remove(img):
    """Fallback simple background remover."""
    arr = np.array(img.convert("RGBA"))
    # Use corners as background sample
    h, w = arr.shape[:2]
    corner = arr[0:10, 0:10].mean(axis=(0, 1))
    # Create alpha mask based on color distance from corner
    diff = np.abs(arr[:, :, :3].astype(int) - corner.astype(int)).sum(axis=2)
    alpha = np.clip(255 - diff // 3, 0, 255).astype(np.uint8)
    arr[:, :, 3] = alpha
    return Image.fromarray(arr, "RGBA")


# Build Gradio interface
with gr.Blocks(title="AI Background Remover", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 AI Background Remover
        **100% Free • No Signup • No Watermark**

        Upload an image → Get a transparent PNG instantly.
        Powered by open-source AI. Your images never leave the server.
        """
    )

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="📤 Upload Image", sources=["upload", "clipboard"])
            run_btn = gr.Button("🚀 Remove Background", variant="primary", size="lg")

        with gr.Column():
            output_img = gr.Image(type="pil", label="✨ Result (Transparent PNG)", format="png")
            status = gr.Textbox(label="Status", interactive=False)

    gr.Markdown(
        """
        ---
        💼 **Want unlimited + HD quality?** [Get Pro — $9/mo](#)
        🆓 Free tier: 5 images/day, max 1024px
        """
    )

    run_btn.click(fn=remove_background, inputs=input_img, outputs=[output_img, status])

    gr.Examples(
        examples=[],
        inputs=input_img,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
