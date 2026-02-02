import os
import gradio as gr
from openai import OpenAI
import time
import tempfile
import atexit
import base64
import io
from PIL import Image

# This list will hold the paths of all generated chat logs for this session.
temp_files_to_clean = []

# --- Function to perform cleanup on exit ---
def cleanup_temp_files():
    """Iterates through the global list and deletes the tracked files."""
    if not temp_files_to_clean:
        return
    print(f"\nCleaning up {len(temp_files_to_clean)} temporary files...")
    for file_path in temp_files_to_clean:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  - Error removing {file_path}: {e}")
    print("Cleanup complete.")

# Register the cleanup function to be called on exit.
atexit.register(cleanup_temp_files)

# Initialize the OpenAI client to connect to a local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

TEXT_PROMPT = """
You are a helpful medical assistant advising a doctor at a hospital.
"""

IMAGE_PROMPT = """
You are an expert radiological assistant AI. Your primary goal is to provide a structured, accurate, 
and clinically relevant analysis of the provided medical image to assist a human radiologist.
You MUST NOT invent, guess, or hallucinate findings that are not clearly visible in the image.
Your analysis must be based solely on the visual data provided. 
If the image quality is insufficient for a confident assessment, you must state this as your primary finding.
Provide a detailed, objective, list of observations from your analysis here. 
Describe normal and abnormal findings.
If applicable, suggest potential next steps or correlations.
"""

def encode_image_to_base64(pil_image):
    """Converts a PIL image to a base64 encoded string."""
    if pil_image is None:
        return None
    buffered = io.BytesIO()
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def chat_with_openai(message, image, history, is_image_session, model_name, instructions, temperature, max_tokens):
    initial_download_update = gr.update(visible=False)
    image_input_update = gr.update(interactive=False)

    if not message.strip() and not image:
        return history, is_image_session, "", initial_download_update, gr.update()

    if not history:
        if image:
            system_prompt = IMAGE_PROMPT
            is_image_session = True
        else:
            system_prompt = instructions
            is_image_session = False
    else:
        system_prompt = IMAGE_PROMPT if is_image_session else instructions

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append(turn)

    current_user_content = []
    if message.strip():
        current_user_content.append({"type": "text", "text": message})

    if image and is_image_session and not history:
        base64_image = encode_image_to_base64(image)
        if base64_image:
            current_user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
    
    if len(current_user_content) == 1 and current_user_content[0]["type"] == "text":
         messages.append({"role": "user", "content": current_user_content[0]["text"]})
    else:
        messages.append({"role": "user", "content": current_user_content})
    
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""}
    ]

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=0.95,
            max_tokens=int(max_tokens),
            stream=True,
            extra_body={"top_k": 64},
        )
        
        full_content = ""
        last_yield_time = time.time()
        flush_interval_s = 0.04

        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
                history[-1]["content"] = full_content
                
                now = time.time()
                if now - last_yield_time >= flush_interval_s:
                    last_yield_time = now
                    yield history, is_image_session, "", initial_download_update, image_input_update

        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as temp_file:
            output_filepath = temp_file.name
            temp_file.write(full_content)
        temp_files_to_clean.append(output_filepath)
        final_download_update = gr.update(visible=True, value=output_filepath)
        yield history, is_image_session, "", final_download_update, image_input_update

    except Exception as e:
        error_message = f"❌ An error occurred: {str(e)}"
        history[-1]["content"] = error_message
        yield history, is_image_session, "", initial_download_update, image_input_update

# --- 4. Gradio UI ---
with gr.Blocks(title="💬 Local LLM Chatbot (Multimodal)") as demo:
    gr.Markdown("# 💬 Multimodal Chatbot (Powered by Local OpenAI-Compatible API)")
    gr.Markdown("Ask a text-based question, or upload an image for analysis at the start of a new chat.")
    
    is_image_session = gr.State(False)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, buttons=["copy"], label="Conversation")
            with gr.Row():
                msg = gr.Textbox(placeholder="Type a message...", scale=4, show_label=False)
                send_btn = gr.Button("Send", scale=1, variant="primary")
            with gr.Row():
                stop_btn = gr.Button("⏹️ Stop", scale=1)
                clear_btn = gr.Button("🗑️ New Chat", scale=1)
                download_btn = gr.DownloadButton("⬇️ Download Last Response", visible=False, scale=3)

        with gr.Column(scale=2):
            model_selector = gr.Dropdown(
                label="Select Model",
                choices=[
                    "google/medgemma-27b-it",
                    "google/medgemma-27b-text-it",
                    "google/medgemma-4b-it"
                ],
                value="google/medgemma-27b-it"
            )
            image_input = gr.Image(type="pil", label="Upload Image (for new chats only)")
            instructions = gr.Textbox(label="System Instructions (for text-only chats)", value=TEXT_PROMPT, lines=3)
            temperature = gr.Slider(0.0, 2.0, value=0.0, step=0.1, label="Temperature")
            max_tokens = gr.Slider(100, 65535, value=65535, step=256, label="Max New Tokens")

    inputs = [msg, image_input, chatbot, is_image_session, model_selector, instructions, temperature, max_tokens]
    outputs = [chatbot, is_image_session, msg, download_btn, image_input]

    e_submit = msg.submit(chat_with_openai, inputs, outputs)
    e_click = send_btn.click(chat_with_openai, inputs, outputs)
    
    stop_btn.click(fn=lambda: None, cancels=[e_submit, e_click], queue=False)

    def clear_all():
        return [], False, gr.update(value=None, interactive=True), "", gr.update(visible=False)

    clear_btn.click(
        fn=clear_all, 
        inputs=None, 
        outputs=[chatbot, is_image_session, image_input, msg, download_btn], 
        cancels=[e_submit, e_click],
        queue=False
    )
    
demo.queue()

if __name__ == "__main__":
    print("Launching Gradio interface... Press Ctrl+C to exit.")
    print("Temporary files for this session will be cleaned up automatically on exit.")
    demo.launch()
