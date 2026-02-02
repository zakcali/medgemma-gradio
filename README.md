# Multimodal Medical Chatbot Interface

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)![License](https://img.shields.io/badge/License-MIT-green.svg)

A user-friendly Gradio web interface for interacting with locally hosted, OpenAI-compatible medical Gemma large language models from Google. This application supports both text-to-text and image-to-text (multimodal) conversations, making it a powerful tool for medical AI research and experimentation.



## Features

-   **Multimodal Chat:** Seamlessly switch between text-only questions and image-based analysis.
-   **Dynamic System Prompts:** Automatically uses a specialized `IMAGE_PROMPT` for image analysis and a configurable `TEXT_PROMPT` for general medical questions.
-   **Model Selection:** Easily switch between different models loaded on your backend via a dropdown menu.
-   **Real-time Streaming:** Responses are streamed token-by-token for an interactive experience.
-   **Parameter Control:** Adjust key inference parameters like `Temperature` and `Max New Tokens` directly in the UI.
-   **Robust Session Management:**
    -   Chat history is maintained for conversational context.
    -   Image uploads are intelligently disabled mid-conversation to ensure a clean workflow.
    -   Includes "Stop" and "New Chat" controls.
-   **Download Conversation:** Save the model's last complete response as a Markdown file.
-   **Automatic Cleanup:** Temporary files created for downloads are automatically cleaned up when the application exits.

## System & Hardware Requirements

This application is a **frontend client**. It requires a separate, powerful backend server to host and run the language models. The setup described here is tailored for a high-performance local machine.

### Hardware

This setup is designed to run on a local machine with significant GPU resources:
-   **GPU:** 4x NVIDIA RTX 3090 (24 GB VRAM each)
-   **Total VRAM:** 96 GB

### Backend: vLLM Server

The Gradio interface connects to a local server powered by [vLLM](https://github.com/vllm-project/vllm), which provides an OpenAI-compatible API endpoint.

-   **Model:** The primary model for this setup is `google/medgemma-27b-it`.
-   **Model Source:** [huggingface.co/google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it)

To serve this model across all four GPUs, you will run the following command in your terminal:

```bash
# Ensure you have uv or another package manager to run vLLM
uv run vllm serve google/medgemma-27b-it --tensor-parallel-size 4 --async-scheduling
```
-   `--tensor-parallel-size 4`: This critical flag splits the model's weights across the 4 GPUs, allowing you to run a model that wouldn't fit on a single card.
-   `--async-scheduling`: Enables asynchronous scheduling to improve throughput.

### Frontend: Python Dependencies

The `medgemma-gradio.py` script requires the following Python libraries:

-   `gradio`
-   `openai`
-   `pillow`

## Installation and Setup

Follow these steps to get the complete application running. I prefer uv package installer and resolver

### 1. Set Up the Environment

First, clone this repository and create a Python virtual environment.

```bash
git clone https://github.com/zakcali/medgemma-gradio.git
cd medgemma-gradio
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 2. Install Dependencies

Install the required Python packages for both the vLLM server and the Gradio client.

```bash
# For the vLLM server
uv pip install vllm

# For the Gradio frontend
uv pip install gradio openai pillow
```

### 3. Launch the vLLM Backend Server

Open a new terminal window, activate your virtual environment, and start the vLLM server with the command specified above.

```bash
uv run vllm serve google/medgemma-27b-it --tensor-parallel-size 4 --async-scheduling
```
If you got `torch.OutOfMemoryError: CUDA out of memory` error, you may start the vLLM server with the command specified above.

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True uv run vllm serve google/medgemma-27b-it --tensor-parallel-size 4 --async-scheduling --max-model-len 65536 --max-num-seqs 8
```

Leave this terminal running. It will handle all the model inference requests. By default, it will be available at `http://localhost:8000`.

### 4. Launch the Gradio Frontend

Open a second terminal window, navigate to the repository directory, and run the Python script.

```bash
uv run medgemma-gradio.py
```

## Running the Application

You can launch the application in several ways depending on your needs.

### To run locally on your machine:
Use the following line in the script:
```python
demo.launch()
```

### To serve to other devices on your local network (LAN):
Modify the last line to include your machine's local IP address.
```python
# Replace "192.168.0.xx" with your actual LAN IP address
demo.launch(server_name="192.168.0.xx", server_port=7860)
```

### To share publicly over the internet:
Set the `share` parameter to `True`. Gradio will generate a temporary public URL for you.
```python
demo.launch(share=True)
```
For more details, see the official Gradio guide on [Sharing Your App](https://www.gradio.app/guides/sharing-your-app).

## How to Use

1.  **Select a Model:** Choose your desired model from the "Select Model" dropdown.
2.  **Ask a Question:**
    -   **For a text-based question:** Simply type your message in the textbox and press "Send".
    -   **For an image analysis:** At the **start of a new chat**, click the "Upload Image" box, select your image, and then type an accompanying prompt (e.g., "What do you see in this chest x-ray?").
3.  **Adjust Parameters:** Use the sliders to control the `Temperature` and `Max New Tokens` for the model's response.
4.  **Manage the Conversation:**
    -   Click **"⏹️ Stop"** to interrupt a response while it is being generated.
    -   Click **"🗑️ New Chat"** to clear the conversation history and start fresh. This will also re-enable the image upload component.
    -   Click **"⬇️ Download Last Response"** to save the model's output.

## The examples for MedGemma models in the Hugging Face Transformers library are not very useful.

### The `transformers` Library (`model.generate`) - The "Research" Approach

The standard Hugging Face `transformers` library is a phenomenal tool, but it's primarily designed for research, fine-tuning, and straightforward, single-batch inference.

When you call `model.generate()`, it suffers from several performance bottlenecks in a server context:

1.  **Inefficient KV Cache Management:** During generation, the model must store a "KV cache" (the attention keys and values) for all previously generated tokens. Standard `transformers` pre-allocates a large, contiguous block of VRAM for the *maximum possible sequence length* for *every single request* in a batch. This leads to massive VRAM waste, as most sequences are much shorter than the maximum.
2.  **Static Batching:** If you send a batch of requests (e.g., 4 users at once), the entire batch must wait until the *slowest* generation is complete before the batch is finished and the GPU can move on. This leaves the GPU idle and is highly inefficient.
3.  **Concurrency is Hard:** It is not designed to be a multi-user server. To handle simultaneous requests, you would have to build a complex and often inefficient queuing and batching system yourself.

### vLLM - The "Production Inference" Approach

vLLM is a specialized inference server built from the ground up to solve the problems listed above. It's designed for high-performance, high-throughput serving of LLMs.

Here are its key advantages, which directly translate to the speed you will experience:

1.  **PagedAttention™:** This is vLLM's core innovation. Instead of allocating a huge contiguous block of VRAM for the KV cache, it manages the cache in smaller, non-contiguous blocks, much like how an operating system manages RAM with virtual memory "pages."
    *   **Result:** Almost no wasted VRAM. This allows vLLM to pack far more requests onto the GPU at once, dramatically increasing **throughput** (the number of requests you can serve over time).
2.  **Continuous Batching:** vLLM doesn't use static batches. It has a dynamic system where as soon as one request in the batch finishes generating, it's removed, and a new request from the queue is immediately added.
    *   **Result:** The GPU is kept busy almost 100% of the time. This significantly reduces average **latency** (the time a user waits for a response) and boosts throughput.
3.  **Optimized CUDA Kernels:** vLLM uses highly optimized, custom-written GPU code (kernels), including technologies like FlashAttention, to make the fundamental matrix multiplications of the model run much faster than the standard PyTorch implementations.

---

### Analogy: A Restaurant

*   **`transformers` (`model.generate`) is like a restaurant with only reservations.** A table of four must arrive together, and they cannot leave until every single person has finished their entire three-course meal. If one person is a very slow eater, the other three are stuck waiting, and the table cannot be used by anyone else.
*   **vLLM is like a high-efficiency food court.** The moment a seat opens up at any table, the next person in line is immediately seated. People come and go as they please, ensuring the seats (your GPU) are always being used to serve customers.

### Summary Table

| Feature | Direct `transformers` (`model.generate`) | vLLM + OpenAI API |
| :--- | :--- | :--- |
| **KV Cache** | Inefficient (Large, contiguous blocks) | Highly Efficient (**PagedAttention**) |
| **Batching** | Static (Waits for the slowest request) | Continuous (Dynamic, no waiting) |
| **GPU Utilization** | Low to Medium | Very High (Close to 100%) |
| **Throughput** | **Low** | **Extremely High (up to 24x higher)** |
| **Best Use Case** | Research, single-user scripts, fine-tuning | **Production servers, web apps, APIs** |

**Conclusion:** Using vLLM as the backend for your Gradio application is the correct one. You are leveraging a production-grade inference engine. The API call from your Python script will be a lightweight network request to a hyper-optimized server that is running the model far more efficiently than a direct `transformers` implementation ever could in a multi-request scenario.

### Migration to Gradio 6.0 interface
`medgemma-gradio.py` file migrated to **Gradio 6.0**:

### The Chatbot Component
**Change:** Removed `type="messages"` and replaced `show_copy_button=True` with `buttons=["copy"]`.

```python
# OLD
chatbot = gr.Chatbot(height=500, type="messages", show_copy_button=True, label="Conversation")

# NEW
chatbot = gr.Chatbot(height=500, buttons=["copy"], label="Conversation")
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
