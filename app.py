import os
import gradio as gr
import requests
from dotenv import load_dotenv
from rag_utils import load_and_index_docs, get_top_k_context

load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
print("📚 Indexing docs...")
chunks, index, embeddings = load_and_index_docs()

# Call the Model to get a response.
def ask_llm(prompt):
    try:
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "mistralai/Mistral-7B-Instruct-v0.1",
            "prompt": f"[INST] You're an expert Gradio developer. {prompt} [/INST]",
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9
        }

        response = requests.post("https://api.together.xyz/v1/completions", headers=headers, json=payload)

        if response.status_code != 200:
            return f"API Error: {response.status_code} - {response.text}"

        return response.json()['choices'][0]['text'].strip()

    except Exception as e:
        return f"Error: {str(e)}"

# Image classifier **** WORK ON AND TEST MORE ***
def predict_image(img):
    return "This would be classified here — add a model to go deeper!"

# Gradio UI
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("Gradio Playground Clone")
    gr.Markdown("Ask anything about Gradio code — get explanations or code via Together.ai + RAG.\n\nTry prompts like:\n- `How do I create a chatbot?`\n- `Explain gr.Interface`")

    with gr.Row():
        mode = gr.Radio(["Generate Code", "Explain Code"], label="Mode", value="Generate Code")

    user_input = gr.Textbox(label="Prompt", placeholder="Ask about Gradio...", lines=2)
    output = gr.Code(label="Response")

    def run_with_mode(user_query, selected_mode):
        prefix = "Generate Gradio code for: " if selected_mode == "Generate Code" else "Explain this Gradio code: "
        context_chunks = get_top_k_context(user_query, chunks, index, embeddings)
        context = "\n\n".join(context_chunks)
        full_prompt = f"""Use the following context to answer:
{context}

{prefix} {user_query}
"""
        return ask_llm(full_prompt)

    btn = gr.Button("Run")
    btn.click(fn=run_with_mode, inputs=[user_input, mode], outputs=output)

    gr.Markdown("---")
    gr.Markdown("Powered by Mistral on [Together.ai](https://together.ai) — built by [Darren Chambers](https://github.com/darrencc1)")

    with gr.Accordion("Image Classifier Demo", open=False):
        image_input = gr.Image(type="filepath")
        label_output = gr.Textbox(label="Label")
        image_btn = gr.Button("Classify Image")
        image_btn.click(fn=predict_image, inputs=image_input, outputs=label_output)

print("✅ App is starting...")
demo.launch()
