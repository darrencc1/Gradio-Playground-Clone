# How to Use Gradio Interface

You can use `gr.Interface(fn, inputs, outputs)` to quickly create GUIs.

### Example:
```python
import gradio as gr

def greet(name):
    return f"Hello {name}!"

gr.Interface(fn=greet, inputs="text", outputs="text").launch()
```

# How to make an image classifier

### Example:

```python
gr.Interface(fn=predict, inputs="image", outputs="label").launch()

```

# Markdown Display

### Example 
```python
gr.Markdown("### Hello **World**")
```

# Radio Button 
### Example
```python
gr.Radio(["Option A", "Option B"], label="Choose one")
```

# Blocks Layout For Custom UIs
### Example
```python
with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Greeting")
    btn = gr.Button("Greet")

    def greet(name):
        return f"Hello, {name}!"

    btn.click(fn=greet, inputs=name, outputs=output)

demo.launch()
```


