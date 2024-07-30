import gradio as gr
import os
import torch

from model import create_effnetb0_model
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
class_names = ["cat", "dog"]

# Create EffNetB2 model
effnetb0, effnetb0_transforms = create_effnetb0_model(
    num_classes=2,
)

# Load saved weights
effnetb0.load_state_dict(
    torch.load(
        f="best.pth",
        map_location=torch.device("cpu"),  # load to CPU
    )
)

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    # Start the timer
    start_time = timer()

    # Transform the target image and add a batch dimension
    img = effnetb0_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    effnetb0.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb0(img), dim=1)

    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


### 4. Gradio app ###

# Create title, description and article strings
title = "Dog and Cat Classification 🍕🥩🍣"
description = "An EfficientNetB0 feature extractor computer vision model to classify images of dogs and cats."
article = "Created at 30/7/2024.\n By BlackRose."

# Create examples list from "examples/" directory
example_list = [

]

# Create the Gradio demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=2, label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch(debug=False,
            share=True)