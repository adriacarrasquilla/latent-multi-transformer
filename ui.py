import gradio as gr
import yaml
import torch
import os

from attr_dict import ATTR_TO_NUM
from trainer import Trainer

# Some constants
LABEL_FILE = './data/celebahq_anno.npy'
DEVICE = torch.device('cuda')
LOG_DIR = './logs/reduction'
STYLEGAN = './pixel2style2pixel/pretrained_models/psp_ffhq_encode.pt'
CLASSIFIER = './models/latent_classifier_epoch_20.pth'


# Load basic config
config = yaml.safe_load(open('./configs/' + "reduction" + '.yaml', 'r'))
attrs = config['attr'].split(',')
attr_num = [ATTR_TO_NUM[a] for a in attrs]

# Init trainer
trainer = Trainer(config, attr_num, attrs, LABEL_FILE)
trainer.initialize(STYLEGAN, CLASSIFIER)   
trainer.load_model_multi(LOG_DIR)
trainer.to(DEVICE)


# dummy predict placeholder
def predict(*sliders):
    name = ""
    for s in sliders:
        name += str(s)

    return [f"Loss_{i} = {s}" for i, s in enumerate(sliders)]

sliders = [gr.Slider(-1, 1, 0, label=attr) for attr in attrs]
demo = gr.Interface(
    fn=predict,
    inputs=sliders,
    outputs=["text" for _ in range(len(trainer.attrs))]
)

demo.launch()
