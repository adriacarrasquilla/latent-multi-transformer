import gradio as gr
from torch.utils import data
from attr_dict import ATTR_TO_NUM
from datasets import LatentDataset
from itertools import islice
import torch
import yaml
import numpy as np

from trainer import Trainer

LABEL_FILE = './data/celebahq_anno.npy'
LATENT_PATH = './data/celebahq_dlatents_psp.npy'

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

dataset = LatentDataset(LATENT_PATH, LABEL_FILE, training_set=True)
loader = data.DataLoader(dataset, batch_size=1, shuffle=False)


def update_image(image_number):
    sample = next(islice(loader, int(image_number), None))[0].to(DEVICE)
    result = trainer.get_original_image(sample).numpy()
    result = np.transpose(result, (1, 2, 0))
    return gr.Image.update(value=result), gr.State(value=sample)

def transform_image(sample, *sliders):
    org_size = sample.value.size()
    sample = sample.value.view(sample.value.size(0), -1)
    coeffs = [[coeff for coeff in sliders]]
    w = trainer.T_net(sample, torch.tensor(coeffs).to(DEVICE))
    w = w.view(org_size)
    result = trainer.get_original_image(w).numpy()
    result = np.transpose(result, (1, 2, 0))
    return gr.Image.update(value=result)


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Face editing
    Select image id and intensity of the attributes to transform
    """
    )

    sample = gr.State()

    with gr.Row():
        with gr.Column():
            image_number = gr.Number(value=None, label="Image ID", interactive=True)

        with gr.Column():
            sliders = [gr.Slider(-1.5, 1.5, 0, label=attr) for attr in attrs]

    with gr.Row():
        photo = gr.Image(value=None, label="Image", interactive=False)
        photo.style(height=512,width=512)
        output = gr.Image(value=None, label="Output", interactive=False)
        output.style(height=512,width=512)

    generate_btn = gr.Button("Generate")

    image_number.change(update_image, image_number, [photo, sample])

    generate_btn.click(transform_image, [sample] + sliders, output)


if __name__ == "__main__":
    demo.launch()
