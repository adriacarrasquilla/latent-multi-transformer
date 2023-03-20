import gradio as gr
from torch.utils import data
from constants import ATTR_TO_NUM
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



with gr.Blocks() as demo:
    sample = gr.State(value=None)

    def update_image(image_number):
        sample = next(islice(loader, int(image_number), None))[0].to(DEVICE)
        result = trainer.get_original_image(sample).numpy()
        result = np.transpose(result, (1, 2, 0))
        return gr.Image.update(value=result), gr.State(value=sample)

    def transform_image(image_number, sample):
        # sample = next(islice(loader, int(image_number), None))[0].to(DEVICE)
        w = trainer.T_net(sample.value.view(sample.value.size(0), -1), torch.tensor([[1.0,1.0]]).to(DEVICE))
        size = w.size()
        w = w.view(size)
        result = trainer.get_original_image(w).numpy()
        result = np.transpose(result, (1, 2, 0))
        return gr.Image.update(value=result)

    gr.Markdown(
        """
    # Animal Generator
    Once you select a species, the detail panel should be visible.
    """
    )

    with gr.Row():
        with gr.Column():
            species = gr.Radio(label="Animal Class", choices=["Mammal", "Fish", "Bird"])
            animal = gr.Dropdown(label="Animal", choices=[])

            image_number = gr.Slider(minimum=0, maximum=len(loader), label="Image ID", interactive=True)
            photo = gr.Image(value=None, label="Image", interactive=False)
            photo.style(height=512,width=512)

        with gr.Column(visible=True) as details_col:
            weight = gr.Slider(0, 20)
            details = gr.Textbox(label="Extra Details")
            generate_btn = gr.Button("Generate")
            output = gr.Image(value=None, label="Output", interactive=False)
            output.style(height=512,width=512)

    species_map = {
        "Mammal": ["Elephant", "Giraffe", "Hamster"],
        "Fish": ["Shark", "Salmon", "Tuna"],
        "Bird": ["Chicken", "Eagle", "Hawk"],
    }

    def filter_species(species):
        return gr.Dropdown.update(
            choices=species_map[species], value=species_map[species][1]
        ), gr.update(visible=True)

    species.change(filter_species, species, [animal, details_col])
    image_number.change(update_image, image_number, [photo, sample])

    def filter_weight(animal):
        if animal in ("Elephant", "Shark", "Giraffe"):
            return gr.update(maximum=100)
        else:
            return gr.update(maximum=20)

    animal.change(filter_weight, animal, weight)
    weight.change(lambda w: gr.update(lines=int(w / 10) + 1), weight, details)

    generate_btn.click(transform_image, [image_number, sample], output)


if __name__ == "__main__":
    demo.launch()
