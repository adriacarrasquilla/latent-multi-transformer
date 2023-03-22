import gradio as gr
from torch.utils import data
from attr_dict import NUM_TO_ATTR
from constants import ATTR_TO_NUM
from datasets import LatentDataset
from itertools import islice
import torch
import yaml
import numpy as np

from trainer import Trainer

from constants import LABEL_FILE, LATENT_PATH, LABEL_FILE, DEVICE, LOG_DIR, STYLEGAN, CLASSIFIER

experiment = "limit"

# Load basic config
config = yaml.safe_load(open('./configs/' + experiment + '.yaml', 'r'))
attrs = config['attr'].split(',')
attr_num = [ATTR_TO_NUM[a] for a in attrs]

# Init trainer
trainer = Trainer(config, attr_num, attrs, LABEL_FILE)
trainer.initialize(STYLEGAN, CLASSIFIER)   
trainer.load_model_multi(LOG_DIR + experiment, "10_attrs")
trainer.to(DEVICE)

# Load data
dataset = LatentDataset(LATENT_PATH, LABEL_FILE, training_set=True)
main_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)

ATTR_IDXS = {}

for n_iter, list_A in enumerate(main_loader):
    w_A, lbl_A = list_A
    for i, l in enumerate(lbl_A[0]):
        if l.item() == 1:
            attr = NUM_TO_ATTR[i]
            if attr in ATTR_IDXS:
                ATTR_IDXS[attr].append(n_iter)
            else:
                ATTR_IDXS[attr] = [n_iter]


# Filter out positive and negative attributes
TOTAL_ATTRS = []
for a in ATTR_TO_NUM:
    TOTAL_ATTRS.append(a)
    TOTAL_ATTRS.append(f"not {a}")


def filter_dataloader(filters):
    if filters:
        if filters[0][:3] == "not":
            all_samples_set = set(range(len(main_loader)))
            idx = list(all_samples_set - set(ATTR_IDXS[filters[0][4:]]))
        else:
            idx = ATTR_IDXS[filters[0]]

        for f in filters[1:]:
            if f[:3] == "not":
                idx = list(set(idx) - set(ATTR_IDXS[f[4:]]))
            else:
                idx = list(set(idx) & set(ATTR_IDXS[f]))

        subset = data.Subset(dataset, idx)
    else:
        subset = dataset

    loader = data.DataLoader(subset, batch_size=1, shuffle=False)

    return gr.Slider.update(maximum=len(loader)-1), loader

def update_image(image_number, loader):
    sample = next(islice(loader, int(image_number), None))[0].to(DEVICE)
    classification = trainer.get_classification(sample)
    result = trainer.get_original_image(sample).numpy()
    result = np.transpose(result, (1, 2, 0))
    return gr.Image.update(value=result), gr.State(value=sample), classification

def transform_image(sample, *sliders):
    org_size = sample.value.size()
    sample = sample.value.view(sample.value.size(0), -1)
    coeffs = [coeff for coeff in sliders]
    
    total_loss = trainer.compute_eval_loss(sample, torch.tensor(coeffs).to(DEVICE)).item()
    loss_pb = trainer.loss_pb.item()
    loss_recon = trainer.loss_recon.item()
    loss_reg = trainer.loss_reg.item()

    # w = trainer.T_net(sample, torch.tensor(coeffs).to(DEVICE))
    w = trainer.w_1
    classification = trainer.get_classification(w)
    w = w.view(org_size)

    result = trainer.get_original_image(w).numpy()
    result = np.transpose(result, (1, 2, 0))

    # print(total_loss, loss_pb, loss_reg, loss_recon)
    # print(classification)

    return (
        gr.Image.update(value=result),
        gr.Number.update(value=total_loss),
        gr.Number.update(value=loss_pb),
        gr.Number.update(value=loss_recon),
        gr.Number.update(value=loss_reg),
        classification
    )

def update_dataframe(headers, org, pred):
    idx = [ATTR_TO_NUM[a] for a in headers]
    org = ["Orgininal"] + [f"{org[i]:.3f}" for i in idx]
    pred = ["Transformed"] + [f"{pred[i]:.3f}" for i in idx]
    return gr.DataFrame.update([[" "] + headers, org, pred], visible=True), gr.Column.update(visible=True)

def merge_sliders(s1, s2):
    sliders = []
    for x, y in zip(s1, s2):
        sliders.append(x)
        sliders.append(y)

    if len(s1) > len(s2):
        sliders += s1[len(s2):]
    elif len(s2) > len(s1):
        sliders += s2[len(s1):]

    return sliders


with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Face editing
    Select image id and intensity of the attributes to transform
    """
    )

    # Helper states
    sample = gr.State()
    class_org = gr.State()
    class_pred = gr.State()
    losses = gr.State()
    loader = gr.State(value=main_loader)

    with gr.Column():
        with gr.Row():
            # image_number = gr.Number(value=None, label="Image ID", interactive=True)
            image_number = gr.Slider(minimum=0, maximum=len(loader.value)-1, value=0, step=1, label="Image ID", interactive=True)
            filters = gr.Dropdown(
                TOTAL_ATTRS,
                interactive=True,
                multiselect=True,
                label="Filter out attributes in samples",
            )

        with gr.Row():
            with gr.Column():
                sliders_even = [gr.Slider(-3, 3, 0, label=attr) for attr in attrs[0::2]]
            with gr.Column():
                sliders_odd = [gr.Slider(-3, 3, 0, label=attr) for attr in attrs[1::2]]

        sliders = merge_sliders(sliders_even, sliders_odd)

    with gr.Row():
        photo = gr.Image(value=None, label="Image", interactive=False)
        photo.style(height=512,width=512)
        out_image = gr.Image(value=None, label="Output", interactive=False)
        out_image.style(height=512,width=512)

    generate_btn = gr.Button("Generate")

    with gr.Column(visible=False) as result_col:

        with gr.Row():
            total_loss = gr.Number(value=0, precision=3, label="total_loss")
            loss_pb = gr.Number(value=0, precision=3, label="loss_pb")
            loss_recon = gr.Number(value=0, precision=3, label="loss_recon")
            loss_reg = gr.Number(value=0, precision=3, label="loss_reg")

        attributes = gr.Dropdown(
            [attribute for attribute in ATTR_TO_NUM],
            value=attrs,
            interactive=True,
            multiselect=True,
            label="Filter out attributes to check their classification",
        )

        attribute_df = gr.DataFrame([[" "] + [attribute for attribute in attributes.value],
                                     ["Original"] + [0.0000 for _ in attributes.value],
                                     ["Transformed"] + [0.0000 for _ in attributes.value]], visible=False)

    # Updating events
    image_number.change(update_image, [image_number, loader], [photo, sample, class_org])
    attributes.change(update_dataframe, [attributes, class_org, class_pred], [attribute_df, result_col])
    filters.change(filter_dataloader, filters, [image_number, loader])

    generate_btn.click(
        transform_image,
        inputs=[sample] + sliders,
        outputs=[out_image, total_loss, loss_pb, loss_recon,
                 loss_reg, class_pred]).then(update_dataframe, [attributes, class_org, class_pred], [attribute_df, result_col])

if __name__ == "__main__":
    demo.launch()
