import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from datasets import *
from nets import *

from attr_dict import NUM_TO_ATTR

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

LatentClassifier = LCNet([9216, 2048, 512, 40], activ='leakyrelu')
LatentClassifier.load_state_dict(torch.load('./models/latent_classifier_epoch_20.pth', map_location=device))
LatentClassifier.eval()
LatentClassifier.to(device)

dataset_A = LatentDataset('./data/celebahq_dlatents_psp.npy', './data/celebahq_anno.npy', training_set=True)
loader_A = data.DataLoader(dataset_A, batch_size=1, shuffle=True)


def classifier_distribution():
    run = False
    summary = {}

    if run:
        with torch.no_grad():
            for n_iter, list_A in enumerate(loader_A):
                w_A, lbl_A = list_A
                w_A, lbl_A = w_A.to(device), lbl_A.to(device)
                predict_lbl_0 = LatentClassifier(w_A.view(w_A.size(0), -1))
                lbl_0 = torch.sigmoid(predict_lbl_0)
                attr = NUM_TO_ATTR[torch.argmax(lbl_0, axis=1).item()]
                if attr in summary:
                    summary[attr] += 1
                else:
                    summary[attr] = 1
    else:
        # Hardcoded
        summary = {'No_Beard': 14792, 'Male': 7788, 'Young': 890, 'Smiling': 1885, 'Mouth_Slightly_Open': 998,
                   'Eyeglasses': 205, 'Wavy_Hair': 124, 'Wearing_Hat': 61, 'Bangs': 119, 'Blond_Hair': 7,
                   'Wearing_Lipstick': 31, 'Black_Hair': 44, 'Gray_Hair': 5, 'Big_Nose': 18, 'Heavy_Makeup': 7,
                   'High_Cheekbones': 5, 'Wearing_Earrings': 10, 'Bushy_Eyebrows': 2, 'Bags_Under_Eyes': 3,
                   'Brown_Hair': 4, 'Arched_Eyebrows': 2}
                    

    x = np.char.array(list(summary.keys()))
    y = np.array(list(summary.values()))

    patches, texts = plt.pie(y, startangle=90, radius=1.2, colors=sns.color_palette("Set2"), wedgeprops=dict(edgecolor='#282828'))
    labels = ['{1:1.2f} % - {0}'.format(i,j) for i,j in zip(x, 100.*y/y.sum())]

    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y), key=lambda x: x[2], reverse=True))
    plt.legend(patches, labels, loc='right', fontsize=12, bbox_to_anchor=(-0.2,0.5))

    plt.show()


def data_distribution():
    run = False
    summary = {}
    if run:
        with torch.no_grad():
            for n_iter, list_A in enumerate(loader_A):
                w_A, lbl_A = list_A
                for i, l in enumerate(lbl_A[0]):
                    if l.item() == 1:
                        attr = NUM_TO_ATTR[i]
                        if attr in summary:
                            summary[attr] += 1
                        else:
                            summary[attr] = 1
    else:
        # Hardcoded
        summary = {'Arched_Eyebrows': 10015, 'Heavy_Makeup': 12381, 'No_Beard': 21946, 'Wearing_Earrings': 7274,
                   'Wearing_Lipstick': 15292, 'Bags_Under_Eyes': 7939, 'Big_Nose': 8875, 'Chubby': 1912,
                   'Double_Chin': 1641, 'Eyeglasses': 1287, 'Gray_Hair': 1128, 'Male': 9793,
                   'Mouth_Slightly_Open': 12733, 'Mustache': 1573, 'Receding_Hairline': 2292, 'Smiling': 12706,
                   'Wearing_Necktie': 1913, 'Attractive': 15411, 'Brown_Hair': 6224, 'High_Cheekbones': 12489,
                   'Young': 20973, '5_o_Clock_Shadow': 4023, 'Black_Hair': 5892, 'Bushy_Eyebrows': 5080,
                   'Goatee': 2053, 'Pointy_Nose': 8639, 'Sideburns': 2167, 'Big_Lips': 9978,
                   'Wavy_Hair': 9685, 'Straight_Hair': 5773, 'Bangs': 4940, 'Bald': 642,
                   'Rosy_Cheeks': 3098, 'Oval_Face': 5349, 'Blond_Hair': 4670, 'Narrow_Eyes': 3203,
                   'Wearing_Necklace': 4676, 'Wearing_Hat': 957, 'Pale_Skin': 1372, 'Blurry': 107}


    # data from https://allisonhorst.github.io/palmerpenguins/
    total_samples = len(loader_A)
    species = [str(k) for k in summary.keys()]
    weight_counts = {
        "Present": np.array([i / total_samples for i in summary.values()])*100,
        "Not Present": np.array([(total_samples - i) / total_samples for i in summary.values()])*100,
    }
    colors = ["#8cb369", "#f4a259"]

    width = 0.5

    fig, ax = plt.subplots(figsize=(15,7))
    bottom = np.zeros(len(summary))

    for (boolean, weight_count), color in zip(weight_counts.items(), colors):
        p = ax.bar(species, weight_count, width, label=boolean, bottom=bottom, color=color, edgecolor="black", linewidth=1)
        bottom += weight_count

    ax.set_title("Attribute distribution in CelebA dataset")
    ax.legend(loc="upper right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.savefig("test.png", bbox_inches="tight")

data_distribution()
