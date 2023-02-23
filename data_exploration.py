import torch
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import *
from nets import *

from attr_dict import NUM_TO_ATTR

device = torch.device('cuda')

LatentClassifier = LCNet([9216, 2048, 512, 40], activ='leakyrelu')
LatentClassifier.load_state_dict(torch.load('./models/latent_classifier_epoch_20.pth'))
LatentClassifier.eval()
LatentClassifier.to(device)

dataset_A = LatentDataset('./data/celebahq_dlatents_psp.npy', './data/celebahq_anno.npy', training_set=True)
loader_A = data.DataLoader(dataset_A, batch_size=1, shuffle=True)

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
