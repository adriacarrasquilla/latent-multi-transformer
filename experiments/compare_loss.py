import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

def parse_tensorboard(path):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    # assert all(
    #     s in ea.Tags()["scalars"] for s in scalars
    # ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in ea.Tags()["scalars"]}

scalars = parse_tensorboard("../logs/limit_scaled")

i=1
j=0
losses = [[], [], [], []]
for key, values in scalars.items():
    mean = values["value"].tail(50).mean()
    losses[j].append(mean)
    j += 1
    if j%4 == 0:
        i += 1
        j = 0


loss_titles = ["class", "latent_recognition", "attr_regr", "total"]
fig, ax = plt.subplots(2, 2)
fig.suptitle('Evolution of loss when increasing number of attributes')

i = j = 0
for idx, loss in enumerate(losses):
    ax[i,j].plot(range(len(loss)), loss)
    ax[i,j].set_title(loss_titles[idx])
    ax[i,j].set_xticks(range(len(loss)))
    ax[i,j].set_xlabel("n attrs")
    ax[i,j].set_ylabel("loss value")
    i += 1
    if i == 2:
        j +=1
        i = 0
        
plt.show()
