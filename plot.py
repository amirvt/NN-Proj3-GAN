import pickle
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

def load_object(filename):
    with open(filename, 'rb') as output:  # Overwrites any existing file.
        return pickle.load(output)


MODEL_NAME = 'DC'
DATA_SET_NAME = 'CIFAR10'

directory = "images-" + DATA_SET_NAME + "-" + MODEL_NAME
hists = load_object(directory + "/history.pkl")
hists = list(map(list, zip(*hists)))
# %%
plt.clf()
plt.plot(range(len(hists[0])), hists[0])
plt.plot(range(len(hists[2])), hists[2])
plt.legend(["Discriminator","Generator"])
plt.xlabel("Epoch")
plt.ylabel("Loss (negative log)")
plt.title("Loss for CIFAR10 data set")
plt.savefig("plots/loss-cifar.png", bbox="tight")

#%%
plt.clf()
plt.plot(range(len(hists[1])), hists[1])
plt.plot(range(len(hists[3])), hists[3])
plt.legend(["Discriminator", "Generator"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy for CIFAR10 data set")
plt.savefig("plots/acc-cifar.png", bbox="tight")
