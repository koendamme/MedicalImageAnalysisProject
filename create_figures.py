import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

path = os.path.join("parameter_tuning", "learning_rate")

directories = np.array(next(os.walk(path))[1])

loss_curves = {}

for function in directories:
    loss_curves[function] = np.zeros((20, 5))

    if function == "DiceCE" or function == "DiceFocal":
        file_paths = glob.glob(os.path.join(path, function, "dice val", "*.csv"))
    else:
        file_paths = glob.glob(os.path.join(path, function, "val", "*.csv"))

    for i, file_path in enumerate(file_paths):
        curve_values = pd.read_csv(file_path).iloc[:, 1].to_numpy()

        for j in range(len(curve_values)):
            loss_curves[function][j, i] = curve_values[j]

plt.title("Learning rate selection")
plt.xlabel("Epoch")
plt.ylabel("Dice Focal Loss")
plt.plot(np.mean(loss_curves["1"], axis=1), label="0.001")
plt.plot(np.mean(loss_curves["5"], axis=1), label="0.0005")
plt.plot(np.mean(loss_curves["8"], axis=1), label="0.0008")
plt.plot(np.mean(loss_curves["12"], axis=1), label="0.0012")
plt.xticks(np.arange(20, step=2))
plt.legend()
plt.savefig("figures/learning rate selection.png")
plt.show()

