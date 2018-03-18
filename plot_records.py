import pandas as pd
import matplotlib.pyplot as plt

records_file = "outputs/SegNet_classweight_spacenet_records.csv"
output_file ="training_graphs/training_SegNet_spacenet_classweight.png"
data = pd.read_csv(records_file)

# Set nice plt parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12

# Load data
acc = data["acc"]
loss = data["loss"]
mean_IU = data["mean_IU"]
mean_accuracy = data["mean_accuracy"]
pixel_accuracy = data["pixel_accuracy"]

val_acc = data["val_acc"]
val_loss = data["val_loss"]
val_mean_IU = data["val_mean_IU"]
val_mean_accuracy = data["val_mean_accuracy"]
val_pixel_accuracy = data["val_pixel_accuracy"]

# Summarize history for classical accuracy
fig, axarr = plt.subplots(2,2, sharex=True)
axarr[0,0].plot(acc)
axarr[0,0].plot(val_acc)
axarr[0,0].set_title('Classical accuracy evolution', fontstyle='italic', fontsize='11')
axarr[0,0].set_ylabel('classical accuracy')
axarr[0,0].set_xlabel('epoch')
axarr[0,0].legend(['train', 'test'], loc='best')

# summarize history for mean_IU metric
axarr[0,1].plot(mean_IU)
axarr[0,1].plot(val_mean_IU)
axarr[0,1].set_title('Mean IU evolution', fontstyle='italic', fontsize='11')
axarr[0,1].set_ylabel('mean IU')
axarr[0,1].set_xlabel('epoch')
axarr[0,1].legend(['train', 'test'], loc='best')

# summarize history for pixel accuracy metric
axarr[1,0].plot(mean_accuracy)
axarr[1,0].plot(val_mean_accuracy)
axarr[1,0].set_title('Mean accuracy evolution', fontstyle='italic', fontsize='11')
axarr[1,0].set_ylabel('mean accuracy')
axarr[1,0].set_xlabel('epoch')
axarr[1,0].legend(['train', 'test'], loc='best')

# summarize history for loss
axarr[1,1].plot(loss)
axarr[1,1].plot(val_loss)
axarr[1,1].set_title('Loss evolution', fontstyle='italic', fontsize='11')
axarr[1,1].set_ylabel('loss')
axarr[1,1].set_xlabel('epoch')
axarr[1,1].legend(['train', 'test'], loc='best')

plt.subplots_adjust(hspace=0.35, wspace=0.40)
fig.suptitle("Metrics evolution during training")

#fig.tight_layout()
fig.savefig(output_file, bbox_inches='tight')

print("Training plot saved :) !")
