import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = 'experiments\EDSR_CADRotated_wFlaws_REC_FDK_Al_noBH_X3binning_NoNoise_NoBlur_1066_Views_fullscan\loss.npy'
VAL_FREQ = 500

PLOT_EPOCH = True
ITER_PER_EPOCH = 4725

loss = np.load(FILE_PATH, allow_pickle=True)

plt.plot(np.arange(0, VAL_FREQ * len(loss), VAL_FREQ, dtype=int), loss)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig('experiments\EDSR_CADRotated_wFlaws_REC_FDK_Al_noBH_X3binning_NoNoise_NoBlur_1066_Views_fullscan\loss_iters.png')
plt.close()

if PLOT_EPOCH:
    x_data = np.arange(0, (VAL_FREQ * len(loss)) / ITER_PER_EPOCH, VAL_FREQ / ITER_PER_EPOCH)
    plt.plot(x_data, loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('experiments\EDSR_CADRotated_wFlaws_REC_FDK_Al_noBH_X3binning_NoNoise_NoBlur_1066_Views_fullscan\loss_epochs.png')
    plt.close()