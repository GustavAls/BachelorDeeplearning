import numpy as np
import matplotlib.pyplot as plt
import mat4py

directory = r'C:\Users\Bruger\Desktop\\'

train = mat4py.loadmat(directory + 'res101_train_progression.mat')
val = mat4py.loadmat(directory + 'res101_val_progression.mat')

wacc_train = np.mean(train['wacc'], axis=1)
wacc_val = np.mean(val['wacc'], axis=1)

loss_train = train['loss']
loss_val = val['loss']

epochs = np.hstack(([1], range(2, 101, 2)))
#Plotting
fig1 = plt.figure()
plt.grid(color='black', linestyle='-', linewidth=0.1)
plt.plot(epochs, loss_train, '--', label='Loss on training set')
plt.plot(epochs, loss_val, '--', label='Loss on validation set')
plt.title('Error rates for a ResNet101 with random resize cropping')
plt.ylabel('Error (%)')
plt.xlabel('Epochs')
plt.legend()
fig1.savefig(r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\pictures\loss_plot.png')

#Plotting weighted accuracy
fig2 = plt.figure()
plt.grid(color='black', linestyle='-', linewidth=0.1)
plt.plot(epochs, wacc_train, '--', label='Mean sensitivity on training set')
plt.plot(epochs, wacc_val, '--', label='Mean sensitivity on validation set')
plt.title('Mean sensitivity for a ResNet101 with random resize cropping')
plt.ylabel('Mean sensitivity')
plt.xlabel('Epochs')
plt.legend()
fig2.savefig(r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\pictures\wacc_plot.png')

