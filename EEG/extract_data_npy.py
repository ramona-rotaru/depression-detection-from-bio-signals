import numpy as np
import mat73

mdd = mat73.loadmat("./data/dep_c1_new.mat")
control = mat73.loadmat("./data/controls_c1_new.mat")

data_control = control['controls_r']
eeg_control = data_control['G']



print(len(eeg_control))
list_of_shapes_control = [eeg_control[i][0][0].shape[1] for i in range(len(eeg_control))]
min_controls = np.min(list_of_shapes_control)


data_mdd = mdd['dep_r']
eeg_mdd = data_mdd['G']

print(len(eeg_mdd))
list_of_shapes_mdd = [eeg_mdd[i][0].shape[1] for i in range(len(eeg_mdd))]
min_mdd = np.min(list_of_shapes_mdd)


min_data = np.min([min_controls, min_mdd])
print(min_data)

list_controls = []
list_mdd = []
#cut all data and save 2 npy array files
#cut the data to the same length
for i in range(30): #take the firt 30 participants
    eeg_control[i][0][0] = eeg_control[i][0][0][:,0:min_data]
    print(eeg_control[i][0][0].shape)
    list_controls.append(eeg_control[i][0][0])
    
for i in range(len(eeg_mdd)):
    eeg_mdd[i][0] = eeg_mdd[i][0][:,0:min_data]
    print(eeg_mdd[i][0].shape)
    list_mdd.append(eeg_mdd[i][0])
    
    
array_controls = np.array(list_controls)
array_mdd = np.array(list_mdd)
  
    
#Save data as array matix with 3 d dimensions
np.save('eeg_control.npy', array_controls)
np.save('eeg_mdd.npy', array_mdd)




