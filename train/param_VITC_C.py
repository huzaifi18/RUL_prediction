# Data
pth = 'data/NASA'
seq_len = 5  # time steps or history window size
hop = 1  # overlap between steps
k = 3  # kfold cross validation
random = 1
sample = 10

lr = 0.001  # default Adam optim
batch_size = 50
epochs = 100

test_data = '_B18'  # change based on which test data is used in the current experiment
save_dir = 'saved/fusion/'
model_dir = "VITC_C_newprep_F_" + str(seq_len) + test_data
# model_dir = "C_C_" + str(seq_len) + test_data
