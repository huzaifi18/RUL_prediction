from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import os


def preprocess(dataset):
    scalers = {}
    for i in range(dataset.shape[1]):
        scalers[i] = MinMaxScaler(feature_range=(0, 1))
        dataset[:, i, :] = scalers[i].fit_transform(dataset[:, i, :])
    return dataset, scalers


def extract_VIT_capacity(x_datasets=None, y_datasets=None, seq_len=5, hop=1, sample=10, extract_all=True,
                         extract_c_only=False, extract_v_only=False, extract_i_only=False, extract_t_only=False,
                         extract_vit_only=False, extract_vc_only=False):
    x = []  # VITC = inputs voltage, current, temperature (in vector) + capacity (in scalar)
    y = []  # target capacity (in scalar)
    z = []  # cycle index
    SS = []  # scaler
    VITC = []  # temporary input

    for x_data, y_data in zip(x_datasets, y_datasets):
        # Load VIT from charging profile
        x_df = read_csv(x_data).dropna()
        x_df = x_df[['cycle', 'voltage_battery', 'current_battery', 'temp_battery']]
        x_df['cycle'] = x_df['cycle'] + 1
        x_len = len(x_df.cycle.unique())  # - seq_len

        # Load capacity from discharging profile
        y_df = read_csv(y_data).dropna()
        y_df['cycle_idx'] = y_df.index + 1
        y_df = y_df[['capacity', 'cycle_idx']]
        y_df = y_df.values  # Convert pandas dataframe to numpy array
        y_df = y_df.astype('float32')  # Convert values to float
        y_len = len(y_df)  # - seq_len

        data_len = np.int32(np.floor((y_len - seq_len - 1) / hop)) + 1
        for i in range(y_len):
            cy = x_df.cycle.unique()[i]
            df = x_df.loc[x_df['cycle'] == cy]
            # Voltage measured
            le = len(df['voltage_battery']) % sample
            vTemp = df['voltage_battery'].to_numpy()
            if le != 0:
                vTemp = vTemp[0:-le]
            vTemp = np.reshape(vTemp, (len(vTemp) // sample, -1))  # , order="F")
            vTemp = vTemp.mean(axis=0)
            # Current measured
            iTemp = df['current_battery'].to_numpy()
            if le != 0:
                iTemp = iTemp[0:-le]
            iTemp = np.reshape(iTemp, (len(iTemp) // sample, -1))  # , order="F")
            iTemp = iTemp.mean(axis=0)
            # Temperature measured
            tTemp = df['temp_battery'].to_numpy()
            if le != 0:
                tTemp = tTemp[0:-le]
            tTemp = np.reshape(tTemp, (len(tTemp) // sample, -1))  # , order="F")
            tTemp = tTemp.mean(axis=0)
            # Capacity measured
            cap = np.array([y_df[i, 0]])
            # Combined
            if extract_c_only:
                VITC_inp = cap
                VITC.append(VITC_inp)
            elif extract_v_only:
                VITC_inp = vTemp
                VITC.append(VITC_inp)
            elif extract_i_only:
                VITC_inp = iTemp
                VITC.append(VITC_inp)
            elif extract_t_only:
                VITC_inp = tTemp
                VITC.append(VITC_inp)
            elif extract_all:
                VITC_inp = np.concatenate((vTemp, iTemp, tTemp, cap))
                VITC.append(VITC_inp)
            elif extract_vit_only:
                VITC_inp = np.concatenate((vTemp, iTemp, tTemp))
                VITC.append(VITC_inp)
            elif extract_vc_only:
                VITC_inp = np.concatenate((vTemp, cap))
                VITC.append(VITC_inp)

        # Normalize using MinMaxScaler
        df_VITC = DataFrame(VITC).values
        scaled_x, scaler = preprocess(df_VITC[:, :, np.newaxis])
        scaled_x = scaled_x.astype('float32')[:, :, 0]  # Convert values to float

        # Create input data
        for i in range(data_len):
            x.append(scaled_x[(hop * i):(hop * i + seq_len), :])
            y.append(scaled_x[hop * i + seq_len, -1])
            # z.append(y_df[hop*i+seq_len, 1])
        SS.append(scaler)
        # import pdb; pdb.set_trace()
    return np.array(x), np.array(y)[:, np.newaxis], SS


def plot_loss(history, save_dir, model_dir):
    # Plot model loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(save_dir, model_dir, 'train_loss.png'))


def plot_pred(predict, true, save_dir, model_dir, name):  # Plot test prediction
    predict = predict.reshape(predict.shape[0])
    true = true.reshape(true.shape[0])
    plt.figure(figsize=(12, 4), dpi=150)
    plt.plot(predict, label='Prediction')
    plt.plot(true, label='True')
    plt.xlabel('Number of Cycle', fontsize=13)
    plt.ylabel('Discharge Capacity (Ah)', fontsize=13)
    plt.legend(loc='upper right', fontsize=12)
    plt.savefig(os.path.join(save_dir, model_dir, name + '.png'))
