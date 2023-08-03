import random as rn
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import json
import re
import utils
import param_V_CNN_C_LSTM as pr

SEED = 12345
os.environ["CUDA_VISIBLE_DEVICES"] = str(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
tf.compat.v1.random.set_random_seed(SEED)
tf.random.set_seed(SEED)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

def preprocess(dataset):
    from sklearn.preprocessing import MinMaxScaler
    scalers = MinMaxScaler(feature_range=(0, 1))
    scaled = scalers.fit_transform(dataset)
    return scaled, scalers


def extract_VIT_capacity(x_datasets, y_datasets, seq_len, hop, sample, extract_c_only=False):
    from pandas import read_csv, DataFrame
    V = []
    I = []
    T = []
    C = []

    x = []
    y = []

    SS = []

    for x_data, y_data in zip(x_datasets, y_datasets):
        # Load VIT from charging profile
        x_df = read_csv(x_data).dropna()
        x_df = x_df[['cycle', 'voltage_battery', 'current_battery', 'temp_battery']]
        x_df = x_df[x_df['cycle'] != 0]  # cycle ke-0 tidak masuk
        x_df = x_df.reset_index().drop(columns="index")
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
            # Capacity measured
            cap = np.array([y_df[i, 0]])
            C.append(cap)
            df_C = DataFrame(C).values
            scaled_C, scaler_C = preprocess(df_C)
            scaled_C = scaled_C.astype('float32')[:, :]

            le = len(df['voltage_battery']) % sample

            # Voltage measured
            vTemp = df['voltage_battery'].to_numpy()
            if le != 0:
                vTemp = vTemp[0:-le]
            vTemp = np.reshape(vTemp, (len(vTemp) // sample, -1)) #, order="F")
            vTemp = vTemp.mean(axis=0)
            V.append(vTemp)
            df_V = DataFrame(V).values
            scaled_V, scaler = preprocess(df_V)
            scaled_V = scaled_V.astype('float32')[:, :]

        if extract_c_only:
            for i in range(data_len):
                x.append(scaled_C[(hop * i):(hop * i + seq_len)])
                y.append(scaled_C[hop * i + seq_len])
        else:
            for i in range(data_len):
                x.append(scaled_V[(hop * i):(hop * i + seq_len)])
                y.append(scaled_C[hop * i + seq_len])
    return np.array(x), np.array(y), scaler_C


def main():
    pth = pr.pth
    train_x_files = [os.path.join(pth, 'charge/train', f) for f in os.listdir(os.path.join(pth, 'charge/train'))]
    train_x_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    train_y_files = [os.path.join(pth, 'discharge/train', f) for f in os.listdir(os.path.join(pth, 'discharge/train'))]
    train_y_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    test_x_data = [os.path.join(pth, 'charge/test', f) for f in os.listdir(os.path.join(pth, 'charge/test'))]
    test_y_data = [os.path.join(pth, 'discharge/test', f) for f in os.listdir(os.path.join(pth, 'discharge/test'))]
    print("train X:", train_x_files)
    print("train Y:", train_y_files)

    folds = list(KFold(n_splits=pr.k, shuffle=True, random_state=pr.random, ).split(train_x_files))

    for j, (train_idx, val_idx) in enumerate(folds):
        print('\nFold', j + 1)
        train_x_data = [train_x_files[train_idx[i]] for i in range(len(train_idx))]
        train_y_data = [train_y_files[train_idx[i]] for i in range(len(train_idx))]
        val_x_data = [train_x_files[val_idx[i]] for i in range(len(val_idx))]
        val_y_data = [train_y_files[val_idx[i]] for i in range(len(val_idx))]
        print("train X:", train_x_data)
        print("train y:", train_y_data)
        print("val X:", val_x_data)
        print("val y", val_y_data)
        print("test: x", test_x_data)
        print("test: y", test_y_data)

        # lstm data
        trainX_lstm, trainY_lstm, SS_tr_lstm = extract_VIT_capacity(train_x_data, train_y_data, pr.seq_len_lstm, pr.hop, pr.sample,
                                                                          extract_c_only=True)
        valX_lstm, valY_lstm, SS_val_lstm = extract_VIT_capacity(val_x_data, val_y_data, pr.seq_len_lstm, pr.hop, pr.sample,
                                                                       extract_c_only=True)
        testX_lstm, testY_lstm, SS_tt_lstm = extract_VIT_capacity(test_x_data, test_y_data, pr.seq_len_lstm, pr.hop, pr.sample,
                                                                        extract_c_only=True)
        print('Input shape: {}'.format(trainX_lstm.shape))

        # CNN data
        trainX_cnn, trainY_cnn, SS_tr_cnn = extract_VIT_capacity(train_x_data, train_y_data, pr.seq_len_cnn, pr.hop, pr.sample)
        valX_cnn, valY_cnn, SS_val_cnn = extract_VIT_capacity(val_x_data, val_y_data, pr.seq_len_cnn, pr.hop, pr.sample)
        testX_cnn, testY_cnn, SS_tt_cnn = extract_VIT_capacity(test_x_data, test_y_data, pr.seq_len_cnn, pr.hop, pr.sample)
        print('Input shape: {}'.format(trainX_cnn.shape))

        # define inputs
        input_CNN = Input(shape=(pr.seq_len_cnn, trainX_cnn.shape[-1]), name="CNN_Input")
        input_LSTM = Input(shape=(pr.seq_len_lstm, trainX_lstm.shape[-1]), name="LSTM_Input")

        LSTM_layer = LSTM(32, activation='tanh', return_sequences=True, name="LSTM_layer")(input_LSTM)

        CNN_layer = Conv1D(32, 5, activation='relu', strides=1, padding="same", name="CNN_layer")(input_CNN)

        concat = concatenate([LSTM_layer, CNN_layer])

        flat = Flatten()(concat)

        output = Dense(32, activation='relu', name="predictor")(flat)
        output = Dense(1, name="Output")(output)

        model = Model(inputs=[input_LSTM, input_CNN], outputs=[output])

        optim = Adam(learning_rate=0.001)
        model.compile(loss='mse', optimizer=optim)
        model.summary()

        history = model.fit(x=[trainX_lstm, trainX_cnn],
                            y=[trainY_lstm, trainY_cnn],
                            validation_data=([valX_lstm, valX_cnn],
                                             [valY_lstm, valY_cnn]),
                            batch_size=50,
                            epochs=100)

        save_dir = pr.save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model_dir = pr.model_dir + '_k' + str(j + 1)
        if not os.path.exists(os.path.join(save_dir, model_dir)):
            os.makedirs(os.path.join(save_dir, model_dir))

        model.save(save_dir + model_dir + "/saved_model_and_weight")
        print("bobot dan model tersimpan")

        val_loss = []
        val_results = model.evaluate([valX_lstm, valX_cnn],
                                     [valY_lstm, valY_cnn])
        val_loss.append(val_results)
        print('Val loss:', val_results)

        test_loss = []
        results = model.evaluate([testX_lstm, testX_cnn],
                                 [testY_lstm, testY_cnn])
        test_loss.append(results)
        print('Test loss:', results)

        valPredict = model.predict([valX_lstm, valX_cnn])
        testPredict = model.predict([testX_lstm, testX_cnn])

        inv_valY = SS_val_cnn.inverse_transform(valY_cnn)
        inv_valPredict = SS_val_cnn.inverse_transform(valPredict)

        inv_testY = SS_tt_cnn.inverse_transform(testY_cnn)
        inv_testPredict = SS_tt_cnn.inverse_transform(testPredict)

        test_mae = mean_absolute_error(inv_testY, inv_testPredict)
        test_mse = mean_squared_error(inv_testY, inv_testPredict)
        test_mape = mean_absolute_percentage_error(inv_testY, inv_testPredict)
        test_rmse = np.sqrt(mean_squared_error(inv_testY, inv_testPredict))
        print('\nTest Mean Absolute Error: %f MAE' % test_mae)
        print('Test Mean Square Error: %f MSE' % test_mse)
        print('Test Mean Absolute Percentage Error: %f MAPE' % test_mape)
        print('Test Root Mean Squared Error: %f RMSE' % test_rmse)

        with open(os.path.join(save_dir, model_dir, 'eval_metrics.txt'), 'w') as f:
            f.write('Train data: ')
            f.write(json.dumps(train_x_data))
            f.write('\nVal data: ')
            f.write(json.dumps(val_x_data))
            f.write('\nTest data: ')
            f.write(json.dumps(test_x_data))
            f.write('\n\nTest Mean Absolute Error: ')
            f.write(json.dumps(str(test_mae)))
            f.write('\nTest Mean Square Error: ')
            f.write(json.dumps(str(test_mse)))
            f.write('\nTest Mean Absolute Percentage Error: ')
            f.write(json.dumps(str(test_mape)))
            f.write('\nTest Root Mean Squared Error: ')
            f.write(json.dumps(str(test_rmse)))

        # Save test prediction to text file
        testPred_file = open(os.path.join(save_dir, model_dir, 'test_predict.txt'), 'w')
        for row in inv_testPredict:
            np.savetxt(testPred_file, row)
        testPred_file.close()

        testY_file = open(os.path.join(save_dir, model_dir, 'test_true.txt'), 'w')
        for row in inv_testY:
            np.savetxt(testY_file, row)
        testY_file.close()

        # plot graph
        utils.plot_loss(history, save_dir, model_dir)
        utils.plot_pred(inv_valPredict, inv_valY, save_dir, model_dir, "val_pred")
        utils.plot_pred(inv_testPredict, inv_testY, save_dir, model_dir, "test_pred")


if __name__ == "__main__":
    main()