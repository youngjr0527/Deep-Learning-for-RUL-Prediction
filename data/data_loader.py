import math
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------- DATA PRE-PROCESSING ---------------------------------------
def add_remaining_useful_life(df):
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    
    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    
    # Calculate remaining useful life for each row
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life
    
    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame

def add_operating_condition(df):
    df_op_cond = df.copy()
    
    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))
    
    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
                        df_op_cond['setting_2'].astype(str) + '_' + \
                        df_op_cond['setting_3'].astype(str)
    
    return df_op_cond

def condition_scaler(df_train, df_test, sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_train.loc[df_train['op_cond']==condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_test.loc[df_test['op_cond']==condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond']==condition, sensor_names])
    return df_train, df_test

def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    # first, take the exponential weighted mean
    
    def create_mask(data, samples):
        result = np.zeros_like(data)
        result[0:samples] = 1
        return result
    
    mask = create_mask(df[sensors[0]].values, n_samples)
    
    for sensor in sensors:
        df[sensor] = df[sensor].ewm(alpha=alpha).mean()
    
    df['mask'] = mask
    return df

def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]
    
    for start, stop in zip(range(0, num_elements-sequence_length), range(sequence_length, num_elements)):
        yield data[start:stop, :]
        
def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()
    
    data = []
    for unit_nr in unit_nrs:
        df_unit = df[df['unit_nr'] == unit_nr]
        g = gen_train_data(df_unit, sequence_length, columns)
        for seq in g:
            data.append(seq)
    return np.asarray(data)

def gen_labels(df, sequence_length, label):
    data = df[label].values
    num_elements = data.shape[0]
    
    return data[sequence_length:num_elements, ...]
    
def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()
    
    data = []
    for unit_nr in unit_nrs:
        df_unit = df[df['unit_nr'] == unit_nr]
        g = gen_labels(df_unit, sequence_length, label)
        data.extend(g)
    return np.asarray(data)

def gen_test_data(df, sequence_length, columns, mask_value):
    data = df[columns].values
    num_elements = data.shape[0]
    
    mask = df['mask'].values >= mask_value
    
    for start, stop in zip(range(0, num_elements-sequence_length), range(sequence_length, num_elements)):
        if mask[stop-1]:
            for idx in range(stop - sequence_length, stop):
                if mask[idx]:
                    yield data[start:stop, :]
                    break

def get_data(dataset, sensors, sequence_length, alpha, threshold):
	# files
    dir_path = './datasets/'

    train_file = dir_path + 'train_'+dataset+'.txt'
    test_file = dir_path + 'test_'+dataset+'.txt'
    # columns
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names
	
    # data readout
    train = pd.read_csv(train_file, sep=r'\s+', header=None, names=col_names)
    test = pd.read_csv(test_file, sep=r'\s+', header=None, names=col_names)
    test['RUL'] = pd.read_csv(dir_path + 'RUL_'+dataset+'.txt', sep=r'\s+', header=None).values

    # drop non-informative features
    # the feature importance analysis is performed in the noteboook
    # first construct lists of informative sensor
    drop_sensors = []
    for sensor in sensor_names:
        if sensor not in sensors:
            drop_sensors.append(sensor)
				
    # labeling data
    train = add_remaining_useful_life(train)
    train = add_operating_condition(train)
    test = add_operating_condition(test)
    
    # exponential smoothing
    train = exponential_smoothing(train, sensors, n_samples=50, alpha=alpha)
    
    # columns to drop
    drop_cols = setting_names + drop_sensors
    
    # filter out sensor data
    train_filtered = train.copy()
    train_filtered = train_filtered.drop(drop_cols, axis=1)
    
    test_filtered = test.copy()
    test_filtered = test_filtered.drop(drop_cols, axis=1)
    
    # scaling data
    train_filtered, test_filtered = condition_scaler(train_filtered, test_filtered, sensors)

    # gen train/valid data
    splits = GroupShuffleSplit(n_splits=1, train_size=0.90, random_state=42)

    for train_idx, val_idx in splits.split(train_filtered.iloc[:, :], groups=train_filtered['unit_nr']):
        train_data = train_filtered.iloc[train_idx, :]
        val_data = train_filtered.iloc[val_idx, :]
    
    # pick a feature columns
    feature_cols = ['s_{}'.format(i) for i in range(1, 22) if 's_{}'.format(i) in sensors]

    # training data
    x_train = gen_data_wrapper(train_data, sequence_length, feature_cols)
    y_train = gen_label_wrapper(train_data, sequence_length, ['RUL']).reshape(-1)

    # validation data
    x_valid = gen_data_wrapper(val_data, sequence_length, feature_cols)
    y_valid = gen_label_wrapper(val_data, sequence_length, ['RUL']).reshape(-1)
    
    # testing data
    x_test = []

    test_unit_nrs = test_filtered['unit_nr'].unique()
    for unit_nr in test_unit_nrs:
        df_unit = test_filtered[test_filtered['unit_nr'] == unit_nr]
        g = gen_test_data(df_unit, sequence_length, feature_cols, threshold)
        for seq in g:
            x_test.append(seq)
    x_test = np.asarray(x_test)
    
    if len(x_test) <= 0:
        raise Exception('No test data was generated, decrease the threshold parameter!')
    
    y_test = test_filtered['RUL'].values[-len(x_test):]
    
    return x_train, y_train, x_valid, y_valid, x_test, y_test 