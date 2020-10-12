# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GRU, PReLU, SimpleRNN, Dense, Activation, BatchNormalization, Conv2D, Conv1D, Flatten, LeakyReLU, Dropout, MaxPooling2D, MaxPooling1D, Reshape


# %%
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


# %%
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df, val_df, test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])


# %%
def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window


# %%
def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset


# %%
@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example


# %%
def plot(self, model=None, plot_col='Confirmed_Smooth', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(1, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      return(predictions[n, :, label_col_index])
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Days')

WindowGenerator.plot = plot


# %%
MAX_EPOCHS = 10

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='auto')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val)
  return history


# %%
from IPython.display import clear_output


# %%
# join county and state column 
# make a looop of this code below for each county 
# put the next 14 days in date formate
# join with final_final_final to make final_final_final_final
df = pd.read_csv('final_final_final.csv')
df = df[['date', 'covid symptoms', 'do i have covid',
       'covid test', 'covid near me', 'covid vaccines', 'covid travel',
       'travel restrictions', 'covid depression', 'covid health insurance',
       'test sites', 'bars near me', 'Confirmed', 'Confirmed_Smooth',
       'Deaths_Smooth', 'Admin2', 'Province_State', 'Lat', 'Long_']]
df = df.dropna(subset = ['Confirmed_Smooth'])
df['date'] = pd.to_datetime(df['date'])
date_time = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S')
timestamp_s = date_time.map(datetime.datetime.timestamp)
date_time = timestamp_s

base = datetime.datetime.today()
date_list = [base - datetime.timedelta(days=x) for x in range(14)]

df["Joined"] = df["Admin2"] + df["Province_State"]

for county in np.unique(np.array(df["Joined"])):
        new_df = df[df.Joined == county].drop(['Joined', "Admin2", "Province_State", "date"], axis=1)
        new_df2 = new_df

        n = len(new_df)
        train_df = new_df[0:int(n*0.7)]
        val_df = new_df[int(n*0.7):int(n*0.9)]
        test_df = new_df[int(n * 0.9):]

        num_features = new_df.shape[1]

        OUT_STEPS = 14
        multi_window = WindowGenerator(input_width=28,
                               label_width=OUT_STEPS,
                               train_df=train_df, val_df=val_df, test_df=test_df,
                               shift=OUT_STEPS, label_columns = ['Confirmed_Smooth'])

        multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(32),
        tf.keras.layers.Dense(OUT_STEPS*num_features),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
        ])

        clear_output()

        history = compile_and_fit(multi_lstm_model, multi_window)

        new_df2 = new_df2[new_df2.shape[0]-28:new_df2.shape[0]]
        
        lat = new_df2[new_df2.shape[0]-14:new_df2.shape[0]]['Lat']
        long = new_df2[new_df2.shape[0]-14:new_df2.shape[0]]['Long_']

        n = len(new_df2)
        train_df = new_df2[0:int(n*0.7)]
        val_df = new_df2[int(n*0.7):int(n*0.9)]
        test_df = new_df2

        num_features = new_df2.shape[1]

        OUT_STEPS = 14
        multi_window = WindowGenerator(input_width=28,
                               label_width=OUT_STEPS,
                               train_df=train_df, val_df=val_df, test_df=test_df,
                               shift=OUT_STEPS, label_columns = ['Confirmed_Smooth'])
        results = np.array(multi_window.plot(multi_lstm_model))
        
        
        print(county)
        
        results_df = pd.DataFrame(list(zip(date_list, results, lat, long)), columns = ['date', 'Confirmed_Smooth', 'Lat', 'Long_'])
        df = pd.concat([df, results_df])

print(df)


# %%



