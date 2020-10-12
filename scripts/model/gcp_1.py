# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import gc


# %%
from os import listdir
from os.path import isfile, join
mypath = "COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = np.sort(onlyfiles)


# %%
df = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/03-22-2020.csv")


# %%
date = '03-22-2020'
dates = []
for i in range(df.shape[0]):
    dates.append(date)
dates = pd.Series(dates, name = 'date')
df = pd.concat([dates, df], join = 'inner', axis = 1)


# %%
for file in onlyfiles:
    date = file[0:10]
    if date > "03-22-2020":
        new_df = pd.read_csv("COVID-19/csse_covid_19_data/csse_covid_19_daily_reports/"+file)
        dates = []
        for i in range(new_df.shape[0]):
            dates.append(date)
        dates = pd.Series(dates, name = 'date')
        new_df = pd.concat([dates, new_df], join = 'inner', axis = 1)
        df = pd.concat([df, new_df])


# %%
df = df.sort_values(by = ['Combined_Key', "date"])


# %%
new_df = df[0:0]
for county in np.unique(list(df.Combined_Key)):
    new_new_df = df[df.Combined_Key == county]
    new_confirmed = []
    new_death = []
    for i in range(new_new_df.shape[0]):
        if i == 0:
            new_confirmed.append(np.array(new_new_df["Confirmed"])[i])
            new_death.append(np.array(new_new_df["Deaths"])[i])
            continue
        new_confirmed.append(np.array(new_new_df["Confirmed"])[i] - np.array(new_new_df["Confirmed"])[i-1])
        new_death.append(np.array(new_new_df["Deaths"])[i] - np.array(new_new_df["Deaths"])[i-1])
    new_new_df.loc[:,"Confirmed"] = new_confirmed
    new_new_df.loc[:,"Deaths"] = new_death
    new_df = pd.concat([new_df, new_new_df])
    gc.collect()


# %%
df = new_df


# %%
df1 = df[0:round(df.shape[0]/4)]


# %%
df2 = df[round(df.shape[0]/4):round(2*df.shape[0]/4)]


# %%
df3 = df[round(2* df.shape[0]/4):round(3*df.shape[0]/4)]


# %%
df4 = df[round(3*df.shape[0]/4):df.shape[0]]


# %%
df1.to_pickle('df1.pkl')
df2.to_pickle('df2.pkl')
df3.to_pickle('df3.pkl')
df4.to_pickle('df4.pkl')


# %%



# %%



# %%



# %%
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
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
df = pd.read_csv('final_final_final.csv')


# %%
df = df[df.Admin2 == 'Los Angeles']


# %%
df = df.dropna(subset = ['Confirmed_Smooth'])


# %%
df = df[['date', 'covid symptoms', 'do i have covid',
       'covid test', 'covid near me', 'covid vaccines', 'covid travel',
       'travel restrictions', 'covid depression', 'covid health insurance',
       'test sites', 'bars near me', 'Confirmed', 'Confirmed_Smooth',
       'Deaths_Smooth']]


# %%
df['date'] = pd.to_datetime(df['date'])


# %%
date_time = pd.to_datetime(df.pop('date'), format='%d.%m.%Y %H:%M:%S')


# %%
timestamp_s = date_time.map(datetime.datetime.timestamp)


# %%
date_time = timestamp_s


# %%
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]


# %%
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


# %%
class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
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
OUT_STEPS = 14
multi_window = WindowGenerator(input_width=30,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)


# %%
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(16, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

#multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)


# %%



# %%
df = pd.read_csv('final_final_final.csv')


# %%
df = df[df.Admin2 == 'Los Angeles']


# %%
df = df.dropna(subset = ['Confirmed_Smooth'])


# %%
df = df[['date', 'covid symptoms', 'do i have covid',
       'covid test', 'covid near me', 'covid vaccines', 'covid travel',
       'travel restrictions', 'covid depression', 'covid health insurance',
       'test sites', 'bars near me', 'Confirmed', 'Confirmed_Smooth',
       'Deaths_Smooth']]


# %%
df = df[df.shape[0]-28:df.shape[0]]


# %%
df['date'] = pd.to_datetime(df['date'])


# %%
date_time = pd.to_datetime(df.pop('date'), format='%d.%m.%Y %H:%M:%S')


# %%
timestamp_s = date_time.map(datetime.datetime.timestamp)


# %%
date_time = timestamp_s


# %%
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df

num_features = df.shape[1]


# %%
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


# %%
OUT_STEPS = 14
multi_window = WindowGenerator(input_width=28,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)


# %%
multi_window.plot(multi_lstm_model)


# %%



