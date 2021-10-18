import os
import sys
import csv
import time
import subprocess
import multiprocessing
import numpy as np
import pandas as pd

def load_prediction_dict(file_path):
  with open(file_path, newline='') as f:
    reader = csv.reader(f)
    f_data = list(reader)
  keys = f_data[0]
  values = f_data[1]
  f_dict = {keys[index]:values[index] for index in range(len(keys))}

  return f_dict

def main(argv):
  p_count = 20
  processes = []
  commands = []

  input_names = ["efficientnet_cifar10_quantization", "efficientnet_cifar10_quantization", "imagenet_may26", "cifar10_july8", "kws_chu_july19", "cifar10_july8", "super", "emnist_28", "emnist_64"]
  output_names = ["efficientnet_cifar10_quantization", "efficientnet_cifar10_quantization_1k", "imagenet_may26", "cifar10_july8", "kws_chu_july19", "cifar10_july8_mask", "super", "emnist_28", "emnist_64"]
  index = 3
  input_name = input_names[index]
  output_name = output_names[index]
  print("Input:", input_name, " Output:", output_name)

  prediction_names = ["inference_time"]
  prediction_dict_v3 = {}
  prediction_dict_og = {}
  prediction_dict_diff = {}
  for n in prediction_names:
    prediction_dict_v3[n] = []
    prediction_dict_og[n] = []

  input_dir = "../../datasets/" + input_name

  tflite_names = next(os.walk(input_dir))[1]
  tflite_length = len(tflite_names)
  for i, tflite_name in enumerate(tflite_names[0:tflite_length]):
    vela_csv_path_v3 = os.path.join(input_dir, tflite_name, "tflites", "output_v3", tflite_name + "_INT8_summary_internal-default.csv")
    vela_csv_path_og = os.path.join(input_dir, tflite_name, "tflites", "output", tflite_name + "_INT8_summary_internal-default.csv")
    
    if os.path.exists(vela_csv_path_v3) and os.path.exists(vela_csv_path_og):
      f_dict_v3 = load_prediction_dict(vela_csv_path_v3)
      f_dict_og = load_prediction_dict(vela_csv_path_og)

      for n in prediction_names:
        prediction_dict_v3[n].append(float(f_dict_v3[n]))
        prediction_dict_og[n].append(float(f_dict_og[n]))

  for n in prediction_names:
    prediction_dict_v3[n] = np.array( prediction_dict_v3[n])
    prediction_dict_og[n] = np.array( prediction_dict_og[n])
    prediction_dict_diff[n] = (prediction_dict_og[n] - prediction_dict_v3[n]) / prediction_dict_og[n]
    
    print(n)
    df_describe = pd.DataFrame(prediction_dict_diff[n])
    print(df_describe.describe())

if __name__ == "__main__":
  main(sys.argv)