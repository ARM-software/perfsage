import os
import sys
import time
import subprocess
import multiprocessing

def do_something(seconds, p):
  time.sleep(seconds)
  print("ProcessID:", p, ", Done Sleeping!")

def run_vela(commands, p):
  start = time.perf_counter()

  for i, c in enumerate(commands):
    subprocess.run(c, stdout=subprocess.DEVNULL)
    if p==0 and (i+1)%100==0:
      finish = time.perf_counter()
      total_time = finish-start
      average_time = total_time / (i+1)
      print("ProcessID:", p, ",", (i+1), f"Examples Done! Total Time:{round(total_time, 2)}, Average Time:{round(average_time, 2)}")

  print("ProcessID:", p, ", Done!")


def main(argv):
  p_count = 20
  processes = []
  commands = []

  input_names = ["efficientnet_cifar10_quantization", "efficientnet_cifar10_quantization", "imagenet_may26", "cifar10_july8", "kws_chu_july19", "cifar10_july8", "super", "emnist_28", "emnist_64", "ssd_80", "ssd_300"]
  output_names = ["efficientnet_cifar10_quantization", "efficientnet_cifar10_quantization_1k", "imagenet_may26", "cifar10_july8", "kws_chu_july19", "cifar10_july8_mask", "super", "emnist_28", "emnist_64", "ssd_80", "ssd_300"]
  index = 9
  input_name = input_names[index]
  output_name = output_names[index]
  print("Input:", input_name, "\nOutput:", output_name)

  input_dir = "../../datasets/" + input_name

  tflite_names = next(os.walk(input_dir))[1]
  tflite_length = len(tflite_names)
  for i, tflite_name in enumerate(tflite_names[0:tflite_length]):
    tflite_path = os.path.join(input_dir, tflite_name, "tflites", tflite_name+"_INT8.tflite")
    vela_path = os.path.join(input_dir, tflite_name, "tflites", "output_v3")
    if os.path.exists(tflite_path):
      command = ["vela", tflite_path, "--output-dir", vela_path]
      commands.append(command)
    #print("PathID:", i, "Done!")

  commands_length = len(commands)
  p_length = round(commands_length / p_count)
  print("Commands Count:", commands_length, ", P Count:", p_length)

  start = time.perf_counter()

  for i in range(p_count):
    cs = i*p_length
    ce = min(cs+p_length, commands_length)
    print("Starting ProcessID:", i, ", start:", cs, ", end:", ce)
    p = multiprocessing.Process(target=run_vela, args=[commands[cs:ce], i])
    p.start()
    processes.append(p)

  for process in processes:
    process.join()

  finish = time.perf_counter()
  total_time = finish-start
  average_time = total_time / commands_length
  print(f"Finished in {round(finish-start, 2)}s, {round(average_time, 2)}s per tflite")

if __name__ == "__main__":
  main(sys.argv)