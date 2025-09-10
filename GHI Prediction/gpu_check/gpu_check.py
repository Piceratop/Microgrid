import tensorflow as tf

def gpu_check():
   # List all available physical devices
   print("Available devices: ", tf.config.list_physical_devices())

   # Check if GPU is available
   print("GPU available: ", tf.config.list_physical_devices('GPU'))

   # Another way to check
   print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

   # Check which device is being used
   print("TensorFlow is using GPU: ", tf.test.is_built_with_cuda())
   print("TensorFlow is built with CUDA: ", tf.test.is_built_with_cuda())

   # To see if a GPU is being actively used
   print("GPU name: ", tf.test.gpu_device_name())
   
gpu_check()