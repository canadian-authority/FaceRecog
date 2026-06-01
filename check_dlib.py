import dlib

# This should return True
print(f"CUDA Enabled: {dlib.DLIB_USE_CUDA}")

# This should return 1 (or however many GPUs you have)
print(f"Number of GPUs: {dlib.cuda.get_num_devices()}")