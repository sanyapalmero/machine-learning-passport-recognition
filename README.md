# Image recognition with python, tensorflow and keras
In this project I implement passports recognition. Most articles are used `tensorflow.keras.datasets` for preparing train and test data. As for me - I'm using my own dataset. You can upload same dataset with `download_images.py` script.

## Requirements
* python 3.8.x 64bit (on 09.12.2020 python 3.9 is not supported tesnorflow).
* tensorflow 2.3.1
* keras (already included in tensorflow)

Other dependencies you can find in requirements.txt file.

## Notes
If you have several python versions installed on Windows, you need to use `py -3.8` command for creating virtual environment.

## Commands
1 Create virtual env
``` bash
py -3.8 -m venv venv
```

2 Activate virtual env
``` bash
source venv/Scripts/activate
```

3 Install project dependencies
``` bash
pip install -r requirements.txt
```

If previous command can't install dependencies, try this
``` bash
py -3.8 -m pip install -r requirements.txt
```

## Todo
1 Install NVIDIA CUDA for running TensorFlow on GPU

Guide (rus): https://www.machinelearningmastery.ru/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781/

Guide (eng): https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781


## Links
Cuda toolkit documentation: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

Cuda toolkit downloads: https://developer.nvidia.com/cuda-downloads

Image recognition with TensorFlow and Keras: https://developer.ibm.com/technologies/artificial-intelligence/articles/image-recognition-challenge-with-tensorflow-and-keras-pt1/

CNN: https://www.tensorflow.org/tutorials/images/cnn