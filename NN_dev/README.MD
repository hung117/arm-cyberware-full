# RUN the GUI app
go to 'bare', run 'flutter run'( after setting up the prerequisite of course)
![plot](./res/homePage.png)
![plot](./res/record_page.png)
![plot](./res/about.png)
# RUN the NN
to to NN_dev, there are multiple files, in which `emg.classifier.ipynb` are multiple ways to approaches those that wield best result however placed in `classifier_to_b_used.ipynb` 
# CAD

![plot](./res/hand_cad/hand.png)
![plot](./res/hand_cad/wire_port.png)
![plot](./res/hand_cad/proximal.png)
![plot](./res/hand_cad/hand_below.png)

# nvidia

CUDA-driver
tensorflow use CUDA for GPU

Version Checks-

    Check which GPU you are using (win+X) -> Device Manager -> Display Adapters
    Determine which version of Tensorflow you want to use.
    Determine CUDA version supported by tensorflow version from here. 2.1.0 supports 10.1

Prerequisites-

    CUDA enabled GPU (check [here](https://www.tensorflow.org/install/source) to check)
    Visual Studio. This is required to install GCC compiler. You can also go with Windows SDK but I have not tried that.

Steps-

    Download CUDA toolkit for the required version here
    Download latest cuDNN for your cuda version here (You have to register with Nvidia for this).
    Install CUDA toolkit. It will automatically install the required Nvidia driver too.
    Install cuDNN using these instructions

```
sudo ubuntu-drivers devices
sudo apt install nvidia-driver-535
sudo apt install nvidia-utils-535
lspci
nvidia-smi
apt search nvidia-driver
uname -a       find current active kernel
```

```py

conda install -c conda-forge cudatoolkit=11.8.0
$condapip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

```

[REMOVE](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#removing-cuda-toolkit-and-driver)

```
sudo apt-get --purge remove "*cublas*" "cuda*" "nsight*"
```

# setup Conda, woking with pip:

```
    Run conda create -n venv_name and conda activate venv_name, where venv_name is the name of your virtual environment.

    Run conda install pip. This will install pip to your venv directory.

    Find your anaconda directory, and find the actual venv folder. It should be somewhere like /anaconda/envs/venv_name/.

    Install new packages by doing /anaconda/envs/venv_name/bin/pip install package_name.
```

In my case, set the bin/pip path to $condapip in bashrc, eg:

```
$condapip install transformers
```