# SAFRON: Stitching Across the Frontier Network for Generating Colorectal Cancer Histology Images

![SAFRON-GAN (2)](https://github.com/Srijay/SAFRON/assets/6882352/6593d1b9-9048-4929-9912-e33e11373233)


This repository contains the official implementation of https://www.sciencedirect.com/science/article/abs/pii/S1361841521003820. Please follow the instructions given below to setup the environment and execute the code.


# Set Up Environment

Clone the repository, and execute the following commands to set up the environment.

```
cd SAFRON

# create base conda environment
conda env create -f environment.yml

# activate environment
conda activate safron

```

# Download data and extract tiles

The colon cancer whole slide images, along with their tissue component masks, can be downloaded from the datasets like [CRAG](https://github.com/XiaoyuZHK/CRAG-Dataset_Aug_ToCOCO) and [DigestPath](https://paperswithcode.com/dataset/digestpath). After downloading images, run the patch extracter code from folder named CRAG. 

```
python ImagesCropper.py
```

The script will create folders named 'images' and 'masks'. To create image-mask pairs, run the following command: 

```
python tools/process.py --input_dir ./masks --b_dir ./images --operation combine --output_dir ./paired

```


# Model Training

Now, we are set to train the model. Please update the training parameters inside the main script, put mode='train' and run the following command:

```
python main.py 
```

# User Interface

To try out user interface for the SAFRON model, please update the relevant paths and execute the command:

```
python interface.py 
```
