# PointDPI
Here is the code for paper **Exploring Drug-protein Interaction by Aligning the Multi-modal Molecular Structures**. 

# System Requirements
The source code developed in Python 3.8 using PyTorch 2.1.1 The required python dependencies are given below. PointDPI is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

numpy==1.24.4

scikit_learn==1.3.2

torch==2.1.1

tqdm==4.66.1

# Installation Guide
It normally takes about 10 minutes to install a new conda environment on a normal desktop computer. Run the following code under the conda environment to create the new virtual environment and install the required packages.
    
    $ conda create --name PointDPI python=3.8
    
    $ conda activate PointDPI

    $ pip install numpy==1.24.4
    
    $ pip install scikit_learn==1.3.2
    
    $ pip install torch==2.1.1
    
    $ pip install tqdm==4.66.1

# Datasets
We evaluated the performance of the method on three public datasets: DrugBank, BindingDB-IBM, and Luo's dataset.
For Drugbank dataset, we provided the split datasets for:

(1) ablation experiments in folder ./dataset/drugbank/result/

(2) five-fold cross-validation in folder ./dataset/drugbank/result/CV5/

(3) cold-start experiments in folder ./dataset/drugbank/result/cold_drug/ and ./dataset/drugbank/result/cold_protein/

For Luo's dataset, we provided the split datasets for five-fold cross-validation in folder ./dataset/dtinet/result/CV5/.

For BindingDB dataset, we provided the standard splited dataset in folder ./dataset/bingdingdb/result/.

Due to storage space restrictions on github, you can download our dataset by visiting the link: aaa.
Unzip the dataset folder and place it in the root directory of the project to achieve: ./dataset/
