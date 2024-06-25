# Expanding the coverage of spatial proteomics 

### Overview
This is the repository for the paper "Expanding the coverage of spatial proteomics"
([Bioinformatics](https://academic.oup.com/bioinformatics/article/40/2/btae062/7600423)). In thiswork, we first sought to develop a flexible approach for finding a small subset of markers and using them to predict the full-image expression pattern of the remaining markers, this enlarges the coverage of spatial proteomics. All code and intermediate results are available in this repository.

### System and Hardware Requirements
Linux-based systems and NVIDIA graphics hardware with CUDA and cuDNN support are required. Our program is tested on Linux CentOS with CUDA 12.1.

### Requirements
- python (tested on version 3.9.1)
- scikit-learn (tested on version 1.0.2)
- scipy (tested on version 1.10.0)
- matplotlib (tested on version 3.7.1) 
- numpy (tested on version 1.25.1) 
- tifffile (tested on version 2023.7.18) 
- torch (tested on 1.9.1+cu111)
- networkx (tested on version 3.0) 
- seaborn, tqdm (optional)

To reproduce the environment, use the following command:
```bash
pip install -r requirements.txt
```

Our experiments are conducted on the conda environment. If use conda to install the dependencies, use the following command:
```bash
conda env create -f environment.yml
```


### Usage
To reproduce the results in the paper, open the Jupyter notebook files in the folder `single` or `mp` and run the cells. For example, to reproduce the single panel experiment results for spleen dataset, open the file `panel_selection_spleen.ipynb` under the folder `single` and run the cells.

The folder `single` is for single-panel experiments and the folder `mp` is for multi-panel experiments. 

The following is the description of juptyernotebook files under `single` folder:
- `panel_selection_spleen.ipynb` is the single panel experiment for spleen dataset;
- `panel_selection_lymph_node.ipynb` is the single panel experiment for lymph node dataset;
- `panel_selection_large_intestine.ipynb` is the single panelexperiment for large intestine dataset;
- `panel_selection_small_intestine.ipynb` is the single panel experiment for small intestine dataset;
- `panel_selection_pancreas.ipynb` is the single panel experiment for pancreas dataset;

The following is the description of juptyernotebook files under `mp` folder:
- `panel_selection_li` is the multi-panel experiment for large intestine dataset;
- `panel_selection_si` is the multi-panel experiment for small intestine dataset;

### Example
To reproduce the result of single panel experiment for spleen dataset:
- Download the data from [HuBMAP](https://portal.hubmapconsortium.org/). The image names can be found under the folder `csv`. `train_path_SP.csv` contains the training image names, `val_path_SP.csv` contains the validation image names, `test_path_SP.csv` contains the test image names. The `image_address_spleen.txt` stores the image addresses to download. Run the following command to download the data to the folder 'data/spleen':
```bash
cd data
wget -i image_address_spleen.txt -P spleen
```


- Find the jupyternotebook file `panel_selection_spleen.ipynb` under the folder `single` and run the cells. It will do the following steps:
  - Initialize the graph;
  - Select the minimal predictive subset of markers rounds by rounds. In each round: 
    - Add the marker that has the highest predictive power for the remaining markers;
    - Train the model with the selected markers;
  - Evaluate the model.

It may take days to run the whole experiment. In each round, the running time for training a model will take about 4 hours to on a single GPU. We provide the intermediate results in the folder `single/spleen_output`. The intermediate results can be used to reproduce the results figures and tables without needing to run all the training steps.



### Datasets
All the data are relevant CODEX image data from [The Human BioMolecular Atlas Program (HuBMAP)](https://portal.hubmapconsortium.org/). The image names can be found under the folder 'csv'. The IMC pancreas dataset can be downloaded from [The Human Tumor Atlas Network (HTAN)](https://humantumoratlas.org/explore?selectedFilters=%5B%7B%22value%22%3A%22IMC%22%2C%22group%22%3A%22assayName%22%2C%22count%22%3A79%2C%22isSelected%22%3Afalse%7D%5D&tab=file).

Here also provides an instruction to download data by 'download_data.md'.


### Acknowlegements
The fnet code is modified from [AllenCellModeling/pytorch_fnet](https://github.com/AllenCellModeling/pytorch_fnet). We thank their contribution for providing the open source software.

### TO_DO: License 
