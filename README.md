# PINNS 

Evaluating Physics-Informed Neural Networks for Mesh-Free Structural Analysis 

# Reference paper
This repo includes the implementation of physics-informed neural networks in paper:

[Evaluating Physics-Informed Neural Networks for Mesh-Free Structural Analysis] (https://uu.diva-portal.org/smash/record.jsf?dswid=-3642&pid=diva2%3A1972440&c=2&searchType=SIMPLE&language=sv&query=alexander+sundl%C3%B6f&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%5D%5D&aqe=%5B%5D&noOfRows=50&sortOrder=author_sort_asc&sortOrder2=title_sort_asc&onlyFullText=false&sf=all)

## Setup Instructions

This project requires a Python environment with specific packages installed. The recommended way to set up the environment is using **Conda**.

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.
- (Optional) CUDA 12.8 installed if you want to use GPU acceleration with PyTorch.

---

### Step 1: Clone this repository

```bash
git clone https://github.com/alleswe2k/PINNs
cd PINNs
```

### Step 2: Create and activate the environment

```bash
conda create -n PINN
conda activate PINN
```

### Step 3: Install Dependencies

Some dependencies are downloaded from the **requirements.txt** file and some direct from PyTorch.

```bash
pip install -r requirements.txt
```

Install PyTorch with CUDA 12.8 support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

```

Install PyTorch Geometric:
```bash
pip install torch-geometric
```

### Step 4: Install Jupyter Interface (Choose One)

To run the notebooks, you'll need one of the following installed inside your environment:

#### Option A: Install JupyterLab

```bash
pip install jupyterlab
```
Then run:
```bash
jupyter lab
```

#### Option B: Install Jupyter Notebook

```bash
pip install notebook
```
Then run:
```bash
jupyter notebook
```

#### Option C: Use Visual Studio Code

If you use VS Code, install the Python and Jupyter extensions. Make sure you select the correct Conda environment (myenv) as the interpreter.

### Step 5: Run the Notebooks
Launch JupyterLab, Jupyter Notebook, or open a .ipynb file in VS Code. You should now be able to run all cells without import errors.



## Running Instruction

This project contains three folders dedicated for different parts: Ansys, beam_deflection, and plane_stress.

### Ansys:

This folder contains the mesh files used in Ansys togheter with the Gmsh scripts used to generate the files. It also contains a data folder containing CSV files with results from all Ansys simulations. A Python file is also included used to export the results from Ansys to a CSV file in a folder that needs to be specified by the user.  

### beam_deflection:

This folder contains a notebook for solving the one-dimensional cantilever beam presented in the paper aswell as a class based version that uses normal **.py** files. 

### plane_stress:

This folder contains four notebooks. **baseline.ipynb** has been used for the training optimization and solving the baseline metal plate problem presented in the paper. **plotting.ipynb** has been used to create the training loss plots. **transfer_learning.ipynb** was used to conduct the transfer learning experiments and lastly **two_holes.ipynb** was used to solved the plate that had two hole cutouts. 
