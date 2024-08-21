# Optimizing Sparse Tensor Train Decomposition to Solve Linear Systems of Equations

### Code repository for my Computer Science MSc thesis at the Delft University of Technology.

#### Project folder structure
```data```: contains the sparse matrices from [1] that we use and the data collected from the experiments


```src/experiments```: contains all scripts needed for running our experiments

```src/analysis```: can be used to analyse the experimental data

```src/optimizers```: source code of proposed improvements

```test```: test suite for custom implementations

#### Reproducing results
In order to obtain my results, you can follow the below outlined procedure.

**Part 1: Setup**

- 1.1 After cloning the repository, be sure to set up a virtual environment (venv). 
We have used Python 3.10 as the base interpreter.
- 1.2 Next, activate the venv and run ```pip install -r requirements.txt``` to install the bulk of the required libraries.
- 1.3 Additionally, you will need *scikit-tt*, that you can obtain by running:
```pip install git+https://github.com/PGelss/scikit_tt```
- 1.4 In order to rerun and track the experiments, you will need a (free) Weights & Biases (wandb) account.
Be sure to get one.
- 1.5 \[Optional\]: If you would like to experiment with different matrices from [1], 
you can use the ```data/ssgetpy_download.py``` script to download the matrix files. 
These can be extracted using ```data/extract_data.py``` program.

**Part 2: Run experiments**

**Part 3: Plot and analyse results**


 


#### References
[1] Timothy A Davis and Yifan Hu. “The University of Florida sparse matrix collection”. In: ACM
Transactions on Mathematical Software (TOMS) 38.1 (2011), pp. 1–25.
