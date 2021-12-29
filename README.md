# Spike Sorting

Spike waveform featurization using partitioned-subspace and vanilla variational autoencoders, evaluated on downstream clustering ARI and compared against a PCA benchmark.

### User Instructions
First, create a virtual environment (there are many ways to do so, I use [Anaconda](https://www.anaconda.com/products/individual) and will demonstrate with that):
```
conda create --name=spikesorting python=3.7.11
```
Activate the environment:
```
conda activate spikesorting
```
Navigate to the directory you want to contain the spike sorting code with `cd`.
Clone the repository:
```
git clone https://github.com/johnlyzhou/spike-sorting.git
```
Then, `cd` into the repository:
```
cd spike-sorting
```
Install the required packages:
```
pip install -r requirements.txt
```
Install local packages and set up directory structure:
```
pip install -e .
```
Launch `example.ipynb` in the `notebooks` directory with software of choice (Jupyter Notebook, Google Colab, etc.) to see how to use this package!
