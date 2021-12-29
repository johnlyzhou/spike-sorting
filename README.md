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
To generate your own artificial datasets, you will need a library of waveform templates and a description of your probe geometry. We include the data used in our example analysis in `data/raw`. As of now, this package is designed for Neuropixels 2.0 probes, with tentative plans to expand to other geometries in the future.

Launch Jupyter notebook and open `example.ipynb` in the `notebooks` directory to see how to use this package!
```
jupyter notebook
```
