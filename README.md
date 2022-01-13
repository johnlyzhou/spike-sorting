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
To generate your own artificial datasets, you will need a library of waveform templates and a description of your probe geometry. We include the probe geometry used in our example analysis in `data/raw` and the templates (a larger file) [here](https://drive.google.com/file/d/1FY86UUkV-QdPpAMQGzNiU-wN3W9dtPLo/view?usp=sharing). As of now, this package is designed for [Neuropixels 2.0](https://www.science.org/doi/10.1126/science.abf4588) probes (Steinmetz et al. 2021), with tentative plans to expand to other geometries in the future.

Launch Jupyter notebook and open the examples in the `notebooks` directory to see how to use this package!
```
jupyter notebook
```
