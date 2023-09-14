# Neurocognitive dynamics of preparatory and adaptive cognitive control

## Setting up programming environment

### Install Anaconda / Miniconda

See [here](https://docs.anaconda.com/free/anaconda/install/index.html).

### Install MNE

```shell
conda create --strict-channel-priority --channel=conda-forge --name=mne-1.5 mne=1.5.1
```

Activate conda environment:

```shell
conda activate mne-1.5
```

### Install dependencies

```shell
python -m pip install -r requirements.txt
```
