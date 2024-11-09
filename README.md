# Speech-driven Face Synthesis with 3D Gaussian Splatting
Repo for my master thesis.

## Abstract
TODO

## Installation

To install the repository, run the following commands:

```bash
conda env create --file environment.yml
conda activate master_thesis
pre-commit install
```

**Note:** When executing the code for the first time, gsplat will be compiled. This process will take approximately 3-5 minutes but only needs to be done once.

### FLAME Model
Download "FLAME 2023 (revised eye region, improved expressions, versions w/ and w/o jaw rotation)" from the [official website](https://flame.is.tue.mpg.de/download.php).
Place the downloaded file in the `assets/flame` directory.
You can delete the `flame2023_no_jaw.pkl` file that comes with the download.

**Note**: Manual registration is required on the FLAME website to download the model. Due to this, I cannot provide an automated installation script for this step.

## Usage
To train the model, use the following command:

```bash
./train.sh [optional arguments]
```

## Documentation
The documentation is generated using Sphinx. To generate the documentation, run the following command:

```bash
cd docs && make html && cd ..
```

The entry point for the documentation is `docs/build/html/index.html`.
