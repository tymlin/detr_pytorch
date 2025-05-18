# DETR in PyTorch

This repository contains an implementation of the DEtection TRansformer (DETR) in PyTorch and a training framework for
object detection models.

Code in this repository is based on the original repository [detr](https://github.com/facebookresearch/detr).

## 📋 Table of Contents

- [🔧 Installation](#-installation)
    - [📦 Prerequisites](#-prerequisites)
    - [💻 Environment](#-environment)
- [🚀 Usage](#-usage)
    - [🧪 Training](#-training)
    - [🧪 Evaluation](#-evaluation)
- [⚙️ Configuration](#-configuration)
- [📈 Data visualization](#-data-visualization)
- [🛠️ Development](#-development)
    - [Configuration of the `pre-commit`](#configuration-of-the-pre-commit)
    - [Adding new dependencies](#adding-new-dependencies)
    - [Adding own component](#adding-own-component)
- [👏 Acknowledgements, references](#-acknowledgements-references)

## 🔧 Installation

### 📦 Prerequisites

##### 1. Installing `conda` 🐍

Install `conda` from the official
website: [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)

<details>
<summary>Commands for Linux</summary>

- install `conda`
  ```bash
  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm ~/miniconda3/miniconda.sh
  ```

- after installing, close and reopen your terminal application or refresh it by running the following command:
  ```bash
  source ~/miniconda3/bin/activate
  ```

- initialize `conda` on all available shells, run the following command:
  ```bash
  conda init --all
  ```

</details>

##### 2. Installing `poetry` 🌱

This project is using `poetry` for dependency management, so first you should install `poetry` from official
website: [Poetry](https://python-poetry.org/docs/#installation).

If you are not eager to use `poetry`, you can also install the dependencies using `pip` (see the optional installation
and the end of the section).

<details>
<summary>Commands for Linux</summary>

```bash
sudo apt update
sudo apt install pipx -y
pipx ensurepath
pipx install poetry
```

</details>

### 💻 Environment

##### 1. Create a new `conda` environment

```bash
conda create --name <CONDA_ENV_NAME> python=3.10
```

Example:

```bash
conda create --name detr python=3.10
```

##### 2. Activate the environment

```bash
conda activate <CONDA_ENV_NAME>
```

Example:

```bash
conda activate detr
```

##### 3. Install the dependencies

```bash
poetry install
```

<details>
<summary>Optional installation (pip, PYTHONPATH)</summary>

- `pip` installation:
  ```bash
  pip install -e <PATH_TO_DETR>
  ```
  Example (if you are in the root directory of the project):
  ```bash
  pip install -e .
  ```

- `PYTHONPATH` environment variable:
  ```bash
  export PYTHONPATH="$PYTHONPATH:/path/to/detr"
  ```

</details>

## 🚀 Usage

### 🧪 Training

To train a model using a specific configuration run the following command:

```bash
python scripts/train.py <CONFIG_FILENAME>.yaml
```

> **_NOTE:_** If you do not provide a configuration file, the script will use the default configuration file
`default.yaml`.

Example:

```bash
python scripts/train.py default.yaml
```

After running the script, the training process will start and the results will be saved in the `runs` directory.
Each run creates a folder with the experiment name (see `setup.experiment_name` from the configuration file), and a
folder with the run name (see `setup.run_name` from the configuration file). The current date and time in the format
`YYYY_MM_DD_HH-MM-SS` is added to the run name.

The structure of the `runs` directory is as follows:

```
  ├── runs
  │   ├── experiment_name_1
  │   │   │   YYYY_MM_DD_HH-MM-SS_run_name_1
  │   │   │   YYYY_MM_DD_HH-MM-SS_run_name_2
  │   │   └── ...
  │   ├── experiment_name_2
  │   │   │   YYYY_MM_DD_HH-MM-SS_run_name_3
  │   │   │   YYYY_MM_DD_HH-MM-SS_run_name_4
  │   │   └── ...
  │   └── ...
```

The structure of the `YYYY_MM_DD_HH-MM-SS_run_name` is as follows:

```
  ├── YYYY_MM_DD_HH-MM-SS_run_name
  │   ├── checkpoints         # directory with model checkpoints
  │   ├── epochs_metrics      # directory with metrics, raw files and plots
  │   ├── eval_examples       # directory with evaluation examples based on current model predictions
  │   ├── logs                # directory with logs from the training process (those printed to the terminal)
  │   ├── model               # directory with the model architecture information, summary, etc.
  │   └── config.yaml         # configuration file used for the training
```

If you want to train a model on a [CoCo](https://cocodataset.org/#home) dataset, first download and prepare it
using the following command:

```bash
bash scripts/prepare_coco.sh
```

This script automatically downloads the `CoCo` dataset and prepares it for training in the `data_temp` directory.

### 🧪 Evaluation

To evaluate a model using a specific configuration, run the following command:

```bash
python scripts/eval.py <CONFIG_FILENAME>.yaml
```

> **_NOTE:_** If you do not provide a configuration file, the script will use the default configuration file
`evaluation.yaml`.

Example:

```bash
python scripts/eval.py evaluation.yaml
```

After evaluation is completed, the metrics are saved in the `runs/<experiment_name>/<run_name>/epochs_metrics` directory.
Additionally, the evaluation examples are saved in the `runs/<experiment_name>/<run_name>/eval_examples`

## ⚙️ Configuration

The project uses YAML configuration files for different training and inference scenarios.
Configuration files should be in the:

- `config_training` directory for training configurations
- `config_inference` directory for inference configurations

The Repository provides subsequent configuration files for training:

- `default.yaml` - training a model on the `CoCo` dataset

and for inference:

- `evaluation.yaml` - evaluation of the model on the `CoCo` dataset

## 📈 Data visualization

The repository offers visualization for the training and inference results. In addition to visualizations provided by
logger functionalities provided by logger functionalities (e.g. `WandbLogger`, `MLFlowLogger`), additional plots
generated during the training process can be found in the `runs` directory. There are also scripts available for
visualizing the inference latency evaluation results.

### Example plots generated during the training process

#### Epoch metrics visualization (`YYYY_MM_DD_HH-MM-SS_run_name/epochs_metrics`)

Here is just an example plot for 10 epochs and an already trained model.

<img src="./docs/metrics.png" alt="metrics" width="900">

#### Evaluation examples (`YYYY_MM_DD_HH-MM-SS_run_name/eval_examples`)

<img src="./docs/examples.jpg" alt="examples" width="900">


#### Logger visualization (based on `WandbLogger`)

Interactive comparison of models' predictions.

<img src="./docs/wandb.gif" alt="wandb" width="900">

## 🛠️ Development

### Configuration of the `pre-commit`

This project uses `pre-commit` for code formatting and linting. If you want to use it in your development process,
follow these steps:

1. Init hooks:

```bash
pre-commit install --config .pre-commit-config.yaml
```

or

```bash
pre-commit install
```

After initialization, `pre-commit` will run on every commit.

2. If you want to run `pre-commit` on all files, execute the command:

```bash
pre-commit run --all-files --config .pre-commit-config.yaml
```

or

```bash
pre-commit run --all
```

3. (NOT RECOMMENDED) If you want to skip `pre-commit` on commit, add the `--no-verify` flag:

```bash
git commit -m "<message>" --no-verify
```

### Adding new dependencies

If you want to add a new dependency, you can do it by running:

```bash
poetry add <PACKAGE_NAME>
```

Example:

```bash
poetry add numpy
```

> **_NOTE:_** Remember to commit the `poetry.lock` file after adding a new dependency.

> **_NOTE:_** Remember if you want to use `poetry` do not use `pip` and `conda` for installing dependencies, because it
> can break the `poetry` environment.

### Adding own component

This repository uses decorators to register new components. To add your component, simply apply the corresponding
`@register_<COMPONENT_TYPE>` decorator above your class definition.
