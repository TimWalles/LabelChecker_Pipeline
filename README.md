![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# LabelChecker data processing pipeline
## Installation and Setup

For the scripts to run seamlessly, download Python 3.12 (if you don't have Python already installed). Go to https://www.python.org/downloads/release/python-3123/ and download the installer you need.

> Tensorflow is [CPU only](https://www.tensorflow.org/install/pip#windows-wsl2:~:text=Note%3A%20TensorFlow%20with%20GPU%20access%20is%20supported%20for%20WSL2%20on%20Windows%2010%2019044%20or%20higher.%20This%20corresponds%20to%20Windows%2010%20version%2021H2%2C%20the%20November%202021%20update.%20You%20can%20get%20the%20latest%20update%20from%20here%3A%20Download%20Windows%2010.%20For%20instructions%2C%20see%20Install%20WSL2%20and%20NVIDIA%E2%80%99s%20setup%20docs%20for%20CUDA%20in%20WSL.) supported on windows. [Windows Users](#windows-users) that want to utilize their GPU need to install WSL. For CPU-only users WSL is optional, but we recommend using WSL on Windows systems.

To run the data processing pipeline scripts, follow these steps to set up your environment:

1. Clone the repository:
    ```bash
      git clone https://github.com/TimWalles/LabelChecker
      ```
    Or download and upzip the folder.

2. Navigate to the project directory:
    ```bash
    cd path/to/labelchecker_data_pipeline
    ```

3. Choose your preferred installation method:

   ### Option A: Using pip (Traditional Method)
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows use: .\env\Scripts\activate
   pip install -r requirements.txt
   ```

   ### Option B: Using Poetry (Alternative Method)
   ```bash
   # Install Poetry if you haven't already
   curl -sSL https://install.python-poetry.org | python3 -

   # Install Poetry shell plugin that runs a subshell with virtual environment activated
   poetry self add poetry-plugin-shell

   # Install dependencies using Poetry
   poetry install --no-root

   # Activate the Poetry virtual environment
   poetry shell
   ```


### Windows Users (optional)

It is recommended to use Windows Subsystem for Linux (WSL) for a more consistent development environment. Follow these steps to set up WSL or the steps in the [official documentation](https://learn.microsoft.com/en-gb/windows/wsl/install) of Microsoft:

1. Install WSL if you haven't already:
    ```bash
    wsl --install
    ```

2. Set up WSL environment:
    ```bash
    wsl
    ```

3. Navigate to the project directory within WSL:
    ```bash
    cd /mnt/c/path/to/labelchecker_data_pipeline
    ```

4. Create a virtual environment:
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```
    Replace `env` with the name of your choice.

5. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

Now you have successfully installed and set up your environment to run the data processing pipeline scripts.


> For enabling GPU access on WSL please follow these [instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

## The Data Processing Pipeline

The data processing pipeline, implemented in Python, consists of two scripts: `preprocessing.py` and `classification.py`. These scripts are used for data normalization and saving into a *LabelChecker* readable .csv format. The `preprocessing.py` script preprocesses Flowcam data, while the `classification.py` script performs the classification.

### Run the scripts
- Activate your virtual enviroment:
 ```bash
  workon env
  ```
  Navigate to the project directory:
  ```bash
  cd path/to/labelchecker_data_pipeline
  ``` 
- To preprocess Flowcam data:
  ```bash
  python preprocessing.py \
    -D path/to/flowcam_data \
    -V \ # to print detailed steps instead of only the progress bar
    -R # for reprocessing already processed data
  ```
- To classify the preprocessed data:
  ```bash
  python classification.py
    -D path/to/flowcam/data \
    -V  # to print detailed steps instead of only the progress bar
  ```
  Once the script starts running it will list the available classification models. 
  You can select the model of your choice by typing the respective number.
  
For more information on the input flags, you can use the following commands:
To get help for the `preprocessing.py` script, run:
```bash
python preprocessing.py --help
```
To get help for the `classification.py` script, run:
```bash
python classification.py --help
```

### Modularity

In order to address different situations and requirements, the data processing pipeline is modular. Both `preprocessing.py` and `classification.py` utilize services, which are independent modules that perform specific tasks. For example, size-threshold filters are used to remove objects that are too small for classification and artifacts that are too large. Users have the flexibility to modify or remove these services to customize the data pipeline according to their needs.

## Service Design

Each service is organized in a folder within the `services` directory. The structure of a service folder is as follows:

```bash
|-- src
    |-- services
        |-- ProcessData.py
        |-- config.py
        |-- README.md
        |-- <service category folder>
            |-- <service folder>
                |-- service.py
                |-- ...
    |-- ...
|-- preprocessing.py
|-- classification.py
```

### Service Settings

Services can be configured by editing the `config.py` file within main service folder. In the `config.py` file, users can enable or disable the service and adjust its settings as needed.

Read the [readme](./src/services/README.md) for more information about available services and how to add your own service module.

# The LabelChecker Program
The LabelChecker program is a tool that enables users to review and validate assigned labels returned by the `preprocessing.py` script. This ensures that the data meets the desired quality standards.

In addition, LabelChecker provides support for the validation and correction of automatically assigned classes after running the `classification.py` script. This feature allows users to fine-tune and customize the labels based on their domain-specific knowledge. For example, if classification models are used, they predict labels for each data entry. These predicted labels are populated in the label column. With LabelChecker, users have the flexibility to confirm or adjust these predicted labels. The adjusted labels are then populated in the label_true column, providing a reliable ground truth for subsequent analyses. Any necessary corrections can be easily made within LabelChecker, making it a comprehensive solution for label validation and customization.

## Download

You can download your version of LabelChecker from the [LabelChecker release folder](https://github.com/TimWalles/LabelChecker/releases).

## Support, Comments, and Requests

Join our [LabelChecker Discord server](https://discord.gg/tGBg7z2hSU) to connect with other users and developers. Here, you can also make feature requests or report errors.

## Commitment
The Data processing pipeline and LabelChecker program are available free of charge and compatible with all major operating systems. All data processing occurs locally, ensuring that there is no transfer of ownership of the complete dataset or any of its components from the user.

# Licence
