# Services
Services are divided into two categories: preprocessing and classification. As their names imply, services in each category are used for either preprocessing (e.g., cleaning data through artifact detection) or classification (assigning taxonomical labels to objects). To help distinguish between these categories, preprocessing services are placed in the [preprocessing](./preprocessing/) folder, and classification services in the [classification](./classification/) folder. Within each category folder, you will find folders for each service.

## General Service Structure
Inside every service folder, there are always a main file: `service.py`.

```bash
|--services
    |-- ProcessData.py
    |-- config.py
    |--  <service category folder>
        |-- <service folder>
            |-- service.py
            |-- ...
    |-- ...
```
Besides the service folders, a [`ProcessData.py`](ProcessData.py) script exists that imports all services and contains two functions:

```python
def preprocess_data(...)
    # Executes all preprocess data steps
    ...

def classify_data(...)
    # Executes all data classification steps
    ...
```

and a [`config.py`](config.py) script that contains all preprocessing and classification service settings. 

## Adding New Services
When adding a new service, it needs to be imported into the [`ProcessData.py`](ProcessData.py) script:

```python
from .<ServiceName>.service import <ServiceName>
```
For example:

```python
from .preprocessing.AirBubbleDetection.service import AirBubbleDetector
from .preprocessing.BlurDetection.service import BlurDetection
from .preprocessing.DuplicateDetection.service import DuplicateDetection
from .preprocessing.SizeThreshold.service import SizeThreshold
from .preprocessing.DetritusDetection.service import DetritusDetection
from .classification.ObjectClassification.service import ObjectClassification
```

and settings need to be added to the [`config.py`](config.py). 

```python
class PreprocessingConfig(BaseModel):
    # <ServiceName>
    <ServiceName>_active: bool = ...

class ClassificationConfig(BaseModel):
    # <ServiceName>
    <ServiceName>_active: bool = ...
```

> Ensure consistent naming of the script names and within each service script as the ServiceName is used for saving the service settings. 


## Service Structure
Each service has a similar structure, containing a main `service.py` file. Besides these two files, services can contain a `model` directory and a `utils` directory.

### The `service.py`
This file contains a Python class named identically to the service folder with a class function named process_data. This function processes a list of LabelCheckerData objects based on the configuration specified in config.py. The class inherits the ModelLoader class when a model is used. Examples of services using models include [AirBubbleDetection](./preprocessing/AirBubbleDetection/) (TensorFlow model), [DetritusDetection](./preprocessing/DetritusDetection/) (sklearn model), and [ObjectClassification](./classification/ObjectClassification/) (TensorFlow model).

### The `model` Directory
The `model` directory is assumed to be structured as follows, and models can be loaded using the `ModelLoader` class:

```bash
|--model
    |--<model version>
        |--config.json
        |--model.<framework specific extension>
    |-- ...
```
Multiple versions, each with its own configuration, can exist in the `model` directory. If multiple model versions are present, the `ModelLoader` will prompt the user to select which version to use.