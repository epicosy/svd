# Software Vulnerability Detection

## Installation

```
$ pip install -r requirements.txt

$ python setup.py install
```

## Development

This project includes a number of helpers in the `Makefile` to streamline common development tasks.

### Environment Setup

The following demonstrates setting up and working with a development environment:

```
### create a virtualenv for development

$ make virtualenv

$ source env/bin/activate


### run svd cli application

$ svd --help


### run pytest / coverage

$ make test
```


### Releasing to PyPi

Before releasing to PyPi, you must configure your login credentials:

**~/.pypirc**:

```
[pypi]
username = YOUR_USERNAME
password = YOUR_PASSWORD
```

Then use the included helper function via the `Makefile`:

```
$ make dist

$ make dist-upload
```

## Deployments

### Docker

Included is a basic `Dockerfile` for building and distributing `SVD`,
and can be built with the included `make` helper:

```
$ make docker

$ docker run -it svd --help
```


## Usage

In this example the working directory is the project directory:

```
$ cd project_root_path/svd
```

Curate dataset:

```
$ svd curate -d dataset/dataset.json -o dataset/prepare/dataset.json
```

Mine dataset:

```
$ svd mine -d dataset/prepare/dataset.json -o dataset/feature_models/ -nlp dataset/extracted_features/nlp_features/
```

Evaluate ML models:

```
$ svd evaluate -d dataset/prepare/dataset.csv -nlp dataset/extracted_features/nlp_features -rp dataset/results -mp dataset/models -t 8 -m KNN
```

Evaluate CNN model:

```
$ svd evaluate-cnn -d dataset/prepare/dataset.csv -mp dataset/models/cnn
```

Plot zipf-log: 

```
$ svd zipf -d dataset/dataset.json 
```

Plot histogram: 

```
$ svd histogram -d dataset/dataset.json 
```
