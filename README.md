# Segmentation Model Serving
This repo provides code to serve a pytorch segmentation model for prostate cancer TMA.
An image can be sent to the model running as a service. It gets segmented into the Gleason grades and the percentage of each grade is predicted.

# Run Segmentation Service
To run the service, you can choose between two options, Docker or Conda, to set up the enironment.

## Option 1: Docker
* Navigate into this folder (segmodel_serving)
* Copy the file `model.pt` (from the google drive folder) into the folder 'model'
* Install Docker
* Open a console
* Build the Docker with `docker build --network=host -t seg_serving`.
* Run the Docker: `docker run --rm -it -p 6500:6500`

## Option 2: Conda
* Navigate into this folder (segmodel_serving)
* Copy the file `model.pt` (from the google drive folder) into the folder 'model'
* Install miniconda (or anaconda): https://docs.anaconda.com/anaconda/install/linux/
* Open a console
* Set up the conda environment: `conda env create --file environment.yaml`
* Activate the environment: `conda activate seg_serving`
* Run `python predict_app.py`


# Make a Test request
* Open a console
* Navigate into this folder (segmodel_serving)
* Set up the conda environment: `conda env create --file environment_client.yaml`
* Activate the environment: `conda activate seg_client`
* Run `python client.py`
* The prediction of the percentage of each Gleason grade will be shown in the console
