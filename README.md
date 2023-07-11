# Segmentation Model Serving
This repo provides code to serve a pytorch segmentation model for prostate cancer TMA.
An image can be sent to the model running as a service. It gets segmented into the Gleason grades and the percentage of each grade is predicted.

## Installation
* Install miniconda (or anaconda): https://docs.anaconda.com/anaconda/install/linux/ 
* Set up a conda environment:
    * `conda env create --file environment.yaml`
    * `conda activate seg_serving`

## Start RestAPI Service
* Copy the file `model.pt` into the folder 'model'
* Open a console
* Navigate into this folder (segmodel_serving)
* Run `python predict_app.py`

## Make a Test request
* Open a console
* Navigate into this folder (segmodel_serving)
* Run `python client.py`
* The prediction of the percentage of each Gleason grade will be shown in the console
