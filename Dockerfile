ARG BASE_CONTAINER=pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime
FROM $BASE_CONTAINER
ENV SHELL=/bin/bash

WORKDIR /app/

COPY requirements.txt .
COPY predict_app.py .
COPY model/ ./model/
COPY utils/ ./utils/
COPY Probabilistic_Unet_Pytorch/ ./Probabilistic_Unet_Pytorch/

USER root
#RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0
RUN python3 --version
RUN pip install -r requirements.txt
RUN pip list
RUN ls

#CMD echo ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"] &
#CMD echo ["mlflow", "ui"]
CMD ["/bin/bash", "-c", "python predict_app.py"]

EXPOSE 6500