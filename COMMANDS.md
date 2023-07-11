To create the .mar file:
~~~
torch-model-archiver --model-name pionono \
    --version 1.0 \
    --serialized-file ./model-raw/model.pt \
    --extra-files ./Handler.py \
    --handler my_handler.py \
    --export-path model-store -f
~~~

~~~
torch-model-archiver --model-name pionono \
    --version 1.0 \
    --model-file ./utils/model_pionono.py \
    --serialized-file ./model-raw/model_state_dict.pt \
    --extra-files ./Handler.py \
    --handler my_handler.py \
    --export-path model-store -f
~~~

~~~
torch-model-archiver --model-name pionono \
    --version 1.0 \
    --model-file ./utils/model_pionono.py \
    --serialized-file ./model-raw/model_state_dict.pt \
    --extra-files $TEMP_DIR,./Handler.py \
    --handler my_handler.py \
    --export-path model-store -f
~~~


To start the Docker:
~~~
docker run --rm -it \
-p 3000:8080 -p 3001:8081 \
-v $(pwd)/model-store:/home/model-server/model-store pytorch/torchserve:latest \
torchserve --start --model-store model-store --models pionono=pionono.mar
~~~

To send an image:
~~~
curl -X POST http://127.0.0.1:3000/predictions/pionono -T example_images/images/slide001_core059.png
~~~


Alternative:
~~~
torchserve --start \
           --ncs \
           --model-store model-store \
           --models pionono=pionono.mar
~~~
