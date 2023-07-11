import logging
import torch
import torch.nn.functional as F
import numpy as np
import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler


class Handler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(1024),
            transforms.ToTensor(),
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def preprocess_one_image(self, req):
        """
        Process one single image.
        """
        # get image from the request
        image = req.get("data")
        if image is None:
            image = req.get("body")
         # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        image = self.transform(image).to(self.device)
        # add batch dim
        image = image.unsqueeze(0)
        return image

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)
        return images

    def inference(self, x, **kwargs):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        :param **kwargs:
        """
        self.model.eval()
        self.model.forward(x)
        outs, _ = self.model.get_gold_predictions()
        probs = F.softmax(outs, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds

    def postprocess(self, preds):
        """
        Given the data from .inference, postprocess the output.
        In our case, we get the human readable label from the mapping
        file and return a json. Keep in mind that the reply must always
        be an array since we are returning a batch of responses.
        """
        res = []
        # preds has size [BATCH_SIZE, 1024, 1024]
        # convert it to list
        preds = preds.cpu()
        preds = np.asarray(preds, dtype=np.uint8)


        for pred in preds:
            pred_tissue = preds[pred != 4]
            sum_tissue = np.sum(pred_tissue)

            percentage_nc = np.sum(pred_tissue == 0) / sum_tissue
            percentage_gg3 = np.sum(pred_tissue == 1) / sum_tissue
            percentage_gg4 = np.sum(pred_tissue == 2) / sum_tissue
            percentage_gg5 = np.sum(pred_tissue == 3) / sum_tissue
            res.append({'% NC' : percentage_nc,
                        '% GG3': percentage_gg3,
                        '% GG4' : percentage_gg4,
                        '% GG5' : percentage_gg5})
        return res