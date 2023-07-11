import torch
import logging
import cv2

import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from flask import Flask
from flask import request
from PIL import Image



app = Flask(__name__)
app.debug = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loaded_model = torch.load('./model/model.pt')
model = loaded_model
model.eval()


@app.route('/pc_segmentation', methods = ['POST'])
def segment_prostate_cancer():
    if request.method=='POST':
        # Get requests' parameters needed as input for the classification
        nparr = np.fromstring(request.data, np.uint8)
        # decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = Image.fromarray(img)

        prep = Compose([Resize(1024), ToTensor()])
        x_in = prep(img).to(device)
        x_in = x_in.unsqueeze(0)
        with torch.no_grad():
            model.forward(x_in)
            outs, _ = model.get_gold_predictions()
            probs = F.softmax(outs, dim=1)
            preds = torch.argmax(probs, dim=1)
            preds = preds.cpu()
            preds = np.asarray(preds, dtype=np.uint8)

            pred_tissue = preds[preds != 4]
            sum_tissue = pred_tissue.size

            percentage_nc = np.sum(pred_tissue == 0) / sum_tissue
            percentage_gg3 = np.sum(pred_tissue == 1) / sum_tissue
            percentage_gg4 = np.sum(pred_tissue == 2) / sum_tissue
            percentage_gg5 = np.sum(pred_tissue == 3) / sum_tissue

            return {'% NC': percentage_nc,
                        '% GG3': percentage_gg3,
                        '% GG4': percentage_gg4,
                        '% GG5': percentage_gg5}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6500)

if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.debug')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(logging.DEBUG)
