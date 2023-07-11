import torch
import yaml
import argparse
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import resnet34
from torchvision.transforms import Compose, Resize, ToTensor
from torch.nn.functional import softmax


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open('./example_images/images/slide001_core059.png')

    prep = Compose([Resize(1024), ToTensor()])
    x_in = prep(img).to(device)
    x_in = x_in.unsqueeze(0)

    loaded_model = torch.load('./model/model.pt')
    model = loaded_model
    model.eval()

    with torch.no_grad():
        model.forward(x_in)
        outs, _ = model.get_gold_predictions()
        probs = F.softmax(outs, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds = preds.cpu()
        preds = np.asarray(preds, dtype=np.uint8)

        pred_tissue = preds[preds != 4]
        sum_tissue = np.sum(pred_tissue)

        percentage_nc = np.sum(pred_tissue == 0) / sum_tissue
        percentage_gg3 = np.sum(pred_tissue == 1) / sum_tissue
        percentage_gg4 = np.sum(pred_tissue == 2) / sum_tissue
        percentage_gg5 = np.sum(pred_tissue == 3) / sum_tissue
        print({'% NC': percentage_nc,
                    '% GG3': percentage_gg3,
                    '% GG4': percentage_gg4,
                    '% GG5': percentage_gg5})


if __name__ == "__main__":
    main()
