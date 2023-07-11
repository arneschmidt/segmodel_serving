import torch
import yaml
import argparse
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import resnet34
from torchvision.transforms import Compose, Resize, ToTensor
from torch.nn.functional import softmax
# from utils.model_pionono import PiononoModel

# import utils.globals as globals


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = Image.open('./example_images/images/slide001_core059.png')

    prep = Compose([Resize(1024), ToTensor()])
    x_in = prep(img).to(device)
    x_in = x_in.unsqueeze(0)
    # config = globals.config

    # model = PiononoModel(input_channels=3, num_classes=5,
    #                      annotators=['gold', '1', '2', '3', '4', '5', '6'],
    #                      gold_annotators=config['model']['pionono_config']['gold_annotators'],
    #                      latent_dim=config['model']['pionono_config']['latent_dim'],
    #                      no_head_layers=config['model']['pionono_config']['no_head_layers'],
    #                      head_kernelsize=1,
    #                      head_dilation=1,
    #                      kl_factor=config['model']['pionono_config']['kl_factor'],
    #                      reg_factor=config['model']['pionono_config']['reg_factor'],
    #                      mc_samples=config['model']['pionono_config']['mc_samples'],
    #                      z_prior_sigma=config['model']['pionono_config']['z_prior_sigma'],
    #                      z_posterior_init_sigma=config['model']['pionono_config']['z_posterior_init_sigma'],
    #                      )

    loaded_model = torch.load('./model-raw/model.pt')
    # model.load_state_dict(loaded_model.state_dict())
    # torch.save(loaded_model.state_dict(), './model-raw/model_state_dict.pt')
    # torch.jit.save(loaded_model, './model-raw/model_jit.pt')
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
    # parser = argparse.ArgumentParser(description="Cancer Classification")
    # parser.add_argument("--config", "-c", type=str, default="./config.yaml",
    #                     help="Config path (yaml file expected) to default config.")
    # parser.add_argument("--dataset_config", "-dc", type=str,
    #                     default="./dataset_dependent/gleason19/data_configs/data_config_crossval0.yaml",
    #                     help="Config path (yaml file expected) to dataset config. Parameters will override defaults.")
    # args = parser.parse_args()
    # args.experiment_folder = 'None'
    # globals.init_global_config(args)
    main()
