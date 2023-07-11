import torch
import utils.globals
from segmentation_models_pytorch import Unet, UnetPlusPlus, DeepLabV3Plus, PSPNet, Linknet


def create_segmentation_backbone():
    if utils.globals.config:
        config = utils.globals.config
    else:
        config = {
            'model': {'backbone': 'unet',
                      'encoder': {
                          'backbone': 'resnet34',
                          'weights': 'imagenet'
                      }},
            'data': {'class_no': 5}
        }
    class_no = config['data']['class_no']

    if config['model']['backbone'] == 'unet':
        seg_model = Unet(
            encoder_name=config['model']['encoder']['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config['model']['encoder']['weights'],
            # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=class_no  # model output channels (number of classes in your dataset)
        )
    elif config['model']['backbone'] == 'unetpp':
        seg_model = UnetPlusPlus(
            encoder_name=config['model']['encoder']['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config['model']['encoder']['weights'],
            # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=class_no  # model output channels (number of classes in your dataset)
        )
    elif config['model']['backbone'] == 'deeplabv3p':
        seg_model = DeepLabV3Plus(
            encoder_name=config['model']['encoder']['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config['model']['encoder']['weights'],
            # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=class_no  # model output channels (number of classes in your dataset)
        )
    elif config['model']['backbone'] == 'pspnet':
        seg_model = PSPNet(
            encoder_name=config['model']['encoder']['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config['model']['encoder']['weights'],
            # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=class_no  # model output channels (number of classes in your dataset)
        )
    elif config['model']['backbone'] == 'linknet':
        seg_model = Linknet(
            encoder_name=config['model']['encoder']['backbone'],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=config['model']['encoder']['weights'],
            # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=class_no  # model output channels (number of classes in your dataset)
        )
    else:
        raise Exception('Choose valid model backbone!')
    return seg_model

