import torch
from torch import nn
import utils.globals
import numpy as np
from Probabilistic_Unet_Pytorch.utils import l2_regularisation
from utils.model_headless import UnetHeadless
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from utils.model_pionono_layers import LatentVariable, PiononoHead


class PiononoModel(nn.Module):
    """
    The implementation of the Pionono Model. It consists of a segmentation backbone, probabilistic latent variable and
    segmentation head.
    """

    def __init__(self, input_channels=3, num_classes=5, annotators=['Maps/STAPLE', 'Maps/Maps1_T', 'Maps/Maps2_T', 'Maps/Maps3_T', 'Maps/Maps4_T', 'Maps/Maps5_T', 'Maps/Maps6_T'],
                 gold_annotators=[0], latent_dim=8,
                 z_prior_mu=0.0, z_prior_sigma=2.0, z_posterior_init_sigma=8.0, no_head_layers=3, head_kernelsize=1,
                 head_dilation=1, kl_factor=1.0, reg_factor=0.1, mc_samples=5):
        super(PiononoModel, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.annotators = annotators
        self.gold_annotators = gold_annotators
        self.no_head_layers = no_head_layers
        self.head_kernelsize = head_kernelsize
        self.head_dilation = head_dilation
        self.kl_factor = kl_factor
        self.reg_factor = reg_factor
        self.train_mc_samples = mc_samples
        self.test_mc_samples = 20
        self.unet = UnetHeadless().to(device)
        self.z = LatentVariable(len(annotators), latent_dim, prior_mu_value=z_prior_mu, prior_sigma_value=z_prior_sigma,
                                z_posterior_init_sigma=z_posterior_init_sigma).to(device)
        self.head = PiononoHead(16, self.latent_dim, self.input_channels, self.num_classes,
                                self.no_head_layers, self.head_kernelsize, self.head_dilation, use_tile=True).to(device)
        self.phase = 'segmentation'
        self.name = 'PiononoModel'

    def forward(self, patch):
        """
        Get feature maps.
        """
        self.unet_features = self.unet.forward(patch)

    def map_annotators_to_correct_id(self, annotator_ids: torch.tensor, annotator_list:list = None):
        new_ids = torch.zeros_like(annotator_ids).to(device)
        for a in range(len(annotator_ids)):
            id_corresponds = (annotator_list[int(annotator_ids[a])] == np.array(self.annotators))
            if not np.any(id_corresponds):
                raise Exception('Annotator has no corresponding distribution. Annotator: ' + str(annotator_list[int(annotator_ids[a])]))
            new_ids[a] = torch.nonzero(torch.tensor(annotator_list[int(annotator_ids[a])] == np.array(self.annotators)))[0][0]
        return new_ids

    def sample(self, use_z_mean: bool, annotator_ids: torch.tensor, annotator_list: list = None, use_softmax=True):
        """
        Get sample of output distribution. Annotator list defines the distributions (q|r) that are used.
        """
        if annotator_list is not None:
            annotator_ids = self.map_annotators_to_correct_id(annotator_ids, annotator_list)

        if use_z_mean == False:
            z = self.z.forward(annotator_ids, sample=True)
        else:
            z = self.z.forward(annotator_ids, sample=False)
        pred = self.head.forward(self.unet_features, z, use_softmax)

        return pred

    def get_gold_predictions(self):
        """
        Get gold predictions (based on the gold distribution).
        """
        if len(self.gold_annotators) == 1:
            annotator = torch.ones(self.unet_features.shape[0]).to(device) * self.gold_annotators[0]
            mean, std = self.mc_sampling(annotator, use_softmax=True)
        else:
            shape = [self.train_mc_samples * len(self.gold_annotators), self.unet_features.shape[0], self.num_classes,
                     self.unet_features.shape[-2], self.unet_features.shape[-1]]
            samples = torch.zeros(shape).to(device)
            for a in range(len(self.gold_annotators)):
                for i in range(self.train_mc_samples):
                    annotator_ids = torch.ones(self.unet_features.shape[0]).to(device) * self.gold_annotators[a]
                    samples[(a * self.train_mc_samples) + i] = self.sample(use_z_mean=False,
                                                                           annotator_ids=annotator_ids,
                                                                           use_softmax=True)
            mean = torch.mean(samples, dim=0)
            std = torch.std(samples, dim=0)
        return mean, std

    def mc_sampling(self, annotator: torch.tensor = None, use_softmax=True):
        """
        Monte-Carlo sampling to get mean and std of output distribution.
        """
        if self.training:
            mc_samples = self.train_mc_samples
        else:
            mc_samples = self.test_mc_samples
        shape = [mc_samples, annotator.shape[0], self.num_classes, self.unet_features.shape[-2], self.unet_features.shape[-1]]
        samples = torch.zeros(shape).to(device)
        for i in range(mc_samples):
            samples[i] = self.sample(use_z_mean=False, annotator_ids=annotator, use_softmax=use_softmax)
        mean = torch.mean(samples, dim=0)
        std = torch.std(samples, dim=0)
        return mean, std

    def elbo(self, labels: torch.tensor, loss_fct, annotator: torch.tensor):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        # self.preds = self.sample(use_z_mean=False, annotator=annotator)
        self.preds, _ = self.mc_sampling(annotator=annotator, use_softmax=False)
        self.log_likelihood_loss = loss_fct(self.preds, labels)
        self.kl_loss = self.z.get_kl_loss(annotator) * self.kl_factor

        return -(self.log_likelihood_loss + self.kl_loss)

    def combined_loss(self, labels, loss_fct, annotator):
        """
        Combine ELBO with regularization of deep network weights.
        """
        elbo = self.elbo(labels, loss_fct=loss_fct, annotator=annotator)
        self.reg_loss = l2_regularisation(self.head.layers) * self.reg_factor
        loss = -elbo + self.reg_loss
        return loss

    def train_step(self, images, labels, loss_fct, ann_ids):
        """
        Make one train step, returning loss and predictions.
        """
        self.forward(images)
        loss = self.combined_loss(labels, loss_fct, ann_ids)
        y_pred = self.preds

        return loss, y_pred

