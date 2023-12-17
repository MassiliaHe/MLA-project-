# Import libraries
import json
import numpy as np
import torch
import torch.nn
from logging import getLogger
import itertools
# from FaderNetwork.autoencoder import modify_predictions, toggle_attributes

# Initialize eval_logger
eval_logger = getLogger()


class ModelEvaluator(object):

    def __init__(self, autoencoder, discriminator, dataloader, args):
        """
        Initialize the model evaluator.
        """
        # Assign dataset and settings
        self.dataloader = dataloader
        self.args = args

        # Assign model components
        self.autoencoder = autoencoder
        self.discriminator = discriminator
    
    def eval_reconstruction_loss(self):
        """
        Compute the autoencoder reconstruction perplexity.
        """
        self.autoencoder.eval()

        costs = []
        limited_val_dataloader = itertools.islice(self.dataloader, self.args.val_slice)
        for iter, (images, attributes) in enumerate(limited_val_dataloader):
            images, attributes = images.to(self.args.device), attributes.to(self.args.device)
            _, dec_outputs = self.autoencoder(images, attributes)
            costs.append(((dec_outputs[-1] - images) ** 2).mean().item())

        return np.mean(costs)

    def eval_disc_accu(self):
        """
        Compute the discriminator prediction accuracy.
        """
        self.autoencoder.eval()
        self.discriminator.eval()
        limited_val_dataloader = itertools.islice(self.dataloader, self.args.val_slice)
        all_preds = []
        for iter, (images, attributes) in enumerate(limited_val_dataloader):
            images, attributes = images.to(self.args.device), attributes.to(self.args.device)
            enc_outputs = self.autoencoder.encode(images)
            preds = self.discriminator(enc_outputs).data
            all_preds.append((preds.max(1)[1] == attributes.max(1)[1]).tolist())

        return [np.mean(x) for x in all_preds]

    def evaluate(self, n_epoch):
        """
        Evaluate all models / log evaluation results.
        """
        eval_logger.info('')

        # reconstruction loss
        ae_loss = self.eval_reconstruction_loss()

        # log autoencoder loss
        eval_logger.info('Autoencoder loss: %.5f' % ae_loss)

        # discriminator accuracy
        log_disc = []
        discriminator_accu = self.eval_disc_accu()
        log_disc.append(('disc_accu', np.mean(discriminator_accu)))
        # eval_logger.info('discriminator accuracy:')
        print_accuracies(log_disc)

        # JSON log
        to_log = dict([
            ('n_epoch', n_epoch),
            ('ae_loss', ae_loss)
        ] + log_disc)

        eval_logger.debug("__log__:%s" % json.dumps(to_log))

        return to_log
    
    def update_models(self, autoencoder, discriminator):
        self.autoencoder = autoencoder
        self.discriminator = discriminator
    

def print_accuracies(values):
    """
    Pretty plot of accuracies.
    """
    assert all(len(x) == 2 for x in values)
    for name, value in values:
        eval_logger.info('{:<20}: {:>6}'.format(name, '%.3f%%' % (100 * value)))
    eval_logger.info('')