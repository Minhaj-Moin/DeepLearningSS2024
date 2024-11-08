import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
import numpy as np
# from tqdm import tqdm


class Trainer:

    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        self._optim.zero_grad()
        out = self._model(x)    # -propagate through the network
        loss = self._crit(out.to(t.float32), y.to(t.float32))   # -calculate the loss
        loss.backward()         # -compute gradient by backward propagation
        self._optim.step()      # -update weights
        return loss.item()      # -return the loss



    def val_test_step(self, x, y):              # predict
        pred = self._model(x)                   # propagate through the network and calculate the loss and predictions
        return self._crit(pred, y).item(), pred # return the loss and the predictions

    def train_epoch(self):
        self._model.train()                         # set training mode
        average_loss = 0
        for data in self._train_dl:                 # iterate through the training set
            x, y = data
            y = y.to(t.float32)
            if self._cuda:                          # transfer the batch to "cuda()" -> the gpu if a gpu is given
                x, y = x.cuda(), y.cuda()           # perform a training step
            average_loss += self.train_step(x, y)   # calculate the average loss for the epoch and return it

        return average_loss / len(self._train_dl)

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        self._model.eval()
        average_test_loss = 0.0
        ypred = []
        ytrue = []
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        with t.no_grad():
            # iterate through the validation set
            for data in self._val_test_dl:
                x, y = data
                y = y.to(t.float32)
                ytrue.append(y.detach().numpy().astype(float))
                # transfer the batch to the gpu if given
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()
                # perform a validation step
                loss, pred = self.val_test_step(x, y)
                ypred.append(pred.detach().cpu().numpy())   ## if cuda is used we have to detach().cpu().numpy() to convert to np

                average_test_loss += loss                   ## save the predictions and the labels for each batch

            # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions

            ytrue = np.resize(ytrue,(len(ytrue),2))
            ypred = np.resize(ypred,(len(ypred),2))
            ypred = np.round(ypred, 0)
            f1score = f1_score(ytrue, ypred, average=None)
            average_test_loss = average_test_loss / len(self._val_test_dl)
            print("TEST: average loss and f1 score: {:.3f}, ".format(average_test_loss), f1score, f1score.mean(), end=' ')
            return average_test_loss    # return the loss and print the calculated metrics


    def fit(self, epochs=-1):
        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        train_losses, val_losses = [], []
        loss_counter = 0
        curr_loss = np.inf
        min_val_loss = np.inf

        for i in range(epochs):
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            train_losses.append(self.train_epoch())
            print("TRAIN: average loss {:.3f}".format(train_losses[i]), end=' | ')
            val_losses.append(self.val_test())

            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            if (val_losses[i] < min_val_loss):
                self.save_checkpoint(i)
                print("Saving Checkpoint 'checkpoints/checkpoint_{:03d}.ckp'".format(i))
                min_val_loss = val_losses[i]
            print(f"Current Loss:{curr_loss} | Minimum loss: {min_val_loss}")
            if train_losses[i] < curr_loss:
                loss_counter = 0 ## reset counter for early_stopping_patience, or else increase if loss is >= curr_loss
            else: loss_counter += 1
            curr_loss = train_losses[i]
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            if (loss_counter >= self._early_stopping_patience): break

        return train_losses, val_losses
        
