import copy
import itertools
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import tqdm


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


def LogSumExp(x, dim=0):
    m, _ = torch.max(x, dim=dim, keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(dir, epoch, name="checkpoint", **kwargs):
    state = {"epoch": epoch}
    state.update(kwargs)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def train_epoch(
    loader,
    model,
    criterion,
    optimizer,
    cuda=True,
    regression=False,
    verbose=False,
    subset=None,
    loader_scaler=None,
):
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        try:
            current_batch_size = input[0].size(0)
        except:
            current_batch_size = input.size(0)

        if cuda:
            input = (
                input[0].cuda(non_blocking=True),
                input[1].cuda(non_blocking=True),
            )
            target = target.cuda(non_blocking=True)
            model.cuda()

        input, target = loader_scaler.scale(input, target)

        loss, output = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data.item() * current_batch_size

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += current_batch_size

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print(
                "Stage %d/10. Loss: %12.4f. Acc: %6.2f"
                % (
                    verb_stage + 1,
                    loss_sum / num_objects_current,
                    correct / num_objects_current * 100.0,
                )
            )
            verb_stage += 1

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None
        if regression
        else correct / num_objects_current * 100.0,
    }


def train_epoch_adversarial(
    loader,
    model,
    criterion,
    optimizer,
    cuda=True,
    regression=False,
    verbose=False,
    subset=None,
    loader_scaler=None,
):
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = (
                input[0].cuda(non_blocking=True),
                input[1].cuda(non_blocking=True),
            )
            target = target.cuda(non_blocking=True)
            model.cuda()

        input, target = loader_scaler.scale(input, target)
        try:
            current_batch_size = input[0].size(0)
        except:
            current_batch_size = input.size(0)

        optimizer.zero_grad()

        # loss, output = criterion(model, input, target)
        # ESSA PARTE SUBSTITUI O CRITERION COMPLETAMENTE
        #####################

        epsilon = 0.01
        lossfn = F.gaussian_nll_loss

        # scale epsilon by min and max (should be [0,1] for all experiments)
        # see algorithm 1 of paper
        (wtg, fct) = input
        scaled_epsilon_0 = epsilon * (wtg.max() - wtg.min())
        scaled_epsilon_1 = epsilon * (fct.max() - fct.min())

        # force inputs to require gradient
        input[0].requires_grad = True
        input[1].requires_grad = True

        # standard forwards pass
        output, variance = model(input)
        loss = lossfn(output, target, variance)

        # now compute gradients wrt input
        loss.backward(retain_graph=True)

        # now compute sign of gradients
        inputs_grad_0 = torch.sign(input[0].grad)
        inputs_grad_1 = torch.sign(input[1].grad)

        # perturb inputs and use clamped output
        inputs_perturbed_0 = torch.clamp(
            input[0] + scaled_epsilon_0 * inputs_grad_0, 0.0, 1.0
        ).detach()
        inputs_perturbed_1 = torch.clamp(
            input[1] + scaled_epsilon_1 * inputs_grad_1, 0.0, 1.0
        ).detach()
        inputs_perturbed_0.requires_grad = False
        inputs_perturbed_1.requires_grad = False
        inputs_perturbed = (inputs_perturbed_0, inputs_perturbed_1)
        if cuda:
            inputs_perturbed = (
                inputs_perturbed[0].cuda(non_blocking=True),
                inputs_perturbed[1].cuda(non_blocking=True),
            )
            target = target.cuda(non_blocking=True)
            model.cuda()

        input[0].grad.zero_()
        input[1].grad.zero_()
        # model.zero_grad()

        outputs_perturbed, variance_perturbed = model(inputs_perturbed)

        # compute adversarial version of loss
        adv_loss = lossfn(outputs_perturbed, target, variance_perturbed)
        adv_loss.backward()

        #####################
        optimizer.step()

        # return mean of losses
        loss = (loss + adv_loss) / 2.0

        loss_sum += loss.data.item() * current_batch_size

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += current_batch_size

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print(
                "Stage %d/10. Loss: %12.4f. Acc: %6.2f"
                % (
                    verb_stage + 1,
                    loss_sum / num_objects_current,
                    correct / num_objects_current * 100.0,
                )
            )
            verb_stage += 1

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None
        if regression
        else correct / num_objects_current * 100.0,
    }


def eval(
    loader,
    model,
    criterion,
    cuda=True,
    regression=False,
    verbose=False,
    loader_scaler=None,
):
    loss_sum = 0.0
    correct = 0.0
    num_objects_current = 0
    num_batches = len(loader)

    model.eval()

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(loader):
            if cuda:
                input = (
                    input[0].cuda(non_blocking=True),
                    input[1].cuda(non_blocking=True),
                )
                target = target.cuda(non_blocking=True)
                model.cuda()

            input, target = loader_scaler.scale(input, target)
            try:
                current_batch_size = input[0].size(0)
            except:
                current_batch_size = input.size(0)

            loss, output = criterion(model, input, target)

            loss_sum += loss.item() * current_batch_size

            num_objects_current += current_batch_size

            if not regression:
                pred = output.data.argmax(1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None
        if regression
        else correct / num_objects_current * 100.0,
    }


# def predict(loader, model, verbose=False):
#     predictions = list()
#     targets = list()

#     model.eval()

#     if verbose:
#         loader = tqdm.tqdm(loader)

#     offset = 0
#     with torch.no_grad():
#         for input, target in loader:
#             input = input.cuda(non_blocking=True)
#             output = model(input)

#             batch_size = input.size(0)
#             predictions.append(F.softmax(output, dim=1).cpu().numpy())
#             targets.append(target.numpy())
#             offset += batch_size

#     return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}


def predict(
    loader, model, verbose=False, use_training_true=False, loader_scaler=None
):
    predictions = list()
    targets = list()
    if use_training_true:
        model.train()
    else:
        model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)
    with torch.no_grad():
        for input, target in loader:
            input = (
                input[0].cuda(non_blocking=True),
                input[1].cuda(non_blocking=True),
            )
            model.cuda()
            target = target.cuda(non_blocking=True)
            input, target = loader_scaler.scale(input, target)
            output = model(input)
            output = output.cuda(non_blocking=True)
            output = loader_scaler.unscale_output(output)
            target = loader_scaler.unscale_output(target)
            predictions.append(output)
            targets.append(target)

    return {
        "predictions": torch.vstack(predictions),
        "targets": torch.concatenate(targets),
    }


def predict_gaussian(loader, model, verbose=False, loader_scaler=None):
    predictions = list()
    variances = []
    targets = list()

    if verbose:
        loader = tqdm.tqdm(loader)
    with torch.no_grad():
        for input, target in loader:
            input = (
                input[0].cuda(non_blocking=True),
                input[1].cuda(non_blocking=True),
            )
            target = target.cuda(non_blocking=True)
            model.cuda()

            input, target = loader_scaler.scale(input, target)
            output, variance = model(input)
            output = loader_scaler.unscale_output(output)

            variance = loader_scaler.unscale_output_variance(variance)
            target = loader_scaler.unscale_output(target)
            predictions.append(output)
            variances.append(variance)
            targets.append(target)

    return {
        "predictions": (torch.vstack(predictions), torch.vstack(variances)),
        "targets": torch.concatenate(targets),
    }


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= 1.0 - alpha
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
    BatchNorm buffers update (if any).
    Performs 1 epochs to estimate buffers average using train dataset.

    :param loader: train dataset loader for buffers average estimation.
    :param model: model being update
    :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:
            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = (
                input[0].cuda(non_blocking=True),
                input[1].cuda(non_blocking=True),
            )
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))


def inv_softmax(x, eps=1e-10):
    return torch.log(x / (1.0 - x + eps))


def predictions(
    test_loader, model, seed=None, cuda=True, regression=False, **kwargs
):
    # will assume that model is already in eval mode
    # model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        if seed is not None:
            torch.manual_seed(seed)
        if cuda:
            input = (
                input[0].cuda(non_blocking=True),
                input[1].cuda(non_blocking=True),
            )
            model.cuda()
        output = model(input, **kwargs)
        if regression:
            preds.append(output.cpu().data.numpy())
        else:
            probs = F.softmax(output, dim=1)
            preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
    t = (epoch) / (swa_start if swa else epochs)
    lr_ratio = swa_lr / lr_init if swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return lr_init * factor
