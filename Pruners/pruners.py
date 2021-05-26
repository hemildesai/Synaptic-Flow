from collections import OrderedDict
import copy

import torch
import torch.nn.functional as F
import numpy as np

output_activations = []


class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally."""
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = torch.tensor([0.0]).to(mask.device)
                one = torch.tensor([1.0]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise."""
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.0]).to(mask.device)
                one = torch.tensor([1.0]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope."""
        if scope == "global":
            self._global_mask(sparsity)
        if scope == "local":
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters."""
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model."""
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v ** 2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters."""
        remaining_params, total_params = 0, 0

        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


def get_activations(module, input, output):
    global output_activations
    output_activations.append(torch.clone(output))


class TaylorPruner(Pruner):
    def __init__(self, masked_parameters):
        super().__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        self.diagonals = []
        global output_activations

        self.generate_mapping(model)
        self.register_hooks(model)

        # with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            output_activations = []
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()
            break

        assert len(output_activations) == len(self.mapping)
        for activation, (layer_index, param_index) in zip(
            output_activations, self.mapping
        ):
            params = self.masked_parameters[param_index][1]
            neuron_scores = torch.std(activation, dim=0)
            scores = neuron_scores.unsqueeze(1).repeat(1, params.shape[-1])
            self.scores[id(params)] = scores
            act_mean = torch.mean(torch.clone(activation), dim=0).detach()
            act_mean.requires_grad = True
            output = F.relu(act_mean)
            # output = F.sigmoid(act_mean)
            output.backward(torch.ones_like(act_mean))
            diag = torch.diag(act_mean.grad)
            self.diagonals.append(diag)

        self.deregister_hooks(model)

    def add_skip_weights(self, model):
        global output_activations
        for i, (layer_index, param_index) in enumerate(self.mapping):
            layer = model[layer_index]
            mask_complement = 1 - layer.weight_mask
            # for j in range(i+1, len(self.mapping)):
            if i + 1 < len(self.mapping):
                next_layer_index = self.mapping[i + 1][0]
                next_layer = model[next_layer_index]
                neuron_mask = mask_complement[:, 0]
                next_layer_mask = neuron_mask.unsqueeze(0).repeat(
                    next_layer.weight.shape[0], 1
                )
                diag = self.diagonals[i]

                w1 = torch.clone(layer.weight * mask_complement).detach()
                w2 = torch.clone(next_layer.weight * next_layer_mask).detach()
                w_c = w1.T @ diag @ w2.T
                w_c = w_c.T
                w_c.requires_grad = True

                b1 = torch.clone(layer.bias * neuron_mask).detach()
                act_mean = torch.mean(torch.clone(output_activations[i]), dim=0).detach()
                b_c = w2 @ (F.relu(act_mean) + diag @ (b1 - act_mean))
                b_c.requires_grad = True

                next_layer.add_skip_weights(w_c, b_c)

            if i + 2 < len(self.mapping):
                next_layer_index = self.mapping[i + 2][0]
                next_layer = model[next_layer_index]
                neuron_mask = mask_complement[:, 0]
                next_layer_mask = neuron_mask.unsqueeze(0).repeat(
                next_layer.weight.shape[0], 1
                )
                diag = self.diagonals[i]

                w1 = torch.clone(layer.weight * mask_complement).detach()
                w2 = torch.clone(next_layer.weight * next_layer_mask).detach()
                w_c = w1.T @ diag @ w2.T
                w_c = w_c.T
                w_c.requires_grad = True

                b1 = torch.clone(layer.bias * neuron_mask).detach()
                act_mean = torch.mean(torch.clone(output_activations[i]), dim=0).detach()
                b_c = w2 @ (F.relu(act_mean) + diag @ (b1 - act_mean))
                b_c.requires_grad = True

                next_layer.add_skip_weights_2(w_c, b_c)

    def retrieve_activations(self):
        global output_activations
        print(len(output_activations))

    def register_hooks(self, model):
        self.old_hooks = []
        for index, _ in self.mapping:
            layer = model[index]
            self.old_hooks.append(copy.deepcopy(layer._forward_hooks))
            layer.register_forward_hook(get_activations)

    def deregister_hooks(self, model):
        for i, (index, _) in enumerate(self.mapping):
            layer = model[index]
            layer._forward_hooks = self.old_hooks[i]

    def generate_mapping(self, model):
        self.mapping = []
        counter = 0
        for i, layer in enumerate(model):
            if not hasattr(layer, "weight"):
                continue

            weights = layer.weight
            while counter < len(self.masked_parameters):
                if bool(torch.all(weights == self.masked_parameters[counter][1])):
                    self.mapping.append((i, counter))
                    counter += 1
                    break

                counter += 1


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(
                L, [p for (_, p) in self.masked_parameters], create_graph=False
            )
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(
                L, [p for (_, p) in self.masked_parameters], create_graph=True
            )
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(
            device
        )  # , dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)
