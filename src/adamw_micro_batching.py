from os import makedirs
import os

import math
import torch
import torch.nn as nn

from torch.autograd.functional import hvp
from torch.nn.utils import parameters_to_vector
from torch.optim import Optimizer
from pyhessian import hessian

import argparse
from archs import load_architecture
from utilities import get_gd_directory, get_loss_and_acc, compute_losses, \
    save_files, save_files_final, get_hessian_eigenvalues, iterate_dataset
from data import load_dataset, take_first, DATASETS

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def get_esd_plot(eigenvalues, weights, itr):
    density, grids = density_generate(eigenvalues, weights)
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.tight_layout()
    plt.savefig(f"plots/{itr}.png")
    plt.close()
    plt.clf()


def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.01):

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)


# in the basic Lanczos call, the grad is computed once for the batch
# HVP is computed num_iteration times => O(num_itrs + 1) for forward-backward passes per batch

# if you cannot cache grads, the HVP computation is O(num_itrs * batch + num_itrs) for forward-backward passes per batch

# can you cache this by adding to a list, and placing it off GPU
# this line will need to be done by brining one micro-batch computation graph in at a time: grad_v_prod = torch.sum(self.grad_params * v)
class MicrobatchLanczos:
    def __init__(self, num_iterations=20, tol=1e-6):
        self.num_iterations = num_iterations
        self.tol = tol
        self.init_vector = None

    def hvp(self, model, loss_fn, train_dataset, params, v, preconditioner=None):
        hvp = None

        # if preconditioner is not None, we need to compute P^(-1/2) * v
        if preconditioner is not None:
            v = v / preconditioner.sqrt()

        itr = 0
        total_items = 1000
        for (X, y) in iterate_dataset(train_dataset, 100):
            outputs = model(X.cuda())
            loss = loss_fn(outputs, y.cuda()) / total_items

            grad_params = torch.autograd.grad(loss, params, create_graph=True)
            grad_params = torch.cat([g.flatten() for g in grad_params])

            grad_v_prod = torch.sum(grad_params * v)
            _hvp = torch.autograd.grad(grad_v_prod, params, retain_graph=True)
            _hvp = torch.cat([g.flatten() for g in _hvp])

            if hvp is None:
                hvp = _hvp.detach()
            else:
                hvp += _hvp.detach()

            itr += 1
            if itr == 10:
                break

        # if preconditioner is not None, we need to compute P^(-1/2) * hv
        if preconditioner is not None:
            hvp = hvp / preconditioner.sqrt()

        return hvp

    def __call__(self, model, loss_fn, train_dataset, preconditioner=None):
        """
        Runs the Lanczos algorithm to get extreme eigenvalues of the Hessian.
        https://iclr-blogposts.github.io/2024/blog/bench-hvp/
        """
        params = list(model.parameters())
        num_params = sum(p.numel() for p in params)
        device = params[0].device

        # variables
        q_vectors = []
        alpha_list = []
        beta_list = []

        # Initialize the first Lanczos vector (normalized random vector)
        v = torch.randn(num_params, device=device)
        v = v / torch.norm(v)

        if self.init_vector is None:
            self.init_vector = v

        # initialize the 0-th iteration of Lanczos's algorithm
        w = self.hvp(model, loss_fn, train_dataset, params, v, preconditioner)
        alpha = torch.dot(v, w)
        w = w - alpha * v

        # we start collecting from alpha_0
        q_vectors.append(v)
        alpha_list.append(alpha.item())

        for i in range(1, self.num_iterations):
            beta = torch.norm(w)
            v = w / beta

            w = self.hvp(model, loss_fn, train_dataset, params, v, preconditioner)
            alpha = torch.dot(v, w)
            w = w - alpha * v - beta * q_vectors[i - 1]

            # after accessing q_vectors[i - 1], we add current itrerations v to q_vectors
            # here we collect alpha_i and beta_i
            q_vectors.append(v)
            alpha_list.append(alpha.item())
            beta_list.append(beta.item())

            # if beta is less than tolerance in the 1st iteration, we need a minimum of 2x2 T matrix
            if beta < self.tol:
                break

        # Construct the tridiagonal matrix T
        T = torch.diag(torch.tensor(alpha_list, device=device))
        for i in range(len(beta_list)):
            # above the diagonal
            T[i, i + 1] = beta_list[i]

            # below the diagonal
            T[i + 1, i] = beta_list[i]

        # Compute eigenvalues of T
        eigenvalues, eigenvectors = torch.linalg.eigh(T)

        # Recover Ritz vectors from Q and eigenvectors of T
        Q = torch.stack(q_vectors, dim=1)
        largest_index = torch.argmax(eigenvalues)
        smallest_index = torch.argmin(eigenvalues)

        largest_eigenvalue = eigenvalues[largest_index]
        smallest_eigenvalue = eigenvalues[smallest_index]

        largest_ritz_vector = Q @ eigenvectors[:, largest_index]
        smallest_ritz_vector = Q @ eigenvectors[:, smallest_index]

        # get the eigenvalue spectrum density of the Hessian
        eigen_list = eigenvalues.tolist()
        weight_list = torch.pow(eigenvectors[0,:], 2).tolist()

        # TODO: angle between grad, ritz vectors, update vector

        return eigen_list, weight_list


class LanczosAlgorithm:
    def __init__(self, num_iterations=20, tol=1e-6, init_vector=None):
        self.num_iterations = num_iterations
        self.tol = tol

        self.loss = None
        self.grad_params = None
        self.init_vector = init_vector

    def hvp(self, loss, params, v, preconditioner=None):
        if self.grad_params is None:
            grad_params = torch.autograd.grad(loss, params, create_graph=True)
            self.grad_params = torch.cat([g.flatten() for g in grad_params])

        # if preconditioner is not None, we need to compute P^(-1/2) * v
        if preconditioner is not None:
            v = v / preconditioner.sqrt()

        grad_v_prod = torch.sum(self.grad_params * v)
        hvp = torch.autograd.grad(grad_v_prod, params, retain_graph=True)
        hvp = torch.cat([g.flatten() for g in hvp])

        # if preconditioner is not None, we need to compute P^(-1/2) * hv
        if preconditioner is not None:
            hvp = hvp / preconditioner.sqrt()

        return hvp

    def __call__(self, model, loss_fn, data, target, preconditioner=None):
        """
        Runs the Lanczos algorithm to get extreme eigenvalues of the Hessian.
        https://iclr-blogposts.github.io/2024/blog/bench-hvp/
        """
        params = list(model.parameters())
        num_params = sum(p.numel() for p in params)
        device = params[0].device

        # variables
        q_vectors = []
        alpha_list = []
        beta_list = []

        # Forward pass and compute loss, but dont call loss.backward()
        if self.loss is None:
            outputs = model(data)
            loss = loss_fn(outputs, target) / data.shape[0]
            self.loss = loss

        # Initialize the first Lanczos vector (normalized random vector)
        if self.init_vector is not None:
            v = self.init_vector
        else:
            v = torch.randn(num_params, device=device)
            v = v / torch.norm(v)

        # initialize the 0-th iteration of Lanczos's algorithm
        w = self.hvp(self.loss, params, v, preconditioner)
        alpha = torch.dot(v, w)
        w = w - alpha * v

        # we start collecting from alpha_0
        q_vectors.append(v)
        alpha_list.append(alpha.item())

        for i in range(1, self.num_iterations):
            beta = torch.norm(w)
            v = w / beta

            w = self.hvp(self.loss, params, v, preconditioner)
            alpha = torch.dot(v, w)
            w = w - alpha * v - beta * q_vectors[i - 1]

            # after accessing q_vectors[i - 1], we add current itrerations v to q_vectors
            # here we collect alpha_i and beta_i
            q_vectors.append(v)
            alpha_list.append(alpha.item())
            beta_list.append(beta.item())

            # if beta is less than tolerance in the 1st iteration, we need a minimum of 2x2 T matrix
            if beta < self.tol:
                break

        # Construct the tridiagonal matrix T
        T = torch.diag(torch.tensor(alpha_list, device=device))
        for i in range(len(beta_list)):
            # above the diagonal
            T[i, i + 1] = beta_list[i]

            # below the diagonal
            T[i + 1, i] = beta_list[i]

        # Compute eigenvalues of T
        eigenvalues, eigenvectors = torch.linalg.eigh(T)

        # Recover Ritz vectors from Q and eigenvectors of T
        Q = torch.stack(q_vectors, dim=1)
        largest_index = torch.argmax(eigenvalues)
        smallest_index = torch.argmin(eigenvalues)

        largest_eigenvalue = eigenvalues[largest_index]
        smallest_eigenvalue = eigenvalues[smallest_index]

        largest_ritz_vector = Q @ eigenvectors[:, largest_index]
        smallest_ritz_vector = Q @ eigenvectors[:, smallest_index]

        # get the eigenvalue spectrum density of the Hessian
        eigen_list = eigenvalues.tolist()
        weight_list = torch.pow(eigenvectors[0,:], 2).tolist()

        # TODO: angle between grad, ritz vectors, update vector

        return eigen_list, weight_list


class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)

        self._param_tensors = []
        self._grad_tensors = []

        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_adam_nu(optimizer) -> torch.Tensor:
    vec = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            vec.append(state['exp_avg_sq'].view(-1))
    return torch.cat(vec)

def main(dataset: str, arch_id: str, loss: str, opt: str,
         lr: float, beta1: float, beta2: float, epsilon: float,
         max_steps: int, neigs: int = 0,
         physical_batch_size: int = 1000, eig_freq: int = -1, iterate_freq: int = -1, save_freq: int = -1,
         save_model: bool = False, beta: float = 0.0, nproj: int = 0,
         loss_goal: float = None, acc_goal: float = None, abridged_size: int = 5000, seed: int = 0):
    results_dir = os.environ["RESULTS"]
    directory = f"{results_dir}/{dataset}/{arch_id}/seed_{seed}/{loss}/adam/lr_{lr}_beta1_{beta1}_beta2_{beta2}_eps_{epsilon}"
    print(f"output directory: {directory}")
    makedirs(directory, exist_ok=True)

    train_dataset, test_dataset = load_dataset(dataset, loss)
    abridged_train = take_first(train_dataset, abridged_size)

    loss_fn, acc_fn = get_loss_and_acc(loss)

    torch.manual_seed(seed)
    network = load_architecture(arch_id, dataset).cuda()

    torch.manual_seed(7)
    projectors = torch.randn(nproj, len(parameters_to_vector(network.parameters())))

    optimizer = AdamW(network.parameters(), lr, (beta1, beta2), epsilon)

    train_loss, test_loss, train_acc, test_acc = \
        torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps), torch.zeros(max_steps)
    iterates = torch.zeros(max_steps // iterate_freq if iterate_freq > 0 else 0, len(projectors))
    eigs = torch.zeros(max_steps // eig_freq if eig_freq >= 0 else 0, neigs)

    for step in range(0, max_steps):
        optimizer.zero_grad()

        lanczos = MicrobatchLanczos(num_iterations=20)
        eigen_list1, weight_list1 = lanczos(network, loss_fn, train_dataset)
        init_vector = lanczos.init_vector

        # don't use micro-batch to match computed eigenvalues
        lanczos = LanczosAlgorithm(num_iterations=20, init_vector=init_vector)
        itr = 0
        # micro-batching begins
        for (X, y) in iterate_dataset(train_dataset, 1000):
            # TODO: lanczos
            eigen_list2, weight_list2 = lanczos(network, loss_fn, X, y)
            itr += 1

            if itr == 1:
                break

        # TODO: assert grads and Hessians computed over 10 batches vs for 1 batch is the same
        import ipdb
        ipdb.set_trace()


        # micro-batching for grad updates
        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            
            eigen_list, weight_list = lanczos(network, loss_fn, X, y, preconditioner)
            get_esd_plot([eigen_list], [weight_list], itr)


            loss = loss_fn(network(X.cuda()), y.cuda()) / len(train_dataset)
            loss.backward()

            print(f"Step {step}, Loss: {loss.item()}")

            optimizer.step()

    save_files_final(directory,
                     [("eigs", eigs[:(step + 1) // eig_freq]), ("iterates", iterates[:(step + 1) // iterate_freq]),
                      ("train_loss", train_loss[:step + 1]), ("test_loss", test_loss[:step + 1]),
                      ("train_acc", train_acc[:step + 1]), ("test_acc", test_acc[:step + 1])])
    if save_model:
        torch.save(network.state_dict(), f"{directory}/snapshot_final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train using gradient descent.")
    parser.add_argument("dataset", type=str, choices=DATASETS, help="which dataset to train")
    parser.add_argument("arch_id", type=str, help="which network architectures to train")
    parser.add_argument("loss", type=str, choices=["ce", "mse"], help="which loss function to use")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--opt", type=str, choices=["gd", "polyak", "nesterov"],
                        help="which optimization algorithm to use", default="gd")
    parser.add_argument("--seed", type=int, help="the random seed used when initializing the network weights",
                        default=0)
    parser.add_argument("--beta1", type=float, help="Adam beta1 parameter", default=0.9)
    parser.add_argument("--beta2", type=float, help="Adam beta2 parameter", default=0.995)
    parser.add_argument("--epsilon", type=float, help="Adam epsilon parameter", default=1e-7)
    parser.add_argument("--physical_batch_size", type=int,
                        help="the maximum number of examples that we try to fit on the GPU at once", default=1000)
    parser.add_argument("--acc_goal", type=float,
                        help="terminate training if the train accuracy ever crosses this value")
    parser.add_argument("--loss_goal", type=float, help="terminate training if the train loss ever crosses this value")
    parser.add_argument("--neigs", type=int, help="the number of top eigenvalues to compute")
    parser.add_argument("--eig_freq", type=int, default=-1,
                        help="the frequency at which we compute the top Hessian eigenvalues (-1 means never)")
    parser.add_argument("--nproj", type=int, default=0, help="the dimension of random projections")
    parser.add_argument("--iterate_freq", type=int, default=-1,
                        help="the frequency at which we save random projections of the iterates")
    parser.add_argument("--abridged_size", type=int, default=5000,
                        help="when computing top Hessian eigenvalues, use an abridged dataset of this size")
    parser.add_argument("--save_freq", type=int, default=-1,
                        help="the frequency at which we save resuls")
    parser.add_argument("--save_model", type=bool, default=False,
                        help="if 'true', save model weights at end of training")
    args = parser.parse_args()

    main(dataset=args.dataset, arch_id=args.arch_id, loss=args.loss, opt=args.opt, lr=args.lr, max_steps=args.max_steps,
         neigs=args.neigs, physical_batch_size=args.physical_batch_size, eig_freq=args.eig_freq,
         iterate_freq=args.iterate_freq, save_freq=args.save_freq, save_model=args.save_model, beta1=args.beta1,
         beta2=args.beta2, epsilon=args.epsilon, nproj=args.nproj, loss_goal=args.loss_goal,
         acc_goal=args.acc_goal, abridged_size=args.abridged_size, seed=args.seed)
