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


class LanczosAlgorithm:
    def __init__(self, num_iterations=50, tol=1e-6):
        self.num_iterations = num_iterations
        self.tol = tol

        # beta are off-diagonal elements
        # alphas are diagonal elements
        self.alphas = []
        self.betas = []
        self.q_vectors = []

        self.loss = None
        self.grad_params = None

    def hvp(self, loss, params, v):
        if self.grad_params is None:
            grad_params = torch.autograd.grad(loss, params, create_graph=True)
            self.grad_params = torch.cat([g.flatten() for g in grad_params])

        grad_v_prod = torch.sum(self.grad_params * v)
        hvp = torch.autograd.grad(grad_v_prod, params, retain_graph=True)

        return torch.cat([g.flatten() for g in hvp])

    def __call__(self, model, loss_fn, data, target):
        """
        Runs the Lanczos algorithm to get extreme eigenvalues of the Hessian.
        https://iclr-blogposts.github.io/2024/blog/bench-hvp/
        """
        params = list(model.parameters())
        num_params = sum(p.numel() for p in params)
        device = params[0].device

        # Forward pass and compute loss
        outputs = model(data)
        loss = loss_fn(outputs, target)
        self.loss = loss

        # Initialize the first Lanczos vector (normalized random vector)
        v = torch.randn(num_params, device=device)
        v = v / torch.norm(v)

        # initialize the 0-th iteration of Lanczos's algorithm
        w = self.hvp(self.loss, params, v)
        alpha = torch.dot(v, w)
        w = w - alpha * v

        # we start collecting from alpha_0
        self.q_vectors.append(v)
        self.alphas.append(alpha)

        flag = None

        for i in range(1, self.num_iterations):
            beta = torch.norm(w)
            v = w / beta

            w = self.hvp(self.loss, params, v)
            alpha = torch.dot(v, w)
            w = w - alpha * v - beta * self.q_vectors[i - 1]

            # after accessing self.q_vectors[i - 1], we add current itrerations v to self.q_vectors
            # here we collect alpha_i and beta_i
            self.q_vectors.append(v)
            self.alphas.append(alpha.item())
            self.betas.append(beta.item())

            if beta < self.tol:
                print('------------------------------------------------------------')
                print('This executes!!!')
                print('------------------------------------------------------------')
                flag = 1
                break

        # Construct the tridiagonal matrix T
        T = torch.diag(torch.tensor(self.alphas))
        for i in range(len(self.betas)):
            # above the diagonal
            T[i, i + 1] = self.betas[i]

            # below the diagonal
            T[i + 1, i] = self.betas[i]

        if flag:
            import ipdb
            ipdb.set_trace()

        # Compute eigenvalues of T
        eigenvalues, eigenvectors = torch.linalg.eigh(T)

        # TODO: get eigenvalues according to preconditioner P^(-1)H
        # TODO: get ritz vectors? are they a thing with Lanczos's algorithm?
        # TODO: angle between grad, ritz vectors, update vector
        # TODO: get the density plot for eigenvalues

        return eigenvalues


class ArnoldiIteration:
    def __init__(self, n_iterations=10, tol=1e-6):
        self.n_iterations = n_iterations
        self.tol = tol

        # cache model forward and grad
        self.loss = None
        self.outputs = None
        self.grad_params = None

    def hvp(self, loss, params, v):
        """Compute Hessian-vector product"""
        # cache first order grad computation
        if self.grad_params is None:
            grad_params = torch.autograd.grad(loss, params, create_graph=True)
            self.grad_params = torch.cat([g.flatten() for g in grad_params])

        grad_v_prod = torch.sum(self.grad_params * v)
        hvp = torch.autograd.grad(grad_v_prod, params, retain_graph=True)

        return torch.cat([g.flatten() for g in hvp])

    def __call__(self, model, loss_fn, batch_data, batch_labels):
        """
            because of the Arnoldi iteration, we run only a single iteration of stochastic Lanczos
        """
        # Get model parameters
        params = list(model.parameters())
        n_params = sum(p.numel() for p in params)

        # cache forward pass and loss computation
        self.outputs = model(batch_data)
        self.loss = loss_fn(self.outputs, batch_labels)

        # Initialize first basis vector randomly
        q1 = torch.randn(n_params, device=params[0].device)
        q1 = q1 / torch.norm(q1)

        # Initialize matrices for Arnoldi iteration
        Q = torch.zeros(n_params, self.n_iterations + 1, device=params[0].device)
        H = torch.zeros(self.n_iterations + 1, self.n_iterations, device=params[0].device)
        Q[:, 0] = q1

        # Run Arnoldi iteration
        early_stop = False
        for k in range(self.n_iterations):
            # Compute Hessian-vector product
            w = self.hvp(self.loss, params, Q[:, k])

            # Arnoldi iteration step
            for j in range(k + 1):
                H[j, k] = torch.dot(Q[:, j], w)
                w = w - H[j, k] * Q[:, j]

            H[k + 1, k] = torch.norm(w)
            if H[k + 1, k] < self.tol:
                H = H[:k + 1, :k + 1]
                Q = Q[:, :k + 1]

                early_stop = True
                break

            Q[:, k + 1] = w / H[k + 1, k]

        # Compute eigenvalues and eigenvectors of H
        # Convert Ritz vectors back to original space
        if early_stop:
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            # ritz_vectors = torch.mm(Q, eigenvectors)
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(H[:-1, :])
            # ritz_vectors = torch.mm(Q[:, :-1], eigenvectors)

        return eigenvalues  # , ritz_vectors


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
        # del self._param_tensors
        # del self._grad_tensors

        # self._param_tensors = []
        # self._grad_tensors = []

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                # self._param_tensors.extend([p])
                # self._grad_tensors.extend([p.grad])

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

    # def post_step_metrics(self):
    #     _param_tensors = nn.utils.parameters_to_vector(self._param_tensors)
    #     _grad_tensors = nn.utils.parameters_to_vector(self._grad_tensors)

    #     eigenvalues = arnoldi_iteration_hvp(_param_tensors, _grad_tensors, k=50).real

    #     return eigenvalues


# def hessian_vector_product(func, inputs, v):
#     return hvp(func, inputs, v, create_graph=True)


# def arnoldi_iteration_hvp(loss_fn, inputs, params, k):
#     """
#     Perform the Arnoldi iteration to approximate eigenvalues using Hessian-vector products.
    
#     Parameters:
#     - params: NN parameters.
#     - grad: first order gradient of model parameters from backprop.
#     - k: Integer, the number of Arnoldi iterations to perform.
    
#     Returns:
#     - eigenvalues: torch.Tensor of approximated eigenvalues.
#     """
#     n = params.numel()
#     Q = torch.zeros((n, k + 1)).to(params.device)
#     H = torch.zeros((k + 1, k)).to(params.device)

#     # embedding layer will have NaN grads
#     grad = torch.nan_to_num(grad, nan=0.)
    
#     # Start with a random unit vector
#     q = torch.randn(n).to(params.device)
#     q /= torch.norm(q)
#     Q[:, 0] = q
    
#     for j in range(k):
#         # Compute Hessian-vector product
#         # v = torch.autograd.grad(grad, params, grad_outputs=Q[:, j], retain_graph=True)
#         v = hessian_vector_product(loss_fn, inputs, Q[:, j])
#         v = torch.nan_to_num(v, nan=0.)
#         # v = hessian_vector_product(params, grad, Q[:, j])

#         # Modified Gram-Schmidt orthogonalization
#         for i in range(j + 1):
#             H[i, j] = torch.dot(Q[:, i], v)
#             v -= H[i, j] * Q[:, i]
        
#         H[j + 1, j] = torch.norm(v)
#         if H[j + 1, j] > 1e-10:
#             Q[:, j + 1] = v / H[j + 1, j]
#         else:
#             # Convergence achieved
#             break
    
#     # Truncate H to the correct size
#     H_reduced = H[:j + 1, :j + 1]
#     # Compute eigenvalues of the Hessenberg matrix H
#     eigenvalues = torch.linalg.eigvals(H_reduced)

#     return eigenvalues


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
        # train_loss[step], train_acc[step] = compute_losses(network, [loss_fn, acc_fn], train_dataset,
        #                                                    physical_batch_size)
        # test_loss[step], test_acc[step] = compute_losses(network, [loss_fn, acc_fn], test_dataset, physical_batch_size)

        # # at step = 0, Adam optimizer has no state, so don't record eigs then        
        # if step > 0 and eig_freq != -1 and step % eig_freq == 0:
        #     nu = get_adam_nu(optimizer)
        #     P = (1 - beta1**step) * ((nu / (1 - beta2**step)).sqrt() + epsilon)
        #     eigs[step // eig_freq, :] = get_hessian_eigenvalues(network, loss_fn, abridged_train, neigs=neigs,
        #                                                         physical_batch_size=physical_batch_size, P=P)
        #     print("eigenvalues: ", eigs[step//eig_freq, :])

        # if iterate_freq != -1 and step % iterate_freq == 0:
        #     iterates[step // iterate_freq, :] = projectors.mv(parameters_to_vector(network.parameters()).cpu().detach())

        # if save_freq != -1 and step % save_freq == 0:
        #     save_files(directory, [("eigs", eigs[:step // eig_freq]), ("iterates", iterates[:step // iterate_freq]),
        #                            ("train_loss", train_loss[:step]), ("test_loss", test_loss[:step]),
        #                            ("train_acc", train_acc[:step]), ("test_acc", test_acc[:step])])

        # print(f"{step}\t{train_loss[step]:.3f}\t{train_acc[step]:.3f}\t{test_loss[step]:.3f}\t{test_acc[step]:.3f}")

        # if (loss_goal != None and train_loss[step] < loss_goal) or (acc_goal != None and train_acc[step] > acc_goal):
        #     break

        # optimizer

        optimizer.zero_grad()

        for (X, y) in iterate_dataset(train_dataset, physical_batch_size):
            arnoldi = ArnoldiIteration(n_iterations=100, tol=1e-6)
            a_eigenvals = arnoldi(network, loss_fn, X, y)

            lanczos = LanczosAlgorithm(num_iterations=50, tol=1e3)
            l_eigenvals = lanczos(network, loss_fn, X, y)

            a_eigenvals = torch.sort(a_eigenvals)[0]
            l_eigenvals = torch.sort(l_eigenvals)[0]

            # try:
            #     assert abs(a_eigenvals[-1].item() - l_eigenvals[-1].item()) < 1
            #     assert abs(a_eigenvals[0].item() - l_eigenvals[0].item()) < 1
            # except AssertionError:
            #     print('------------------------------------------------------------')
            #     print(f"Arnoldi eigenvalue: {a_eigenvals[-1].item()}, {a_eigenvals[0].item()}")
            #     print(f"Lanczos eigenvalue: {l_eigenvals[-1].item()}, {l_eigenvals[0].item()}")
            #     print('------------------------------------------------------------')

            # import ipdb; ipdb.set_trace()

            # hessian_comp = hessian(network, loss_fn, data=(X, y), cuda=True)
            # top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(maxIter=50, tol=1e-6, top_n=10)

            # try:
            #     assert abs(torch.sort(eigenvalues)[0][-1].item() - top_eigenvalues[0]) < 10
            # except AssertionError:
            #     print('------------------------------------------------------------')
            #     print(f"Arnoldi eigenvalue: {torch.sort(eigenvalues)[0][-1].item()}")
            #     print(f"Power iteration eigenvalue: {top_eigenvalues[0]}")
            #     print('------------------------------------------------------------')

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
