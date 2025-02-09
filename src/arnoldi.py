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


class LanczosAlgorithm:
    def __init__(self, num_iterations=20, tol=1e-6):
        self.num_iterations = num_iterations
        self.tol = tol

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

        # variables
        q_vectors = []
        alpha_list = []
        beta_list = []

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
        q_vectors.append(v)
        alpha_list.append(alpha.item())

        for i in range(1, self.num_iterations):
            beta = torch.norm(w)
            v = w / beta

            w = self.hvp(self.loss, params, v)
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

        # TODO: get eigenvalues according to preconditioner P^(-1)H
        # TODO: angle between grad, ritz vectors, update vector

        return eigen_list, weight_list