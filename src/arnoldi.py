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


class ArnoldiIterationDensitySpectrum:
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

        ### temp soln for eigen density stuff
        from copy import deepcopy
        model2 = deepcopy(model)
        model2.train()

        output2 = model2(batch_data)
        loss2 = loss_fn(output2, batch_labels)
        loss2.backward(create_graph=True)

        ### generate eigen density plot
        self.params, self.gradsH = get_params_grad(model2)
        _v = [torch.randint_like(p, high=2, device=params[-1].device) for p in self.params]
        # generate Rademacher random variables
        for v_i in _v:
            v_i[v_i == 0] = -1
        _v = normalization(_v)

        # standard lanczos algorithm initlization
        _v_list = [_v]
        _w_list = []
        _alpha_list = []
        _beta_list = []

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

            ### code for eigen density part
            if k == 0:
                _w_prime = hessian_vector_product(self.gradsH, self.params, _v)
                _alpha = group_product(_w_prime, _v)
                _alpha_list.append(_alpha.cpu().item())
                _w = group_add(_w_prime, _v, alpha=-_alpha)
                _w_list.append(_w)
            else:
                _beta = torch.sqrt(group_product(_w, _w))
                _beta_list.append(_beta.cpu().item())
                if _beta_list[-1] != 0.:
                    # We should re-orth it
                    _v = orthnormal(_w, _v_list)
                    _v_list.append(_v)
                else:
                    # generate a new vector
                    _w = [torch.randn(p.size()).to(device) for p in self.params]
                    _v = orthnormal(_w, _v_list)
                    _v_list.append(_v)

                _w_prime = hessian_vector_product(self.gradsH, self.params, _v)
                _alpha = group_product(_w_prime, _v)
                _alpha_list.append(_alpha.cpu().item())

                _w_tmp = group_add(_w_prime, _v, alpha=-_alpha)
                _w = group_add(_w_tmp, _v_list[-2], alpha=-_beta)

        ### code for eigen density part
        _T = torch.zeros(self.n_iterations, self.n_iterations).to(params[-1].device)
        for i in range(len(_alpha_list)):
            _T[i, i] = _alpha_list[i]
            if i < len(_alpha_list) - 1:
                _T[i + 1, i] = _beta_list[i]
                _T[i, i + 1] = _beta_list[i]
        _eigenvalues, _eigenvectors = torch.linalg.eig(_T)

        _eigen_list = list(_eigenvalues.real.cpu().numpy())
        _weight_list = list(torch.pow(_eigenvectors[0,:], 2).cpu().numpy())

        # Compute eigenvalues and eigenvectors of H
        # Convert Ritz vectors back to original space
        if early_stop:
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            # ritz_vectors = torch.mm(Q, eigenvectors)
        else:
            eigenvalues, eigenvectors = torch.linalg.eigh(H[:-1, :])
            # ritz_vectors = torch.mm(Q[:, :-1], eigenvectors)

        return eigenvalues  # , ritz_vectors