**********************************
The maths behind the code
**********************************

.. _math_behind:

Let's talk about math!
######################


The main idea behind the model proposed in this library is to learned the latent structure
of the input data :math:`x \in \mathcal{X}`.

Variational AutoEncoder
~~~~~~~~~~~~~~~~~~~~~~~

**Model Setting**

Assume we are given a set of input data :math:`x \in \mathcal{X}`. A VAE aims at maximizing the 
likelihood of a given parametric model :math:`\{\mathbb{P}_{\theta}, \theta \in \Theta\}`. It is 
assumed that there exist latent variables :math:`z` living in a lower dimensional space 
:math:`\mathcal{Z}`, referred to as the *latent space*, such that the marginal distribution 
of the data can be written as 


.. math::

    p_{\theta}(x) = \int \limits _{\mathcal{Z}} p_{\theta}(x|z)q(z) dz \,,



where :math:`q` is a prior distribution over the latent variables acting as a *regulation factor* 
and :math:`p_{\theta}(x|z)` is most of the time taken as a simple parametrized distribution (*e.g.*
Gaussian, Bernoulli, etc.) and is referred to as the *decoder* the parameters of which are 
given by neural networks. Since the integral of teh objective is most of the time intractable,
so is the posterior distribution:

.. math::

    p_{\theta}(z|x) = \frac{p_{\theta}(x|z) q(z)}{\int \limits_{\mathcal{Z}} p_{\theta}(x|z) q(z) dz}\,.

This makes direct application of Bayesian inference impossible and so recourse to approximation
techniques such as variational inference is needed. Hence, a variational distribution 
:math:`q_{\phi}(z|x)` is introduced and aims at approximating the true posterior distribution 
:math:`p_{\theta}(z|x)`. This variational distribution is often referred to as the *encoder*. In the initial version of the VAE, :math:`q_{\phi}` is taken as a multivariate 
Gaussian whose parameters :math:`\mu_{\phi}` and :math:`\Sigma_{\phi}` are again given by neural 
networks. Importance sampling can then be applied to derive an unbiased estimate of the marginal
distribution :math:`p_{\theta}(x)` we want to maximize.

.. math::

    \hat{p}_{\theta}(x) = \frac{p_{\theta}(x|z)q(z)}{q_{\phi}(z|x)} \hspace{2mm} \text{and} \hspace{2mm} \mathbb{E}_{z \sim q_{\phi}}\big[\hat{p}_{\theta}\big] = p_{\theta}(x)\,.

Using Jensen's inequality allows finding a lower bound on the objective function of the objective

.. math::

     \begin{aligned}
      \log p_{\theta}(x) &= \log \mathbb{E}_{z \sim q_{\phi}}\big[\hat{p}_{\theta}\big]\\
                         &\geq \mathbb{E}_{z \sim q_{\phi}}\big[\log \hat{p}_{\theta}\big]\\
                         & \geq \mathbb{E}_{z \sim q_{\phi}}\big[ \log p_{\theta}(x, z) - \log q_{\phi}(z|x) \big] = ELBO\,.
     \end{aligned}

The Evidence Lower BOund (ELBO) is now tractable since both :math:`p_{\theta}(x, z)` and 
:math:`q_{\phi}(z|x)` are known and so can be optimized with respect to the *encoder* and *decoder* parameters. 


**Bringing Geometry to the Model**

In the RHVAE, the assumption of an Euclidean latent space is relaxed and it is assumed that the 
latent variables live in the Riemannian manifold :math:`\mathcal{Z} =(\mathbb{R}^d, g)` where :math:`g` is the Riemannian metric..
This Riemannian metric is basically a smooth inner product on the tangent space 
:math:`T_{\mathcal{Z}}` of the manifold defined at each point :math:`z \in \mathcal{Z}`. Hence, it can be represented by a definite positive matrix :math:`\mathbf{G}(z)` at each point of the manifold :math:`\mathcal{Z}`. This Riemannian metric plays a crucial role in the modeling of the latent space and since it is not known we propose to **parametrize** it and **learn** it directly from the data :math:`x \in \mathcal{X}`. The metric parametrization writes:

.. math::

    \mathbf{G}^{-1}(z) = \sum_{i=1}^N L_{\psi_i} L_{\psi_i}^{\top} \exp \Big(-\frac{\lVert z -c_i \rVert_2^2}{T^2} \Big) + \lambda I_d \,,

where :math:`N` is the number of observations, :math:`L_{\psi_i}` are lower triangular matrices with positive diagonal coefficients learned from the data and parametrized with neural networks, :math:`c_i` are referred to as the *centroids* and correspond to the mean :math:`\mu_{\phi}(x_i)` of the encoded distributions of the latent variables :math:`z_i` :math:`(z_i \sim q_{\phi}(z_i|x_i) = \mathcal{N}(\mu_{\phi}(x_i), \Sigma_{\phi}(x_i))`, :math:`T` is a temperature scaling the metric close to the *centroids* and :math:`\lambda` is a regularization factor that also scales the metric tensor far from the latent codes. 



**Combining Geometrical Aspect And Normalizing Flows**

A way to improve the vanilla VAE resides in trying to enhance the ELBO expression so that it becomes closer to the true objective. Trying to tweak the approximate posterior distribution so that it becomes *closer* to the true posterior can achieve such a goal. To do so, a method involving parametrized invertible mappings :math:`f_x` called *normalizing flows* were proposed in~\cite{rezende_variational_2015} to *sample* :math:`z`. A starting random variable :math:`z_0` is drawn from an initial distribution :math:`q_{\phi}(z|x)` and then :math:`K` normalizing flows are applied to :math:`z_0` resulting in a random variable :math:`z_K = f_x^K \circ \cdots \circ f_x^1(z_0)`. Ideally, we would like to have access to normalizing flows targeting the true posterior and allowing enriching the above distribution and so improve the lower bound. In that particular respect, a model inspired by the Hamiltonian Monte Carlo sampler~\cite{neal_mcmc_2011} and relying on Hamiltonian dynamics was proposed in~\cite{salimans_markov_2015} and~\cite{caterini_hamiltonian_2018}. The strength of such a model relies in the choice of the normalizing flows which are guided by the gradient of the true posterior distribution. 

