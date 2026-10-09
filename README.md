# Neural Operators
The proposed work will consist, first, of a rigorous exposition of the mathematical
foundations of neural operators, including their functional formulation, approximation
properties, and their relationship with classical numerical methods for differential
equations. Subsequently, a chapter will be dedicated specifically to Fourier Neural
Operators, detailing their algorithmic structure, interpretation, and the theoretical
results justifying their applicability. Finally, we will use this theoretical framework to train a model to predict the solutions to the 2D Navier-Stokes equations for an incompressible fluid.

# Introducction to Neural Operators
Neural Operators constitute a mathematical framework oriented toward learning map
pings between function spaces, with the objective of approximating operators that act
upon solutions to differential equations. Unlike traditional neural models, which learn
finite-dimensional functions, neural operators are formulated as universal approximators
of nonlinear operators between function spaces.

Classical neural networks aim to approximate functions:

$$x \to f(x)$$

between finite-dimensional Euclidean spaces, such as $\mathbb{R}^d$. More specifically, they take a vector $x$ as input and return another vector $f(x)$ as output.

In contrast, we want to construct mappings:

$$f \to g$$

between infinite-dimensional (vector) function spaces, such as the space of continuous functions. These mappings are known as **operators**. Operators take a function $f$ as input and return another function $g$ as output.

Let $D \subset \mathbb{R}^n$ be an open and bounded subset. Consider the following family of partial differential equations:

$$\begin{cases} (L_a u)(x) = f(x), & x \in D, \\ u(x) = 0, & x \in \partial D \end{cases}$$

where $a \in A$, $u \in U$ is the solution to the differential equation, and $f \in U^*$.

Providing a solution to the equation implies finding the function $u \in U$ such that:

$$L_a(u) = f \implies (L_a)^{-1}(f) = u \equiv (L^{-1} f)(a) = u$$

With this, we can construct the operator:

$$G^\dagger := L^{-1} f : A \to U$$

$$a \mapsto u$$

We have converted the resolution of a PDE into a problem of learning its solution operator:

$$G^\dagger : (A, \mu) \to U$$

$$a \mapsto u$$

Given a probability distribution $\mu$ over $A$, we assume we are given the observations $\{a^{(i)}, u^{(i)}\}_{i=1}^N$, where $a^{(i)} \in A$ are i.i.d. samples drawn from $\mu$, and $u^{(i)} = G^\dagger(a^{(i)})$.

We aim to construct an approximation of $G^\dagger$ via a parametric family of operators:

$$G_\theta : A \to U, \quad \theta \in \mathbb{R}^p$$

by choosing $\theta^\dagger \in \mathbb{R}^p$ such that $G_{\theta^\dagger} \approx G^\dagger$.

# Building Neural Operators

We will not always have access to the functions themselves, but rather to a discretization of them.

Assume we are given $n$ values $a_i = a(x_i)$ of a function $a$ at points $\{x_i\}_{i=1}^n$. With these data, we want to design a model that maps the lists:

$$(a(x_1), a(x_2), \dots, a(x_n)) \to (u(y_1), u(y_2), \dots, u(y_m))$$

where $x_i$ are the grid points and $y_j$ are the query points for $u$.

To learn this mapping, we could use an MLP such that:

$$u = f(a) = f_L \circ f_{L-1} \circ \dots \circ f_1(a)$$

where:

$$y^{[\ell]} = f_\ell(y^{[\ell-1]}) = \sigma^{[\ell]}(K^{[\ell-1]} y^{[\ell-1]} + b^{[\ell-1]})$$

Or, in the case of a single layer, the $j$-th component is:

$$u_j := (Ka + b)_j = \sum_{i=1}^n K_{ji} a_i + b_j$$

However, this would still be a mapping between finite-dimensional spaces $\mathbb{R}^n \to \mathbb{R}^m$.

However, if we assume that $K_{ji}$ and $b_j$ are evaluations of the functions $\kappa$ and $b$ at the input points $x_i$ and query points $y_j$, the result is a mapping from the input function $a$ to an output function $u$.

In this way:

$$u_j = \sum_{i=1}^n K_{ji} a_i + b_j$$

$$\downarrow$$

becomes:

$$u(y_j) = \sum_{i=1}^n \kappa(x_i, y_j) a(x_i) \Delta_i + b(y_j)$$

## Transformation into an Integral Operator

For a sufficiently fine point grid ($n \to \infty$):

$$u(y_j) := \sum_{i=1}^n \kappa(x_i, y_j) a(x_i) \Delta_i + b(y_j) \approx \int_D \kappa(x, y_j) a(x) \, dx + b(y_j)$$

The kernel functions $\kappa$ and bias functions $b$ can be parameterized using neural networks.

In this way, we have converted the linear regression operation in finite-dimensional spaces:

$$x \to Wx + b$$

into one in infinite-dimensional spaces via the affine linear operator:

$$a \to K(a) + b$$

Conceptually, this is the analogue of a weight matrix in a standard neural network within function spaces.

With this, we can construct the analogue of a standard neural network in function spaces by successively composing these layers with activation functions.

In this way, we can approximate the operator $G^\dagger$ using the parametric family:

$$G_\theta := \sigma_T (K_{T-1} + b_{T-1}) \circ \dots \circ \sigma_1 (K_0 + b_0)$$

This architecture is known as neural operators.

# Fourier Neural Operators

Among the most prominent architectures are Fourier Neural
Operators (FNO), which perform the operator’s primary updates in the frequency do
main via discrete Fourier transforms. This approach provides an efficient approximation
capable of generalizing across spatial meshes different from those utilized during training.

Although this architecture can be very efficient, it still has a limitation: the input functions $a : D \to \mathbb{R}^m$ are defined on a spatial domain $D \subset \mathbb{R}^n$. This makes the model biased toward the specific training domain.

We can solve this by applying the Fourier transform $\mathcal{F}$ to the input function. The model is now trained in the frequency domain. We assume that $D = \mathbb{T}^d$ is the unit torus and that all functions are complex-valued.

We start from the integral operator $K_\theta$:

$$(K_\theta v)(x) = \int_D \kappa_\theta(x, y) v(y) \, dy$$

In particular, when $\kappa_\theta(x, y) = \kappa_\theta(x - y)$, the operator $K_\theta$ is a convolution:

$$(K_\theta v)(x) = \int_D \kappa_\theta(x - y) v(y) \, dy = \kappa_\theta * v$$

Convolution satisfies:

$$f * g = \mathcal{F}^{-1}(\mathcal{F}(f) \cdot \mathcal{F}(g))$$

We can rewrite the operator as:

$$(K_\theta v)(x) = \mathcal{F}^{-1}(\mathcal{F}(\kappa_\theta) \cdot \mathcal{F}(v))(x)$$

Converting a convolution in the physical domain into a simple product in the frequency domain, where $\mathcal{F}(\kappa_\theta) = R_\theta$:

$$(K_\theta v)(x) = \mathcal{F}^{-1}(R_\theta \cdot \mathcal{F}(v))(x)$$

Note that if $\kappa_\theta : D \to \mathbb{C}^{d_v \times d_v} \implies \mathcal{F}(\kappa_\theta) \equiv R_\theta : \mathbb{Z}^d \to \mathbb{C}^{d_v \times d_v}$.

Instead of learning the kernel $\kappa_\theta$ in the physical domain, we learn $R_\theta$ in the frequency domain.

By operating directly in the frequency domain, it captures non-local relationships that would be complex to model through spatiotemporal coordinates.

This architecture can be represented as follows:

<img src="https://github.com/Percival2003/Neural-Operator/blob/d2b80b437a16ada7afa4e0aed91b98c178eccff4/Images/FNO.png" alt="Texto alternativo" width="500">

# Navier-Stokes equation solutions
Our objective is to critically analyze how neural operators—and
Fourier operators in particular—can be employed to approximate the solutions of these
models, conceptually comparing this approach with traditional numerical methods. To this end, we use the 2D incompressible Navier-Stokes equations as an example. 

$$
\begin{aligned}
\frac{\partial u}{\partial t}(x,t) + u(x,t) \cdot \nabla u(x,t) &= -\nabla p(x,t) + \nu \nabla^2 u(x,t) + f(x) \\
\nabla \cdot u(x,t) &= 0 \\
u(x,0) &= u_0(x)
\end{aligned}
$$

where $x \in \mathbb{T}^2$ and $t \in (0, \infty)$.

We train a model capable of predicting the fluid velocity field as the output (the solution to the equation).

<img src="https://github.com/Percival2003/Neural-Operator/blob/7b7c5a68ffc53f28a1615c3ad97ce26256e4de73/Images/Evoluci%C3%B3n%20error%20relativo%20L2.png" alt="Texto alternativo" width="500">
<img src="https://github.com/Percival2003/Neural-Operator/blob/7b7c5a68ffc53f28a1615c3ad97ce26256e4de73/Images/Evoluci%C3%B3n%20error%20cuadr%C3%A1tico%20medio.png" alt="Texto alternativo" width="500">
<img src="https://github.com/Percival2003/Neural-Operator/blob/7b7c5a68ffc53f28a1615c3ad97ce26256e4de73/Images/Predicci%C3%B3n%20de%20un%20frame.png" alt="Texto alternativo" width="700">
