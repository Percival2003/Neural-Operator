# Neural-Operator
This repository presents the code developed for my Bachelor's Thesis on neural operators, specifically focusing on predicting 2D fluid velocity using the Navier-Stokes equations

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

---

*Entendido. A partir de ahora, traduciré todo el texto que me envíes al inglés con formato Markdown listo para un README. Quedo a la espera del siguiente bloque.*

# Fourier Neural Operators

Among the most prominent architectures are Fourier Neural
Operators (FNO), which perform the operator’s primary updates in the frequency do
main via discrete Fourier transforms. This approach provides an efficient approximation
capable of generalizing across spatial meshes different from those utilized during training.
The proposed work will consist, first, of a rigorous exposition of the mathematical
foundations of neural operators, including their functional formulation, approximation
properties, and their relationship with classical numerical methods for differential
equations. Subsequently, a chapter will be dedicated specifically to Fourier Neural
Operators, detailing their algorithmic structure, interpretation, and the theoretical
results justifying their applicability.
Next, the study will address several classical problems in mathematical physics formu
lated as differential equations, such as the Navier–Stokes equations, Burger’s equation,
and other linear or nonlinear systems. For each, the corresponding formulation will be
presented, alongside its framing as an operator problem amenable to approximation
via neural operators. The objective is to critically analyze how neural operators—and
Fourier operators in particular—can be employed to approximate the solutions of these
models, conceptually comparing this approach with traditional numerical methods.
