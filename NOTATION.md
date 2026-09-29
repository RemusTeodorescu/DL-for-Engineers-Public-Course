# Course Notation

One notation for the slides, the speaker notes and the exercise notebooks of
*Deep Learning for Engineering*. Where the textbooks (UDL, GBC) write something
else, the course keeps its own symbol; the differences are listed at the end.

Checked by `tools/deck/check_notation.py` (course policy C3, `docs/COURSE_POLICIES.md`).

## Typeface

| Kind | Written | Example |
|---|---|---|
| scalar | italic lowercase | $x$, $b$, $\alpha$ |
| vector | bold lowercase | $\mathbf{x}$, $\mathbf{h}_k$, $\mathbf{b}_k$ |
| matrix | bold uppercase | $\mathbf{W}_k$, $\mathbf{A}$ |
| function | italic, argument in parentheses | $f(\mathbf{x})$, $a(z)$ |

## Data

| Symbol | Meaning |
|---|---|
| $x$, $\mathbf{x}$ | input |
| $y$, $\mathbf{y}$ | target (the measured or true value) |
| $\hat{y}$, $\hat{\mathbf{y}}$ | prediction of the network |
| $N$ | number of training samples |
| $i$ | sample index, $i = 1 \dots N$ |
| $\{(\mathbf{x}_i, y_i)\}$ | training set |
| $C$, $c$ | number of classes, class index |

## Network

| Symbol | Meaning | In PyTorch |
|---|---|---|
| $f(\mathbf{x}; \theta)$ | the network | `model(x)` |
| $\theta$ | all trainable parameters | `model.parameters()` |
| $\mathbf{w}$, $b$ | weights and bias of one neuron | |
| $\mathbf{W}_k$, $\mathbf{b}_k$ | weight matrix and bias of layer $k$ | `nn.Linear.weight`, `.bias` |
| $a(\cdot)$ | activation function | `nn.ReLU()` |
| $z$ | pre-activation (logit before softmax) | |
| $\mathbf{h}_k$ | output of layer $k$, $\mathbf{h}_0 = \mathbf{x}$ | |
| $k$, $K$ | layer index, number of hidden layers | |
| $D_i$, $D$, $D_o$ | inputs, neurons per hidden layer (width), outputs | `in_features`, `out_features` |
| $P$ | number of parameters | |

A layer is $\mathbf{h}_k = a(\mathbf{W}_k \mathbf{h}_{k-1} + \mathbf{b}_k)$.
When a second subscript is needed (node $i$, time $t$), the layer index moves
up: $\mathbf{h}_i^{(k)}$.

## Training

| Symbol | Meaning | In PyTorch |
|---|---|---|
| $\mathcal{L}(\theta)$ | loss over the training set or a batch | `loss` |
| $\ell_i$ | loss of one sample | `reduction="none"` |
| $\nabla_\theta \mathcal{L}$ | gradient of the loss | `loss.backward()` |
| $\alpha$ | learning rate | `lr` |
| $\beta$ ($\beta_1$, $\beta_2$) | momentum coefficients (Adam) | `momentum`, `betas` |
| $t$ | optimisation step, or time in a sequence | |
| $\lambda$ | regularisation weight | `weight_decay` |
| $\sigma^2$ | noise variance | |

## Architectures

| Symbol | Meaning |
|---|---|
| $k_{mn}$ | convolution kernel entry |
| $\mathbf{X}$, $\mathbf{A}$ | node features, adjacency matrix |
| $\psi$, $\phi$ | message function, update (combine) function of a graph network |
| $\mathbf{h}_t$, $\mathbf{c}_t$ | hidden state and cell state at time $t$ |
| $\mathbf{f}_t$, $\mathbf{i}_t$, $\mathbf{o}_t$ | forget, input and output gates |
| $\sigma(\cdot)$ | sigmoid function |

## Physics (Part 2)

| Symbol | Meaning |
|---|---|
| $u(\mathbf{x}, t)$ | the solution field the network approximates |
| $\Omega$, $\partial\Omega$ | the domain and its boundary |
| $\mathcal{L}_{\mathrm{PDE}}$, $\mathcal{L}_{\mathrm{BC}}$, $\mathcal{L}_{\mathrm{IC}}$ | loss terms for the equation, boundary and initial conditions |
| $\mathcal{N}$ | the network written as the solution, $u \approx \mathcal{N}$ (L7.1, Solution Ansatz) |
| $N^*$ | the least number of collocation points, set by the neurons (L7.1) |

Physical quantities (temperature, velocity, concentration, voltage) are
defined on the slide where they appear, with their SI unit.

## Accepted double meanings

Three symbols carry two meanings; the context separates them.

| Symbol | Meaning 1 | Meaning 2 |
|---|---|---|
| $\sigma$ | $\sigma(\cdot)$, the sigmoid function | $\sigma$, $\sigma^2$, a standard deviation or variance |
| $a$ | $a(\cdot)$, the activation function | $a$, an action in reinforcement learning (L6.2) |
| $k$ | layer index | $k_{mn}$, a convolution kernel entry |

## Where the books differ

| Book | Book writes | Course writes |
|---|---|---|
| UDL ch. 3 (shallow network) | $\theta$ hidden, $\phi$ output parameters | $\mathbf{W}_1, \mathbf{b}_1$ hidden, $\mathbf{W}_2, \mathbf{b}_2$ output |
| UDL ch. 4 (deep network) | $\boldsymbol{\Omega}_k$, $\boldsymbol{\beta}_k$ | $\mathbf{W}_k$, $\mathbf{b}_k$ |
| UDL ch. 7 (backpropagation toy) | $\omega_k$, $\beta_k$ | $w_k$, $b_k$ |
| GBC ch. 8 | $\epsilon$ for the learning rate | $\alpha$ |
| many texts | $\eta$ for the learning rate | $\alpha$ |
| many texts | $N_{\mathrm{in}}$, $N_{\mathrm{out}}$ | $D_i$, $D_o$ |
| Liu ch. 2.5.2 | $P^*$ | $N^*$ ($P$ is the parameter count) |
| many texts | $\mathbf{h}^{(l)}$ | $\mathbf{h}_k$, or $\mathbf{h}^{(k)}$ with a second index |
