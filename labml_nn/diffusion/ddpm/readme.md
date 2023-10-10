# [Denoising Diffusion Probabilistic Models (DDPM)](https://nn.labml.ai/diffusion/ddpm/index.html)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/experiment.ipynb)

This is a [PyTorch](https://pytorch.org) implementation/tutorial of the paper
[Denoising Diffusion Probabilistic Models](https://papers.labml.ai/paper/2006.11239).

In simple terms, we get an image from data and add noise step by step.
Then We train a model to predict that noise at each step and use the model to
generate images.

Here is the [UNet model](https://nn.labml.ai/diffusion/ddpm/unet.html) that predicts the noise and
[training code](https://nn.labml.ai/diffusion/ddpm/experiment.html).
[This file](https://nn.labml.ai/diffusion/ddpm/evaluate.html) can generate samples and interpolations
from a trained model.

---
title: Denoising Diffusion Probabilistic Models (DDPM)
summary: >
  PyTorch implementation and tutorial of the paper
  Denoising Diffusion Probabilistic Models (DDPM).
---

# Denoising Diffusion Probabilistic Models (DDPM)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/experiment.ipynb)

This is a [PyTorch](https://pytorch.org) implementation/tutorial of the paper
[Denoising Diffusion Probabilistic Models](https://papers.labml.ai/paper/2006.11239).

In simple terms, we get an image from data and add noise step by step.
Then We train a model to predict that noise at each step and use the model to
generate images.

The following definitions and derivations show how this works.
For details please refer to [the paper](https://papers.labml.ai/paper/2006.11239).

## Forward Process

The forward process adds noise to the data $x_0 \sim q(x_0)$, for $T$ timesteps.

\begin{align}
q(x_t | x_{t-1}) = \mathcal{N}\big(x_t; \sqrt{1-  \beta_t} x_{t-1}, \beta_t \mathbf{I}\big) \\
q(x_{1:T} | x_0) = \prod_{t = 1}^{T} q(x_t | x_{t-1})
\end{align}

where $\beta_1, \dots, \beta_T$ is the variance schedule.

We can sample $x_t$ at any timestep $t$ with,

\begin{align}
q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
\end{align}

where $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$

## Reverse Process

The reverse process removes noise starting at $p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
for $T$ time steps.

\begin{align}
\textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
 \textcolor{lightgreen}{\mu_\theta}(x_t, t), \textcolor{lightgreen}{\Sigma_\theta}(x_t, t)\big) \\
\textcolor{lightgreen}{p_\theta}(x_{0:T}) &= \textcolor{lightgreen}{p_\theta}(x_T) \prod_{t = 1}^{T} \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) \\
\textcolor{lightgreen}{p_\theta}(x_0) &= \int \textcolor{lightgreen}{p_\theta}(x_{0:T}) dx_{1:T}
\end{align}

$\textcolor{lightgreen}\theta$ are the parameters we train.

## Loss

We optimize the ELBO (from Jenson's inequality) on the negative log likelihood.

\begin{align}
\mathbb{E}[-\log \textcolor{lightgreen}{p_\theta}(x_0)]
 &\le \mathbb{E}_q [ -\log \frac{\textcolor{lightgreen}{p_\theta}(x_{0:T})}{q(x_{1:T}|x_0)} ] \\
 &=L
\end{align}

The loss can be rewritten as  follows.

\begin{align}
L
 &= \mathbb{E}_q [ -\log \frac{\textcolor{lightgreen}{p_\theta}(x_{0:T})}{q(x_{1:T}|x_0)} ] \\
 &= \mathbb{E}_q [ -\log p(x_T) - \sum_{t=1}^T \log \frac{\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)}{q(x_t|x_{t-1})} ] \\
 &= \mathbb{E}_q [
  -\log \frac{p(x_T)}{q(x_T|x_0)}
  -\sum_{t=2}^T \log \frac{\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}
  -\log \textcolor{lightgreen}{p_\theta}(x_0|x_1)] \\
 &= \mathbb{E}_q [
   D_{KL}(q(x_T|x_0) \Vert p(x_T))
  +\sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0) \Vert \textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t))
  -\log \textcolor{lightgreen}{p_\theta}(x_0|x_1)]
\end{align}

$D_{KL}(q(x_T|x_0) \Vert p(x_T))$ is constant since we keep $\beta_1, \dots, \beta_T$ constant.

### Computing $L_{t-1} = D_{KL}(q(x_{t-1}|x_t,x_0) \Vert \textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t))$

The forward process posterior conditioned by $x_0$ is,

\begin{align}
q(x_{t-1}|x_t, x_0) &= \mathcal{N} \Big(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t \mathbf{I} \Big) \\
\tilde\mu_t(x_t, x_0) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
                         + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\
\tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t
\end{align}

The paper sets $\textcolor{lightgreen}{\Sigma_\theta}(x_t, t) = \sigma_t^2 \mathbf{I}$ where $\sigma_t^2$ is set to constants
$\beta_t$ or $\tilde\beta_t$.

Then,
$$\textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big)$$

For given noise $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ using $q(x_t|x_0)$

\begin{align}
x_t(x_0, \epsilon) &= \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon \\
x_0 &= \frac{1}{\sqrt{\bar\alpha_t}} \Big(x_t(x_0, \epsilon) -  \sqrt{1-\bar\alpha_t}\epsilon\Big)
\end{align}

This gives,

\begin{align}
L_{t-1}
 &= D_{KL}(q(x_{t-1}|x_t,x_0) \Vert \textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)) \\
 &= \mathbb{E}_q \Bigg[ \frac{1}{2\sigma_t^2}
 \Big \Vert \tilde\mu(x_t, x_0) - \textcolor{lightgreen}{\mu_\theta}(x_t, t) \Big \Vert^2 \Bigg] \\
 &= \mathbb{E}_{x_0, \epsilon} \Bigg[ \frac{1}{2\sigma_t^2}
  \bigg\Vert \frac{1}{\sqrt{\alpha_t}} \Big(
  x_t(x_0, \epsilon) - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}} \epsilon
  \Big) - \textcolor{lightgreen}{\mu_\theta}(x_t(x_0, \epsilon), t) \bigg\Vert^2 \Bigg] \\
\end{align}

Re-parameterizing with a model to predict noise

\begin{align}
\textcolor{lightgreen}{\mu_\theta}(x_t, t) &= \tilde\mu \bigg(x_t,
  \frac{1}{\sqrt{\bar\alpha_t}} \Big(x_t -
   \sqrt{1-\bar\alpha_t}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big) \bigg) \\
  &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
  \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
\end{align}

where $\epsilon_\theta$ is a learned function that predicts $\epsilon$ given $(x_t, t)$.

This gives,

\begin{align}
L_{t-1}
&= \mathbb{E}_{x_0, \epsilon} \Bigg[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar\alpha_t)}
  \Big\Vert
  \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
  \Big\Vert^2 \Bigg]
\end{align}

That is, we are training to predict the noise.

### Simplified loss

$$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
\epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
\bigg\Vert^2 \Bigg]$$

This minimizes $-\log \textcolor{lightgreen}{p_\theta}(x_0|x_1)$ when $t=1$ and $L_{t-1}$ for $t\gt1$ discarding the
weighting in $L_{t-1}$. Discarding the weights $\frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1 - \bar\alpha_t)}$
increase the weight given to higher $t$ (which have higher noise levels), therefore increasing the sample quality.

This file implements the loss calculation and a basic sampling method that we use to generate images during
training.

Here is the [UNet model](unet.html) that gives $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$ and
[training code](experiment.html).
[This file](evaluate.html) can generate samples and interpolations from a trained model.
