# Discriminative Classification Models in Noise-Free VS Noisy Scenarios


![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![Sklearn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)

In this project I present a randomized classifiation problem using self-generated synthetic data to contrast several discriminative models in presence and absense of noise. The true label distribution is created using hard tresholding over a mixture of randomly generated simple geometries restricted to a fixed space $S$. Uniform samplig over $S$ yields the noise-free observations and the noisy ones are obtained by pertubing the features with Gaussian noise or the labels with Bernoulli noise. Multiple visualizations and classification metrics are included to gain insights of the impact of noise when training such models, particularly in the shape of the models class probability density funtion.

<img src="assets/density.png" width=800>

## Classification

In Machine Learing, **classification** is the task for which a model learns how to identify the members of distinct *groups* or *classes* $\mathcal{C}$ within the same space $X$. There are two main strategies to model this problem, via a **discriminative** or a **generative** approach. In general, we may use some reference space $Y$ to represent the classes $\mathcal{C}$ in a more convinient way.

### Discriminative Classification

A non-probabilistic model that learns a *hard rule* $f:X\rightarrow Y$ to directly identify the class from which an arbitrary observation belongs to, is called a **non-probabilistic discriminative classifcation model**. 

A probabilistic model that learns the conditional distribution $y|x\sim p(y|x)$ where $y$ is a random variable with support in $Y$ that represents the label distribution from a point $x\in X$ is called a **probabilistic discriminative classification model**.

Any non-probabilistic discriminative classification model can be turned into a probabilistic one via any function $`\phi:Y\rightarrow\Delta_{\lvert\mathcal{C}\rvert-1}`$ used to parametrize a $\mathrm{Categorical}$ distribution, where $\Delta_n$ is the $n$-standard simplex.

$$y|x\sim\mathrm{Categorical}(y|(\phi\circ f)(x))$$

***Note:** Any categorical distribution parametrized by some $\pi\in\Delta_n$ can be expressed as a categorical distribution with $n-1$ parameters as follows*
```math
\mathrm{Categorical}(x|\pi) = \mathrm{Categorical}\left(x\middle\vert\left(\pi_1,\dots,\pi_{n-k-1},1-\sum_{j\neq k}\pi_j,\pi_{k+1},\dots,\pi_n\right)\right)
```

*Therefore, a categorical distribution of two classes is equivalent to a Bernoulli distribution.*

### Generative Classification

A probabilistic model that learns the conditional distribution $x|y\sim p(x|y)$ where $x$ is a random variable with support in $X$ that represents the point distribution from a label $y\in Y$ and a prior distribution on the labels $y\sim p(y)$ is called a **generative classification model**, as it models the data distribution itself and can determine the label distribution $`y|x\sim p(y|x)`$ via the Bayes theorem

```math
p(y|x) = \frac{p(y)p(x|y)}{\sum_{y'\in Y} p(y')p(x|y')}
```

## Binary Classification Models
The simplest classification problem is when $`\lvert\mathcal{C}\rvert=2`$ and is called **binary classification**. This project solves a randomized problem of this type with several probabilistic & non-probabilistic discriminative models.

Define the **logistic function** $`\sigma:\mathbb{R}\rightarrow (0,1)`$ as
```math
\sigma(x) = \frac{1}{1+e^{-x}}
```

In this section I briefly present the models that are used in the project under the binary classification context.

### Logistic Regression

This is a probabilistic discriminative linear model for classification. It is prefered by its simplicity and high interpretability. Its parameters are $\theta=(w,b)\in\mathbb{R}^{n+1}$ and is defined as

```math
y\vert x\sim \mathrm{Bernoulli}\left( y\vert\sigma(w^Tx+b) \right)
```

### Support Vector Machine (SVM)
This is a non-probabilistic discriminative model for classification. It relies on the usage of [Mercer Kernels](https://en.wikipedia.org/wiki/Mercer%27s_theorem) to apply non-linearities to the feature space and is trained using [quadratic programming](https://en.wikipedia.org/wiki/Quadratic_programming) algorithms to solve a soft margin hyperplane separation problem. Its parameters (after training) are $`\theta=(w,b)\in\mathbb{R}^{\lvert SV\rvert+1}`$ and is defined as

```math
f(x) = \sum_{i\in SV} w_iy_iK(x_i,x) + b
```

where $SV$ is a subset of the training points $`\{x_i\}_{i=1}^N`$

### Random Forest

This is a non-probabilistic discriminative model for classification. It is defined as an esemble of classification trees trained on randomized subsets of features and training data. Generally, a *comitee method* is used (majoirity vote) to determine the output of the ensemble.

A classification tree is one of the most interpretable models in Machine Learning, defined by a set of decission rules strucutred in the nodes of a tree, that split the feature space into several regions (one per class). If such tree has $J$ nodes, let $R_1,\dots,R_J$ represent the region associated to each one of these, then
```math
f(x) = \sum_{j=1}^Jc_j\mathbb{I}(x\in R_j)
```


### Multi-layer Perceptron (MLP)

This is a probabilistic discriminative model for classification. It is a classic feedforward neural network represented by the composition of multiple afine + non-linear transformations and a final logistic function. Formally, let $`n_1,\dots,n_L\in\mathbb{Z}_+`$ such that $n_L=1$. $`\forall\ k=1,\dots,L`$ let $`\phi_k:\mathbb{R}^{n_{k-1}}\rightarrow\mathbb{R}^{n_k}`$ be an affine function and $`\psi_k:\mathbb{R}^{n_k}\rightarrow\mathbb{R}^{n_k}`$ be a vectorized [activation function](https://en.wikipedia.org/wiki/Activation_function) such that

```math
y|x \sim \mathrm{Bernoulli}\left((\sigma\circ\psi_L\circ\phi_L\circ\cdots\circ\psi_1\circ\phi_1)(x)\right)
```


## Scikit-learn

The former models are implemented in [Scikit-learn](https://scikit-learn.org/stable/), which is an open-source machine learning library for Python that provides simple and efficient tools for data analysis and modeling. It is built on top of [NumPy](https://numpy.org/), [SciPy](https://scipy.org/es/), and [Matplotlib](https://matplotlib.org/), and offers a wide range of algorithms for classification, regression, clustering, dimensionality reduction, and model selection with a consistent API (many custom implementations tend to use a scikit-learn based structure and names).

## Noise

In the real world, there are multiple factors that introduce uncertainty and imperfections into data, commonly referred to as **noise**. Noise can distort the true underlying patterns that a machine learning model is trying to learn, making training and generalization more difficult. Broadly, there are two types of noise: **feature noise** and **label noise**.

### Feature noise
Feature noise affects the input variables and can arise from measurement errors, sensor inaccuracies, data transmission issues, or incomplete information. For example, in regression problems, this type of noise is often modeled as part of the stochastic component (e.g., additive Gaussian noise), and the model implicitly learns to approximate the underlying signal despite these perturbations. There can be additive noise (small random variations added to values) or multiplicative/value-changing noise that distorts the original measurements.

### Label noise
Label noise, on the other hand, affects the target variable. In regression, small perturbations in the target can sometimes be treated as part of the natural variability of the system. However, in classification problems, label noise is often more problematic because it may involve incorrect class assignments. Unlike small continuous deviations, mislabeling can fundamentally mislead the decision boundary, especially in high-capacity models that may overfit to these incorrect labels. As a result, classification tasks typically require more robust strategies—such as regularization, data cleaning, or noise-tolerant loss functions—to mitigate the impact of noisy labels.
