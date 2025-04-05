# Fundamental Concepts of Bayesian Inference

## Bayes' Theorem

Bayes' theorem is the foundation of Bayesian inference and is expressed as:

$$P(\theta|x) = \frac{P(x|\theta)P(\theta)}{P(x)}$$

Where:
- $P(\theta|x)$ is the posterior probability of parameter $\theta$ given the data $x$ (what we want to estimate)
- $P(x|\theta)$ is the likelihood function indicating the probability of observing $x$ given $\theta$
- $P(\theta)$ reflects the prior belief about our parameter $\theta$
- $P(x)$ is the evidence, which in the continuous version is expressed as $\int_\theta P(x,\theta)d\theta$

## Difference Between Frequentist and Bayesian Statistics

Bayesian statistics allows incorporating prior knowledge through probability distributions over parameters of interest. This contrasts with the frequentist school, which assumes that parameters are fixed and unknown, without associated probability distributions.

In frequentist statistics, a 90% confidence interval means that if we take 100 independent samples and calculate the interval for each one, approximately 90 will contain the true parameter value.

In Bayesian statistics, we can make direct statements about the probability that a parameter is within a specific interval, which is more intuitive for many applications.

## Analytical Approaches for Bayesian Inference

When the prior distribution and likelihood have mathematical forms that allow a closed-form solution for the posterior distribution, we talk about conjugate distributions. In these cases, we can solve the Bayesian inference problem analytically.

A classic example is the Gamma-Poisson pair:
- If the likelihood follows a Poisson distribution with parameter $\lambda$
- And the prior distribution for $\lambda$ is a Gamma with parameters $\alpha$ and $\beta$
- Then the posterior distribution will also be a Gamma with parameters $\alpha + \sum X_i$ and $\beta + n$

## MCMC Methods and Their Importance

When it is not possible to obtain an analytical solution for the posterior distribution (which occurs in most practical cases), numerical methods such as Markov Chain Monte Carlo (MCMC) are used.

MCMC allows approximating the posterior distribution by generating samples from it, even when the integral in the denominator of Bayes' theorem is analytically intractable.

The Metropolis-Hastings algorithm is one of the best-known MCMC methods and works as follows:
1. Start with a random initial value of the parameter
2. Propose a new value according to a proposal distribution
3. Calculate the acceptance probability based on the likelihood and prior distribution
4. Accept or reject the new value according to this probability
5. Repeat the process many times

## PyMC and Its Capabilities

PyMC is a Python library for Bayesian inference that implements various MCMC methods and other inference techniques. It allows:
- Defining Bayesian models intuitively
- Using different MCMC algorithms (Metropolis-Hastings, Hamiltonian Monte Carlo, No-U-Turn Sampler)
- Diagnosing chain convergence
- Visualizing and analyzing results

PyMC greatly simplifies the implementation of complex Bayesian models, allowing users to focus on model definition rather than the algorithmic details of MCMC.
