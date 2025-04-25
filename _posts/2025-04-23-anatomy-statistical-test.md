---
layout: post
title: "Anatomy of a Statistical Test"
description: "Understanding the logic of hypothesis testing through simulation, plots, and code." 
---


> *“Once we believe in a law of repetitive life-cycles... we are sure to discover confirmation of it nearly everywhere. But such ‘facts,’ if examined closely, often turn out to be selected in the light of the very theories they are supposed to test.”*  
> — Karl Popper, *The Poverty of Historicism*

---

Statistical testing might be one of the most widely used - and most misunderstood - tools in data analysis. Every time someone runs an experiment or receives freshly collected data, they often Google the “right” test, plug in the numbers, and get a p-value. If it's below 5%, great - the job is done. If not, a flurry of slicing, dicing, and re-testing begins until that magic threshold is crossed. 

But what is this p-value really doing? What *logic* sits beneath it? In this post, we’ll break down **parametric hypothesis testing** into logical pieces, using simple code and visual intuition. 

We'll cover:
- The structure of a parametric test
- How sampling distributions help us "reverse engineer" inference
- The role of critical regions
- Visual intuition for statistical power

To get a better grip on how we make these statistical decisions - and avoid the trap Popper warns about - let's begin with a guiding example.

## Guiding Example

Let’s say you're running an A/B test to evaluate whether a new product page design helps users convert faster. In this case the outcome is the **time** it takes users to complete a desired action, like signing up or completing a purchase. For simplicity, we'll assume that individual conversion times are normally distributed. This means that the sample means for the two groups, and therefore their difference, are also normally distributed. We also assume equal variance across groups, so that the distribution of the difference in means depends only on the common variance and the sample size.

We will call A the group with the current page, and B the group with the new page.

## Parametric Testing: A Logic-First View

A very common family of tests are **parametric tests**. These are procedures that help us answer questions about **parameters** of an assumed probabilistic model. Here’s the general structure of a parametric test:

1. **Model Assumption**:  
   We assume a probabilistic model for the data, typically expressed in terms of random variables $( X_1, X_2, \dots, X_n) \sim f(x; \theta)$, where $\theta$ is a parameter of interest. This part is often overlooked, but it's crucially important. Building our decision-making process upon wrong assumptions can be very costly.
   
   In our example, we can use the following model:
   - $ X_{A, i} \sim \mathcal{N}(\mu_A, \sigma^2) $
   - $ X_{B, i} \sim \mathcal{N}(\mu_A + \theta, \sigma^2) $
   - Here, $\theta$ represents the treatment effect: the true improvement (or deterioration) in mean conversion time due to the new design.

2. **Hypothesis Specification**:  
   We split the parameter space $\Theta$ into two parts:
   - Null hypothesis: $H_0: \theta \in \Theta_0 $
   - Alternative hypothesis: $H_1: \theta \in \Theta_1$

   Notice how we are creating a mapping between hypothesis and parameters; **this is what makes the test parametric: we encapsulate all the information that we need to make a decision in this parameter**. Going back to our example, we can formulate the following hypothesis:
   - $ H_0: \theta \geq 0 $: the new design is no faster (or worse)
   - $ H_1: \theta < 0 $: the new design leads to faster conversions

3. **Statistic Choice**:  
   We select a [statistic](https://en.wikipedia.org/wiki/Test_statistic) $T(X_1, \dots, X_n)$ that summarizes the data and depends on $\theta$. This statistic often is an estimator $\hat{\theta}$ for the true value of the parameter $\theta$. In our A/B test, the statistic of interest is the difference in sample means between the two groups:
   - $\hat{\theta} = \bar{X}_B - \bar{X}_A$
   - Given our assumptions, if we take n samples for each group, $\hat{\theta}$ has sampling distribution: $\mathcal{N}(\theta, 2\sigma^2/n)$

4. **Observed Value**:  
   In the real world, we only get one sample — and thus one observed value $\hat{\theta}$. Our goal is to use this value to make a decision: does it provide enough evidence to reject $H_0$?

5. **Decision Rule (Reject Region)**:  
   We define a **critical region** $\mathcal{C}$ such that, if $\hat{\theta} \in \mathcal{C}$, we reject the null hypothesis. This region is usually determined by fixing a maximum acceptable probability of making a **Type I error** — rejecting $H_0$ when it's actually true. In our A/B test - and actually quite often in practice - a natural way of defining the critical region is through a **critical value** $c$. If $\hat{\theta} < c$, we reject the null.

---

From a philosophical perspective, this whole process is an exercise in *reverse engineering*. We observe one data point from a distribution and ask: *How likely is it that this data would have come from a world where $H_0$ is true?*

If that likelihood is too low, we reject $H_0$. Next, let’s bring this to life with some simulations.

<img src="/assets/img/testing_logic_files/testing_logic_diagram.jpg" alt="Drawing" style="width: 1000px;"/>


## Simulating multiple worlds

> *"Oh my god, Rick. How dumb are you? You're inside a simulation of a simulation... inside another giant simulation!"*  
> — Prince Nebulon, *Rick and Morty, M. Night Shaym-Aliens!*

When we look at a sampling distribution, we are basically looking at the answer to the question: *If I sampled the data again and again, and for every sample I calculate a statistic, how would this statistic be distributed?*. This process can easily be simulated by assuming a value for $\theta$ and repeatedly sampling from the same distribution.

Here, we will go a step further, and ask ourselves: *What happens, in many different worlds, when we sample again and again and look at the sampling distribution?* To explore this, we simulate data - and hence sample distributions - under various values of our parameter $\theta$, generating a multitude of “alternate realities” for each assumed true effect size $\theta$. In each one, we repeatedly simulate what might happen if we reran the same experiment. Then, by varying $\theta$, we create a second layer of alternate realities — different possible worlds where the treatment effect is smaller, larger, or even detrimental.

This is our plan:
* We will simulate many different worlds, with $\theta$ ranging from a reduction in conversion time of 60 seconds to an increase in time of 60 seconds.
* For each of these, we will simulate running the same experiment, on the same users, 10k times.
* In each simulated experiment, we assign 100 users to group A and 100 to group B.

<img src="/assets/img/testing_logic_files/simulation_diagram.jpg" alt="Drawing" style="width: 1000px;"/>


```python
import numpy as np

sample_size = 100
number_of_samples = 10000
thetas = np.arange(-60, 65, 5)
mu_a = 300
sigma = 50

samples_a = {
    theta: np.array([
        np.random.normal(mu_a, sigma, size=sample_size) 
        for _ in range(number_of_samples)
    ])
    for theta in thetas
}

samples_b = {
    theta: np.array([
        np.random.normal(mu_a + theta, sigma, size=sample_size) 
        for _ in range(number_of_samples)
    ])
    for theta in thetas
}
sampling_distributions = {theta:None for theta in thetas}
for theta in thetas:
    sampling_distributions[theta] =  (
        samples_b[theta].mean(axis = 1) - 
        samples_a[theta].mean(axis = 1)
    )
```

Now we have a full set of simulated sampling distributions: for each value of $ \theta $, we know how the difference in sample means $ \hat{\theta} $ behaves over multiple experiments.

Next, let’s look at what happens in two different worlds—one where the null hypothesis is true, and one where it isn’t. We’ll use a decision threshold at $ c = -10 $. If our observed $ \hat{\theta} $ falls below this threshold—that is, if group B's mean conversion time is more than 10 seconds faster than group A's — we reject the null hypothesis. Even if the value is a bit arbitrary, it makes intuitive sense: our null hypothesis is that in the new variant conversion times are the same or slower. Obviously, we want to falsify this for some statistic below zero. Why not just zero? We need to take into account the noise: especially with our limited sample size (100 for each group), we could get a $\hat{\theta}$ well below zero even under the null. Picking -10 gets us some extra safety.

By colouring the distribution according to outcomes, we can visualise the possible consequences of our decision rule:
- **False Positives** (Type I errors): rejecting $ H_0 $ when it’s true
- **False Negatives** (Type II errors): failing to reject $ H_0 $ when it’s false
- **True Positives** and **True Negatives**: when our decision aligns with the truth

This helps us building an intuitive feel for what a statistical test is doing — not just calculating a number, but navigating trade-offs under uncertainty.

    
![png](/assets/img/testing_logic_files/testing_logic_5_0.png)
    


* In the top plot, we see the sampling distribution for $\theta = 0$, divided into two regions, below and above and decision threshold -10. This is the value of $\theta$ compatible with the null hypothesis that could give us the most ambiguos results. Even then, we still get that in the majority of worlds, we do not reject the null. This is the extra safety that picking a number a bit below zero gave us.

* In the bottom plot, we see what happens when the $\theta = -10$: even though 10 seconds faster might be a significant value, we reject the null only half of the times. This is the cost of the extra safety we bought.

This kind of visualisation makes it much easier to understand what’s really going on when we reject the null. It's not a verdict - it’s a probabilistic decision rule applied to a noisy measurement. And depending on which world we're in, that rule might lead us to the right decision - or to an error. In essence, it all boils down to optimising a tradeoff depending on what error (False Positive or False Negative) we deem more costly.

## Power Analysis

> "Everybody's got a choice, Marty... I sure blamed you."  
> — *Rust Cohle, True Detective*

In hypothesis testing, just like in life, there’s no escaping the need to choose.  Minimizing one type of error always increases the risk of the other — the only real question is *which tradeoff you're willing to live with*.


This can be formalised through the **power function**, which answers the question: *For each value of $\theta$, how likely is our test to fall into the critical region?*. Formally, we define the power function:

$$
\gamma(\theta) = \mathbb{P}_{\theta}[T > c]
$$

Ideally, we want this to be low for $\theta \in \Theta_0$ and high for $\theta \in \Theta_1$. This is the central tradeoff we have to face in hypothesis testing: given a sample size, if we want to lower the power function in $\Theta_0$, and hence get a lower False Positive rate, we need to accept that it will also lower in $\Theta_1$, hence increasing the False Negative rate. The only way of having a win-win is increasing the sample size, which increases how steep the power function is. But big samples are not cheap, we better learn to deal with this choice.

Tipically, the information in the power function is reduced to two single numbers: the **test size** and the **power of the test**.

### Test Size

We call the **size** of the test the maximum probability of rejecting the null when it’s true:

$$
\alpha = \sup_{\theta \in \Theta_0} \gamma(\theta)
$$

The last bit of jargon we need is the **level**: a test is said to have level $\alpha$ if its size is less than or equal to $\alpha$. 

### Power of the Test

Given a particular $\theta = \theta_s$ that we are interesting in detecting - say, in our example, 20 seconds of time reduction for conversion - then we can define the power of the test as $\gamma(\theta_s)$. This is the probability of rejecting the null when the true $\theta$ is exactly of the magnitude we are interested in detecting.

Let's dig further into this using the simulation:


```python
cs = np.arange(0, -21, -1)
power_functions = {
    c: [
        (s < c).mean() 
        for _, s in sampling_distributions.items()
    ] for c in cs
}
test_sizes = {
    c: (sampling_distributions[0] < c).mean() for c in cs
}

powers = {
    c: (sampling_distributions[-20] < c).mean() for c in cs
}
```
    
![png](/assets/img/testing_logic_files/testing_logic_9_0.png)
    


In the top plot, we see the sampling distributions for all the different values of $\theta$ we simulated. In the bottom plot, we see the power function for two possible value of the decision threshold:

* One using a threshold of 0. This power function results in a big size (we erroneusly reject the null half of the time), but also high power: we virtually always detect the effect size of interest.
* The other using a threshold of -15. This results in a pretty small size (2%), which is close to what the common standard is (5%). However, we also have a significant reduction in power.

We generalise this and look at this tradeoff for many different values of the decision threshold:


```python
fig, ax = plt.subplots()
ax.plot(test_sizes.values(), powers.values());
ax.set_xlabel(r"Test Size $\alpha$")
ax.set_ylabel(r"Test Power")
ax.set_title("Tradeoff between power and size");
```


    
![png](/assets/img/testing_logic_files/testing_logic_11_0.png)
    


## What about p-values?
> "Don't Panic."  
> — *The Hitchhiker’s Guide to the Galaxy*

At this point, you might be thinking: *"I thought hypothesis testing was all about p-values, where are they? why aren't we talking about them?"*. Well, it's complicated, but let's cover them. 

Let's reverse the logic we used in the power analysis and ask ourselves: *assuming we have run a test and we have calculated a test statistic, what is the smallest level of the test at which we can reject H0*? This is what the p-value is. Confused? You are not alone. An equivalent - and perhaps more intuitive way - of thinking about the p-value is that is the **probability, assuming $H_0$ is true, of observing a value of the test statistic the same as or more extreme than what was actually observed**:

$$
\text{p-value} = \sup_{\theta \in \Theta_0} \mathbb{P}_{\theta}[T(X) \geq T(x_{obs})]
$$

The supremum operation ensures that, for composite hypotheses (i.e., where the hypothesis is not as simple as $\theta$ being one specific value, as in our case) we are considering the most conservative scenario.

I’m not sure how much more intuitive that really is. The point is: p-values were not originally created for Hypothesis Testing, and you can perform the whole business without using them. They were created for a different kind of procedure, called significance testing, in which they played an informal measure of evidence against an hypothesis (see the historical note for a little more context). At any rate, they are today commonly used for hypothesis testing, and the way the are used is actually pretty simple (Whether that's correct... is another matter): rather than stating the decision threshold, what one usually does is calculating the p-value, and then uses it as both a measure of evidence against the null, and as a decision tool. When the p-value is below 5%, score.

## Historical Note

The approach we just went through is called Null Hypothesis Significance Testing (NHST), and it's basically a sort of hybrid between two different approaches born in the last century: Significance Testing, as proposed by Ronald Fisher, and Hypothesis Testing, as proposed by Neyman and Pearson. These two approaches had very different philosophical foundations, but somehow they got merged by educators and practitioner at some point in the forties and fifties. If you want to know more about this, see the reading suggestions.

## Further Reading

* [The Empire of Chance](https://www.cambridge.org/core/books/empire-of-chance/9DAF0E94CEB7D88FB8E2BD52460AC70F) is a good resource on the history of probability theory and statistics. In particular, it convers the controversy between Fisher and Neyman/Pearson on the philosophical approaches to testing.
* [Casella and Berger](https://www.taylorfrancis.com/books/mono/10.1201/9781003456285/statistical-inference-roger-berger-george-casella) remains one of the gold standard references for statistical inference, including hypothesis testing.
* A lighter but equally amazing book is [All of Statistics](https://www.stat.cmu.edu/~larry/all-of-statistics/) by Larry Wasserman. It's great, honestly, and if you want an accessible book for mathematical statistics, this is it.
