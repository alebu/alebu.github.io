---
layout: post
title: "Probability Theory Basics"
date: 2025-03-26
categories: probability
---

# Bare Minimum Probability Theory - 1

![title](img/probability.webp)

Probability theory is a deep and fascinating branch of mathematics that deserves dedicated study. If you want to get a solid understanding of Statistical Inference, having a good grasp of Probability is the best place to start. If you want to go down that route, check the further reading section.

However, the purpose of this is to get *some* understanding of Statistical Inference, with as little maths as possible. Hence, a very rough introduction to a few Probability concepts should suffice here. 

In this article we will cover the foundations of the theory: 
* the Sample Space which we'll use to model the possible outcomes of the random phenomenon we want to study
* the Probability Measure, which we'll use to determine the probability of an event
* Conditional Probability, which allows us to make Probability Statements by incorporating partial knowledge about the outcomes.
* Independence, which covers the possible relationship between different events and their probabilities.

## Sample Space and Events

In the intro article, I stated that we can use Probability Theory to model random phenomena. What components does our model need? First of all, we need to list all the possible outcomes of the phenomenon. For example, when we flip a coin, there are two possible outcomes: heads or tails. When we roll a die, we have 6. The set of all the possible outcomes is called the **Sample Space**, which is traditionally denoted with the letter $\Omega$. We call any subset of $\Omega$ an Event. Notice that this is a more general concept than outcomes: an event can include multiple outcomes; for example, we can talk of the event "rolling a die and getting an even number", which includes the outcomes 2, 4 and 6. 

## Probability Measures

Once we define the sample space, we need a way to quantify how likely different events are. This is where probability measures come in. There are many formal ways to define probability, but for our purposes, we can think of it as a number that describes how likely an event is to happen. You can think of a probability measure as a function that assigns a probability to each event.

An intuitive way of thinking about probability - which will also come in useful when we start using Python - is the following: if we observe the random phenomenon we are modelling N times (with N being a very big number), we can count how many times the event happens, and call this number n. As N grows larger and larger, the ratio n/N will converge towards a number, that represents the probability of the event.

In practice, Probability Theory is about assigning a probability to events of interest, and using them to make predictions. 

Probability measures have the following properties:

* They are a number between 0 and 1, where 0 means an event is impossible, and 1 means it is certain to happen.
* The probability of the entire sample space - i.e., the probability of "any outcome occurring" - is 1. 
* If two events A and B are disjoint (they have no outcome in common) then the probability of their union is equal to the sum of their probabilities.

Given an event $A$, we usually denote its probability with $\mathbb{P}[A]$.

Let's now see this in action, with the simplified version of a popular brainteaser. Suppose someone tells you that a family has two children.  What is the probability that at least one of them is a girl? We can start by writing out the sample space: $\Omega = ${BB, BG, GB, GG}. Each outcome here is encoded with the following notation: we use two capital letters, if the first child is a Boy the first letter is B, otherwise G. Same for the second child in the second letter. 

Next, we need to build a probability measure. Let's take a simplistic view here and say that each outcome is equally likely. As the sum of their probabilities must sum up to 1, the each of them will have probability $1/4$. More generally, for any event A, $\mathbb{P}[A] = |A|/4$, where we indicate with $|A|$ the number of outcomes it contains. Why? Well, we know that the events {BB}, {BG}, {GB}, {GG} are disjoint. We can certainly express any event A as the union of some of these. As they are equally likely, the probability of A will just be the number of outcomes it contains multiplied by $1/4$.

You can verify yourself that this measure satisfies the properties we listed above.

Now we are ready to solve the question. What is our event of interest E? Since three of the four possible outcomes (BG, GB, GG) contain at least one girl, the probability is $\mathbb{P}[E] = |E|/4 = 3/4$.

A very important learning tool we will use in this series of articles is coding. Almost every problem we will encounter will be solvable either by relying on theory or code. By using Python, and the Numpy Random library, we can also solve this one: 


```python
import numpy as np

omega = ["BB", "BG", "GB", "GG"]
p = [1/4] * len(omega)
n = 10000
outcomes = np.random.choice(omega, size = n, p = p)
at_least_one_g = np.array([r for r in outcomes if "G" in r])

len(at_least_one_g)/n
```




    0.7485



What have we done here? After defining the sample space and the probability measure, we used the `choice` function to randomly select 10k outcomes (using the `size` parameter). Then, with a list comprehension, we selected the subset of those that have at least one girl. Finally, we computed the probability of having at least one girl as the ratio between the two, which turns out to be approximately 0.75, i.e. 3/4. 

## Conditional Probability

Very often we want to talk about the probability of something happening *given that something else has already happened*. This is done through the use of Conditional Probability. We express the probability of the event A given the event B with $\mathbb{P}[A | B]$. How do we calculate it? We have that:

$$
\mathbb{P}[A |B] = \frac{\mathbb{P}[A \cap B]}{\mathbb{P}[B]}
$$

We are essentially ‘zooming in’ on the cases where B has happened (by taking the intersection of A and B) and renormalising the probability within this reduced sample space (by dividing by $\mathbb{P}[B]$).

Let's apply this with our previous example. Actually, we'll now solve the original version of the brainteaser:

*Suppose someone tells you that a family has two children, and that one of the children is a girl. What’s the probability that the other child is also a girl?*

Perhaps partly because of the phrasing, people normally tend to have the wrong intuition here and guess that the probability is $1/2$. We will use the sample space and the probability measure we have built before. Let's define our events of interest:

* B = one of the children is a girl = {GB, BG, GG}
* A = the other child is also a girl, i.e., both children are girls = {GG}

Then what we want is simply:

$$
\mathbb{P}[A |B] = \frac{\mathbb{P}[A \cap B]}{\mathbb{P}[B]} = \frac{\mathbb{P}[{GG}]}{\mathbb{P}[{GB, BG, GG}]} = \frac{1}{4}\frac{4}{3} = \frac{1}{3}
$$

Again, we can also resort to Python to solve the problem:


```python
both_g = np.array([r for r in at_least_one_g if r == "GG" ])
len(both_g)/len(at_least_one_g)
```




    0.3372077488309953



What we have done here is taking the array of outcomes with at least one girl from the exercise before, and selecting the subset of this where both children are girls. Again, computing the ratio gives approximately the probability that we calculated through the formulas. Notice, however, that we took a different approach from the formula: instead of computing conditional probability algebraically, we directly resampled the sample space, keeping only the cases where at least one child is a girl. Then we looked how often, in this new sample space, both of children are girls.

## Independence

We have situations where learning that an event B has happened tells us nothing new about the probability of another event A. In those cases we have that the conditional probability of A given B is the same as just the probability of A:

$$
\mathbb{P}[A | B] = \mathbb{P}[A]
$$

and we say that events A and B are **independent**. It follows from the equation above that, if two events are independent, the probability of both happening is just the product of their probabilities:

$$
\mathbb{P}[A \cap B] = \mathbb{P}[A]\mathbb{P}[B]
$$

## Final Example: Monty Hall

Let's conclude with a popular puzzle, called the Monty Hall problem. From wikipedia:

Suppose you're on a game show, and you're given the choice of three doors: Behind one door is a car; behind the others, goats. You pick a door, say No. 1, and the host, who knows what's behind the doors, opens another door, say No. 3, which has a goat. He then says to you, "Do you want to pick door No. 2?" Is it to your advantage to switch your choice?

## Further Reading

* For a comprehensive yet accessible book on Probability Theory, go for *Probability and Random Processes*, by Grimmett and Stirzaker. Chapter 1 covers roughly the same ground as this article, obviously more in depth.
* *All of Statistics* by Wasserman, despite being a Mathematical Statistics book, has an excellent introductory part on Probability Theory.
* Probability can be studied at different levels of sophistication. If you want a glimpse of the deeper level, which requires using Measure Theory, have a look at [this](https://betanalpha.github.io/assets/case_studies/probability_theory.html) series of articles by Michael Betancourt. They are one of the best resources you can find on the topic, and they are free. 
