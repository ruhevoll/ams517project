# AMS 517 Project
Based on the paper *Policy gradient learning methods for stochastic control with exit time and applications to share repurchase pricing* by Mohamed Hamdouche, Pierre Henry-Labordere, and Huyen Pham. 
## Background
Consider a controlled Markov state process $X = (X^\alpha)_t$ valued in $\mathcal{X} \subset \mathbb{R}^d$ with a control process $\alpha = (\alpha_t)$ valued in $A \subset \mathbb{R}^m$. Given an open set $\mathcal{O}$ of $\mathcal{X}$, denote by $\tau = \tau^\alpha$ the exit
time of of the domain $\mathcal{O}$ before a terminal horizon $T < \infty$, i.e., 

$$\tau = \inf\{t \geq 0 : X_t \notin \mathcal{O}\} \wedge T,$$

with $\inf \emptyset = \infty$. 

**Objective**: To maximize a criterion of the form

$$J(\alpha) = \mathbb{E}[g(X_\tau^\alpha)], \to V_0 \sup_\alpha J(\alpha)$$

over controll processes $\alpha$ for some terminal reward function $g$ on $\mathbb{R}^d$. 
