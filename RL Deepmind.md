# Lecture 2 : David Silverman Deepmind RL



Markov decision processes (MDP) decscribe an environment for reinforcement learning. Here the current state completely characterise the process.

Markov property: The future is independent of the past given the present
$$
p[s_{t + 1} | s_t] = p[s_{t+1}|s_1,\cdots,s_t]
$$

### State transition matrix

For a markov state $ s $ and a successor state $ s' $ the state transition probability is 
$$
P_{ss'} = P[S_{t=1} = s' | S_t = s ]
$$
So state transition matric from all states to all successor state is defined as 
$$
P = \left[P_{11} \cdots P_{1n} \\ 
		P_{21} \cdots P_{2n} \\
		P_{m1} \cdots P_{mn}
		\right]
$$
sum of all rows is 1 <br> The state transition matrix can be sparse or dense depending on our state diagram and how the state diagram works.

### Markov Process

Its a sequesnce of random states $S_1,S_2,\cdots$with markov property

Markov process is a tuple $(S,P)$ <br> Where S = is the state space <br> and P = Transition probability matrix

### Markov Reward Process

Markov reward process is markov process with reward,<br> so a Markov reward process has the tuple $(S,P,R,\gamma)$ <br> where S = is the set of states <br> P = Transition probability martix <br> R = is the reward function for being in state $s$, $R_s = E[R_{t=1}|S_{t} =s]$ <br> $\gamma$ is the discount factor for cummulative reward $\gamma \in [0,1]$

### Return

the return $G_t$ is the total discounted reward into the future (i.e from time step t)
$$
G_t = R_{t+1} + \gamma R_{t=2}+ \cdots =\sum_{k=0}^{\infin} \gamma^k R_{t+k+1}
$$
Why we use discount

- What the model think will happen in the future may not be the perfect model of the world, so we are not relying completely on its assumption on what will happen in the future, so we discount the reward we get in the future.
- Avoid infinite reward if there is a cycle in the model
- Mathematically bound the returns 
- Animals/Humans also prefer immediate discounting

### Value Function

The value function $v(s)$ gives the long term value of state $s$ . The state value function $v(s) $ of a Markov reward process is the expected return for starting in state $s$
$$
v(s) = E[G_t|S_t=s]
$$
### Bellman Equation

It is based on the idea of dynamic programming. The value function can be decomposed into two parts:

* immediate reward $ R_{t+1} $

* discounted value of the successor rewards $ \gamma v(S_{t+1})$
  $$
  \begin{align}
  v(s) & = E[G_t|S_t=s] \\
  & = E[R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots|S_t = s]\\
  & = E[R_{t+1}+\gamma \{R_{t+2}+\gamma R_{t+3}+\cdots\}|S_t=s] \\
  & = E[R_{t+1}+ \gamma G_{t+1}|S_t=s]\\
  & = E[R_{t+1}+ \gamma E[G_{t+1}] | S_t=s] \\
  & = E[R_{t+1}+ \gamma v(S_{t+1})|S_t=s] \text{By the law of iterative expecteation}
  \end{align}
  $$
  

Equation [10] is the bellman equation. Euation 10 comes from the fact that the expectation of return $G_{t+1}$ in Eq [9] is taken following the law of iterative expectation.

The Bellman equation can be written in a matric form
$$
v = R + \gamma P v \\
\text{Here P is the transition matrix}
$$
v is the coloumn vector with one entry per state
$$
\left[]
\begin{array}{c}
v(1) \\
v(2) \\
\vdots \\
v(n)
\end{array}
\right] = 
\left[
\begin{array}{c}
R_1\\
R_2\\
\vdots\\
R_n
\end{array}
\right] + 
\gamma
\left[
\begin{array}{ccc}
P_{11}& \dots & P_{1n}\\
\vdots&&\vdots\\
P_{m1}&\dots&P_{mn}
\end{array}
\right]
\left[]
\begin{array}{c}
v(1) \\
v(2) \\
\vdots \\
v(n)
\end{array}
\right]
$$
The Bellman equations are linear equation so it can be solved.

### Markov Decision Process

A Markov Decision Process is a Markoc Reward process with decision. Like MRP here all states are markov. Here all the machinery is same only we are introduction the action that we can take in a state.

The MDP is a tuple $(S,P,R,A,\gamma)$, where <br> S = is the finite set of states<br> **A = is the finite set of Actions**<br> $P_{ss'}^a = P[S_{t+1}=s'|S_t=s,\textbf{$A_t=a$}]$, P is the transition probability<br> R is the reward function $R_s^a = E[R_{t+1}|S_t=s,\textbf{$A_t = a$}]$<br> $\gamma$ is the diccount factor $\gamma \in [0,1]$

### Policy

Policy is a distribution  over action s in a given state.
$$
\pi(a|s) = P[A_t=a | S_t = s]
$$

- Pilicy defines the behaviour of an agent
- It is dependent on the current state.

### Value function

State Value Function $v_{\pi}(s)$ of an MDP is the Expected Return strating from state $s$ and following the policy $\pi$ 
$$
v_{\pi}(s) = E_{\pi}[G_t | S_t = s]
$$
Action Value Function $q_{\pi}(s,a)$ is the ecpected return starting from state s, taking action a and then following the polcy $\pi$
$$
q_{\pi}(s,a) = E_{\pi} [G_t | S_t = a, A_t = a]
$$
**The Bellman equation** for the **state value function** and **action value function ** can be written as:

Bellman equation for state value function
$$
v_\pi(s) = E_\pi[R_{t+1}+\gamma v_\pi(s_{t+1}) | S_t = s]
$$
Vellman function for action value function 
$$
q_\pi(s,a) = E_\pi[R_{t+1}+\gamma q_\pi(s_{t+1},a_{t+1}) | S_t = s,A_t = a]
$$

---

​																	The relation between $v_\pi$ and $q_\pi$ is
$$
v_\pi(s) = \sum_{a\in A} \pi(a|s)q_\pi(s,a)
$$
When we sum up the product of 

- probability of an action in a state
- Expected return when an action a is taken in that state then follows policy $\pi$

We get the expected return starting from that state and following the policy $\pi$ is the state value function $v_\pi(s)$ 

---

​																The relation between $q_\pi$ and $v_\pi$

So from a state s, if a take an action we get the reard for taking that action. After taking the action I reach state s'. To get the expected return for being in state s and taking the action a we add:

- The Expected Reward for taking action a from state s
- the discounted return from the next state s' covering all possible trajectory

$$
q_\pi(s,a) = R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a v_\pi(s')
$$

Stiching Eq [19] and Eq [20] we get the bell man equation for $v_\pi$
$$
v_\pi(s) = \sum_{a\in A} \pi(a|s)(R_s^a + \gamma \sum_{s' \in S}P_{ss'}^a v_\pi(s'))
$$
we can stich in the opposite direction alos to tget the Bellman Equation for $q_\pi$ 

### Optimal Value Function

The optimal state-value function is the maximum value function for all policies
$$
v_{*}(s) = \max_{\pi} v_{\pi}(s)
$$
The optimal action-value function is the maximum action-value function for all policies 
$$
q_{*}(s,a) = \max_{\pi}q_{\pi}(s,a)
$$
MDP is soved when we know the optimal value function

Partisl ordering of policies 
$$
\pi \geq \pi^{'} \text{if } v_{\pi}(s) \geq v_{\pi}(s'), \forall s
$$

#### Theorem

For any Markov Decision Process

- There exist an optimal policy $\pi_{*}$ that is better than or eqial to other policies $ \pi_{*}\geq \pi , \forall \pi$ 

- Optimal policies achive the optimal value function 

  $v_{\pi_{*}}(s) = v_{*}(s)$

- Optimal policies achive the optimal action-value function $q_{\pi_{*}}(s,a) = q_{*}(s,a)$

- There can exist more than one optimal policy.

To find optimal policy given we have the $q_{*}$, we can find the action that has maximum $q_{*}$ in a state, and then include that action in the optimal policy
$$
\pi_{*}(a|s) = \begin{cases}1, \text{if } a= argmax_{a \in A}q_{*}(s|a) \\ 0, otherwise \end{cases}
$$
There is always a deterministic policy for any MDP

### Bellman Optimality Equation

$$
v_{*}(s) = \max_{a}q_{*}(s,a)
$$

The bellman optima equation for $q_*$
$$
q_*(s,a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a v_*(s')
$$
Now putting eqation 26 and equation 27 together we get the $v_*$ in terms of itself
$$
v_*(s) = \max_\pi R_s^a + \gamma \sum_{s' \in S} P_{ss'}^av_{*}(s)
$$
The equation for $q_*$ is 
$$
q_*(s,a) = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a \max_{a'} q_{*}(s',a')
$$
