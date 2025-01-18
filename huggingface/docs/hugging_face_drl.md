# Hugging Face DRL Course

- [deep-rl-course](https://huggingface.co/learn/deep-rl-course)
- [Deep-RL-Course-Certification](https://huggingface.co/spaces/huggingface-projects/Deep-RL-Course-Certification/)
- [Deep-Reinforcement-Learning-Leaderboard](https://huggingface.co/spaces/huggingface-projects/Deep-Reinforcement-Learning-Leaderboard)
- [Check-My-Progress](https://huggingface.co/spaces/ThomasSimonini/Check-my-progress-Deep-RL-Course)

read_token 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
write_token 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



## What is RL

In RL, we build an agent that can make smart decisions.
To make intelligent decisions, our agent will learn from the environment by interacting with it through trial and error and receiving rewards (positive or negative) as unique feedback.

![agent](pictures/hugging-001.jpg)

Its goal is to maximize its expected cumulative reward (because of the reward hypothesis).

The agent’s decision-making process is called the policy π: given a state, a policy will output an action or a probability distribution over actions. That is, given an observation of the environment, a policy will provide an action (or multiple probabilities for each action) that the agent should take.

Our goal is to find an optimal policy π* , aka., a policy that leads to the best expected cumulative reward.

And to find this optimal policy (hence solving the RL problem), there are two main types of RL methods:

- Policy-based methods: Train the policy directly to learn which action to take given a state.
- Value-based methods: Train a value function to learn which state is more valuable and use this value function to take the action that leads to it.

## Two types of value-based methods

The value of a state is the expected discounted return the agent can get if it starts at that state and then acts according to our policy.

![vbm](pictures/hugging-002.jpg)

To find the optimal policy, we learned about two different methods:

### Policy-based methods 

Directly train the policy to select what action to take given a state (or a probability distribution over actions at that state). In this case, we don’t have a value function.

The policy takes a state as input and outputs what action to take at that state (deterministic policy: a policy that output one action given a state, contrary to stochastic policy that output a probability distribution over actions).

And consequently, we don’t define by hand the behavior of our policy; it’s the training that will define it.

###  Value-based methods

Indirectly, by training a value function that outputs the value of a state or a state-action pair. Given this value function, our policy will take an action.

Since the policy is not trained/learned, we need to specify its behavior. For instance, if we want a policy that, given the value function, will take actions that always lead to the biggest reward, we’ll create a Greedy Policy.

### So the difference is:

In policy-based training, the optimal policy (denoted π*) is found by training the policy directly.

In value-based training, finding an optimal value function (denoted Q* or V*, we’ll study the difference below) leads to having an optimal policy.

![optimal](pictures/hugging-003.jpg)

### The state-value function

We write the state value function under a policy π like this:

![state-value](pictures/hugging-004.jpg)

For each state, the state-value function outputs the expected return if the agent starts at that state and then follows the policy forever afterward (for all future timesteps, if you prefer).

### The action-value function

In the action-value function, for each state and action pair, the action-value function outputs the expected return if the agent starts in that state, takes that action, and then follows the policy forever after.

The value of taking action a in state s under a policy π is:

![action-value](pictures/hugging-005.jpg)

We see that the difference is:

- For the state-value function, we calculate the value of a state St
- For the action-value function, we calculate the value of the state-action pair (St,At) hence the value of taking that action at that state.

In either case, whichever value function we choose (state-value or action-value function), the returned value is the expected return.

However, the problem is that to calculate EACH value of a state or a state-action pair, we need to sum all the rewards an agent can get if it starts at that state.

This can be a computationally expensive process, and that’s where the Bellman equation comes in to help us.


## The Bellman Equation

We know that if we calculate V(St) (the value of a state), we need to calculate the return starting at that state and then follow the policy forever after.

So to calculate V(St), we need to calculate the sum of the expected rewards. 

The Bellman equation is a recursive equation that works like this: instead of starting for each state from the beginning and calculating the return, we can consider the value of any state as: The immediate reward Rt+1 + the discounted value of the state that follows γ∗V(St+1).

![bellman](pictures/hugging-006.jpg)

The idea of the Bellman equation is that instead of calculating each value as the sum of the expected return, which is a long process, we calculate the value as the sum of immediate reward + the discounted value of the state that follows.

## Monte Carlo vs Temporal Difference Learning

Remember that an RL agent learns by interacting with its environment. The idea is that given the experience and the received reward, the agent will update its value function or policy.

Monte Carlo and Temporal Difference Learning are two different strategies on how to train our value function or our policy function. Both of them use experience to solve the RL problem.

On one hand, Monte Carlo uses an entire episode of experience before learning. On the other hand, Temporal Difference uses only a step (St,At,Rt+1,St+1) to learn.

### Monte Carlo: learning at the end of the episode

Monte Carlo waits until the end of the episode, calculates Gt(return) and uses it as a target for updating V(St).
So it requires a complete episode of interaction before updating our value function.

- We always start the episode at the same starting point.
- The agent takes actions using the policy. For instance, using an Epsilon Greedy Strategy, a policy that alternates between exploration (random actions) and exploitation.
- We get the reward and the next state.
- At the end of the episode, we have a list of State, Actions, Rewards, and Next States tuples. 
- The agent will sum the total rewards Gt
- It will then update V(st) based on the formula

![monte-carlo](pictures/hugging-007.jpg)

- Then start a new game with this new knowledge.

### Temporal Difference Learning: learning at each step

Temporal Difference, on the other hand, waits for only one interaction (one step) St+1 to form a TD target and update V(St)
using Rt+1 and γ∗V(St+1).

The idea with TD is to update the V(St) at each step.

But because we didn’t experience an entire episode, we don’t have Gt (expected return). Instead, we estimate Gt by adding Rt+1 and the discounted value of the next state.

This is called bootstrapping. It’s called this because TD bases its update in part on an existing estimate V(St+1) and not a complete sample Gt.

![td](pictures/hugging-008.jpg)

To summarize:

With Monte Carlo, we update the value function from a complete episode, and so we use the actual accurate discounted return of this episode.

With TD Learning, we update the value function from a step, and we replace Gt, which we don’t know, with an estimated return called the TD target.

## Q-Learning

Q-Learning is an off-policy value-based method that uses a TD approach to train its action-value function:

Q-Learning is the algorithm we use to train our Q-function, an action-value function that determines the value of being at a particular state and taking a specific action at that state.

Internally, our Q-function is encoded by a Q-table, a table where each cell corresponds to a state-action pair value.

Given a state and action, our Q-function will search inside its Q-table to output the value.

### The Q-Learning algorithm

![q-learning](pictures/hugging-009.jpg)

- Step 1: We initialize the Q-table
- Step 2: Choose an action using the epsilon-greedy strategy

#### The epsilon-greedy strategy

The epsilon-greedy strategy is a policy that handles the exploration/exploitation trade-off.

The idea is that, with an initial value of ɛ = 1.0:

- With probability 1 — ɛ : we do exploitation (aka our agent selects the action with the highest state-action pair value).
- With probability ɛ: we do exploration (trying random action).

At the beginning of the training, the probability of doing exploration will be huge since ɛ is very high, so most of the time, we’ll explore. But as the training goes on, and consequently our Q-table gets better and better in its estimations, we progressively reduce the epsilon value since we will need less and less exploration and more exploitation.

- Step 3: Perform action At, get reward Rt+1 and next state St+1
- Step 4: Update Q(St, At)

To produce our TD target, we used the immediate reward Rt+1 plus the discounted value of the next state, computed by finding the action that maximizes the current Q-function at the next state. (We call that bootstrap).

![value](pictures/hugging-010.jpg)

![q-value](pictures/hugging-011.jpg)

This means that to update our Q(St,At): 

- We need St,At,Rt+1,St+1.
- To update our Q-value at a given state-action pair, we use the TD target.
  
How do we form the TD target?

- We obtain the reward Rt+1  after taking the action At.
- To get the best state-action pair value for the next state, we use a greedy policy to select the next best action. Note that this is not an epsilon-greedy policy, this will always take the action with the highest state-action value.
  
Then when the update of this Q-value is done, we start in a new state and select our action using a epsilon-greedy policy again.

#### Off-policy vs On-policy

The difference is subtle:

- Off-policy: using a different policy for acting (inference) and updating (training).
For instance, with Q-Learning, the epsilon-greedy policy (acting policy), is different from the greedy policy that is used to select the best next-state action value to update our Q-value (updating policy).

- On-policy: using the same policy for acting and updating.
For instance, with Sarsa, another value-based algorithm, the epsilon-greedy policy selects the next state-action pair, not a greedy policy.

![off-on-policy](pictures/hugging-012.jpg)


- [Why do temporal difference (TD) methods have lower variance than Monte Carlo methods?](https://stats.stackexchange.com/questions/355820/why-do-temporal-difference-td-methods-have-lower-variance-than-monte-carlo-met)
- [When are Monte Carlo methods preferred over temporal difference ones?](https://stats.stackexchange.com/questions/336974/when-are-monte-carlo-methods-preferred-over-temporal-difference-ones)

### DQN

The problem is that Q-Learning is a tabular method. This becomes a problem if the states and actions spaces are not small enough to be represented efficiently by arrays and tables. In other words: it is not scalable.

In this case, the best idea is to approximate the Q-values using a parametrized Q-function Qθ(s,a).

This neural network will approximate, given a state, the different Q-values for each possible action at that state. And that’s exactly what Deep Q-Learning does.

![q-table-vs-nn](pictures/hugging-013.jpg)

In Deep Q-Learning, we create a loss function that compares our Q-value prediction and the Q-target and uses gradient descent to update the weights of our Deep Q-Network to approximate our Q-values better.

![dqn-loss](pictures/hugging-014.jpg)

The Deep Q-Learning training algorithm has two phases:

- Sampling: we perform actions and store the observed experience tuples in a replay memory.
- Training: Select a small batch of tuples randomly and learn from this batch using a gradient descent update step.

![dqn-algo](pictures/hugging-015.jpg)

This is not the only difference compared with Q-Learning. Deep Q-Learning training might suffer from instability, mainly because of combining a non-linear Q-value function (Neural Network) and bootstrapping (when we update targets with existing estimates and not an actual complete return).

To help us stabilize the training, we implement three different solutions:

- Experience Replay to make more efficient use of experiences.
- Fixed Q-Target to stabilize the training.
- Double Deep Q-Learning, to handle the problem of the overestimation of Q-values.


#### Experience Replay to make more efficient use of experiences

Experience Replay in Deep Q-Learning has two functions:

1. Make more efficient use of the experiences during the training. Usually, in online reinforcement learning, the agent interacts with the environment, gets experiences (state, action, reward, and next state), learns from them (updates the neural network), and discards them. This is not efficient.

Experience replay helps by using the experiences of the training more efficiently. We use a replay buffer that saves experience samples that we can reuse during the training.

![dqn-exp-replay](pictures/hugging-016.jpg)

This allows the agent to learn from the same experiences multiple times.

2. Avoid forgetting previous experiences (aka catastrophic interference, or catastrophic forgetting) and reduce the correlation between experiences.

- catastrophic forgetting: The problem we get if we give sequential samples of experiences to our neural network is that it tends to forget the previous experiences as it gets new experiences. For instance, if the agent is in the first level and then in the second, which is different, it can forget how to behave and play in the first level.

The solution is to create a Replay Buffer that stores experience tuples while interacting with the environment and then sample a small batch of tuples. This prevents the network from only learning about what it has done immediately before.

In the Deep Q-Learning pseudocode, we initialize a replay memory buffer D with capacity N (N is a hyperparameter that you can define). We then store experiences in the memory and sample a batch of experiences to feed the Deep Q-Network during the training phase.

#### Fixed Q-Target to stabilize the training

When we want to calculate the TD error (aka the loss), we calculate the difference between the TD target (Q-Target) and the current Q-value (estimation of Q).

But we don’t have any idea of the real TD target. We need to estimate it. Using the Bellman equation, we saw that the TD target is just the reward of taking that action at that state plus the discounted highest Q value for the next state.

However, the problem is that we are using the same parameters (weights) for estimating the TD target and the Q-value. Consequently, there is a significant correlation between the TD target and the parameters we are changing.

Therefore, at every step of training, both our Q-values and the target values shift. We’re getting closer to our target, but the target is also moving. It’s like chasing a moving target! This can lead to significant oscillation in training.

Instead, what we see in the pseudo-code is that we:

- Use a separate network with fixed parameters for estimating the TD Target
- Copy the parameters from our Deep Q-Network every C steps to update the target network.

![dqn-q-target](pictures/hugging-017.jpg)

#### Double DQN

- [Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)

This method handles the problem of the overestimation of Q-values.

We face a simple problem by calculating the TD target: how are we sure that the best action for the next state is the action with the highest Q-value?

We know that the accuracy of Q-values depends on what action we tried and what neighboring states we explored.

Consequently, we don’t have enough information about the best action to take at the beginning of the training. Therefore, taking the maximum Q-value (which is noisy) as the best action to take can lead to false positives. If non-optimal actions are regularly given a higher Q value than the optimal best action, the learning will be complicated.

The solution is: when we compute the Q target, we use two networks to decouple the action selection from the target Q-value generation. We:

- Use our DQN network to select the best action to take for the next state (the action with the highest Q-value).
- Use our Target network to calculate the target Q-value of taking that action at the next state.

Therefore, Double DQN helps us reduce the overestimation of Q-values and, as a consequence, helps us train faster and with more stable learning.

Since these three improvements in Deep Q-Learning, many more have been added, such as Prioritized Experience Replay and Dueling Deep Q-Learning. They’re out of the scope of this course but if you’re interested, check the links we put in the reading list.

