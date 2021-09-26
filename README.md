## Projects of the "Artificial Intelligence" course (CS 188, UC Berkeley)

This repository contains my solutions to the projects of the course 
of "Artificial Intelligence" (CS188) taught by Pieter Abbeel and Dan Klein
at the UC Berkeley. I used the material from 
[Fall 2018](https://inst.eecs.berkeley.edu/~cs188/fa18/).

- [x] Project 1 - Search
- [x] Project 2 - Multi-agent Search
- [x] Project 3 - MDPs and Reinforcement Learning 
- [x] Project 4 - Ghostbusters (HMMs, Particle filtering, Dynamic Bayes Nets)
- [ ] ~~Project 5 - Machine learning~~ (I won't do this because it is about neural networks, topic I've already studied at a deeper level)


### Project 1 - Graph search - Implementation Notes
[Project 1](https://inst.eecs.berkeley.edu/~cs188/fa18/project1.html) is about applying 
graph search algorithms to PacMan (with no adversaries in the maze)

### Project 2 - Multi-Agent Search
[Project 2](https://inst.eecs.berkeley.edu/~cs188/fa18/project2.html) is about using 
MiniMax ed ExpectiMax to implement a PacMan agent.

### Project 3 - MDPs and Reinforcement Learning
[Project 3](https://inst.eecs.berkeley.edu/~cs188/fa18/project3.html) is about developing 
a PacMan agent using reinforcement learning.

### Project 4 - Ghostbusters
[Project 4](https://inst.eecs.berkeley.edu/~cs188/fa18/project4.html#Q4) is about 
Hidden Markov Models and Particle Filtering.

Problem: the maze is populated with `numGhosts` _invisible_ ghosts and we want PacMan to
catch them all; we don't know where the ghosts are precisely, but we are given some noisy
distances from PacMan to them.

The assignment can be divided into 3 parts:
1. in part 1, the problem is solved using the forward algorithm for HMM (exact inference);
2. in part 2, the problem is solved using approximate inference powered by particle filtering;
3. in part 3, ghosts don't move independently from each other, so the model is described
   by a Dynamic Bayes Net; the problem is still solved by using particle filtering;
   the difference is that rather than using `numGhosts` independent `ParticleFilter`s,
   we now have a single `JointParticleFilter` whose particles are tuples of positions 
   (one for each ghost).