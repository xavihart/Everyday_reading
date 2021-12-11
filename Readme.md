# PAPERS

[toc]


## Vision

### 3D Perception / NeRF
- Unsupervised Discovery of Object Radiance Fields
- Spatial Transformer Networks
- Point-Voxel CNN for Efficient 3D Deep Learning

### ViT pretrain


### Video Understanding / Motion Analysis
- Learning Motion Priors for 4D Human Body Capture in 3D Scenes
  - pose estimation, smoothing


### Unsupervised Learning





## NLP

### Syntactic Learning
- Enhancing Machine Translation with Dependency-Aware Self-Attention
  - NLP, syntactically-enhanced Transformer


### Machine Translation





## Robot Learning

### RL-based
- CURL: Contrastive Unsupervised Representations for Reinforcement Learning
  -  RL from pixel, contrastive learning, moco style


- Dynamics Learning with Cascaded Variational Inference for Multi-Step Manipulation (CoRL 2019)
  - sequential task for robots, subgoals
  - C : latent effect space (subgoals), Z: latent motion space (actions), ~N(0, 1)
  - two levels
    - high level task : subgoals -> target states(s_t) : p(s_t|s, c)
    - low level task : actions -> subgoals a~p(s_t|s, c, z) 

  - learn from task-agnostic manipulations


### Dynamic-based



### Generalized Learning

- MarioNette: Self-Supervised Sprite Learning (NIPS2021) :white_check_mark:
  - RGB scene decompose, sprite learning

- Grounding Physical Concepts of Objects and Events Through Dynamic Visual Reasoning (ICLR2021)

- iGibson 2.0: Object-Centric Simulation for Robot Learning of Everyday Household Tasks

- Point-Voxel CNN for Efficient 3D Deep Learning
- SFV: Reinforcement Learning of Physical Skills from Videos
  - video -> pose -> imitation for robot




# Conference Tutorials / Lectures

- MIT Deep Learning Seminar highlighting recent work (January 2020) by Animesh Garg
  - Generalization of Robot for different bu similar tasks
    - current paradigm: e.g. Deep RL, env + action + states + reward, sampling inefficient and unstable
    - current paradigm: Visuo-motor skills, need to carefully design the robot model for specifc scene
    - current paradigm, DRL + Visuo-motor, policy -> velocity, impedence of end effector
  - Heuristics often beats RL
    - How can human guide help with robot training (Human in-loop training): Imitation from human heuristics, off-policy RL DDPG
  -  Sequential Tasks
     - Grasp + Oriented-tasks
     - Multi-step reasoning
       - **CAVIN, hierachical planning** (http://ai.stanford.edu/blog/cavin/)

  - Compositional Planning (Generalized)
       - states (e.g. videos) -> program (pick blue, lift bule, place blue, pick red, lift red ...), instead of just ouput actions using end-to-end way
       - task graphs
  - Data for Robotics
    - largest data source : direcly from videos
    - current robot data is much fewer than that of CV and NLP
      - **experts needs show, not label**
      - RoboTurk (? set games for human to collect human manipulations?)


- CVPR 2021 Workshop on Learning from Instructional Videos

- CVPR workshop on solution to general robot by Pieter Abbeel
  - How to effeciently learn generalized policy from pixels ?
  - How to bring pre-traininig into RL
- RSS 2020 KeyNotes of Cognitive Core for Robot Learning by Josh Tenenbaum
  - 
