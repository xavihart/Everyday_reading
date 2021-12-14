# PAPER READING :cat:

[toc]


## Vision

### 3D Perception / NeRF

- Unsupervised Discovery of Object Radiance Fields

- Spatial Transformer Networks

- Point-Voxel CNN for Efficient 3D Deep Learning

- GRF

- NeX

- pixel-NeRF

- Plenoxels: Radiance Fields without Neural Networks :white_check_mark:

  - No Neural Nets, 100 times faster on training 
  - Data Structure: Sparse Voxel = (Spherical Harmonic Basis + Opacity)
  - Rendering: Neural Rendering, Trilinear Interpolation
  - Optimization: reconstruction loss + total variation loss (smooth variation)
  - Key: the paper demonstrate that : the soa performance on novel view synthesis of NeRF lies on the Neural Rendering Reconstruction, but not on NN.

- D-NeRF: Neural Radiance Fields for Dynamic Scenes (CVPR2021)

- PlenOctTree

  - real-time nerf rendering
  - using spherical harmonic basis to encode view-dependent effects, which is much faster than conventional NeRF rendering where NN inference is needed 
  - learn F: (x, y, z) -> (SHs, $\sigma$)
  
  



### ViT pretrain



### Video Understanding / Motion Analysis
- Learning Motion Priors for 4D Human Body Capture in 3D Scenes
  - pose estimation, smoothing



### Unsupervised Learning





### GAN

- GAN-Supervised Dense Visual Alignment (Junyan Zhu 2021.12 arxiv)
  -  Image/Video Align : learn a transformation to congeal a image 





## NLP

### Syntactic Learning
- Enhancing Machine Translation with Dependency-Aware Self-Attention :white_check_mark:
  - NLP, syntactically-enhanced Transformer


### Machine Translation






## Robot Learning

### RL-based
- CURL: Contrastive Unsupervised Representations for Reinforcement Learning :white_check_mark:
  -  RL from pixel, contrastive learning, moco style


- Dynamics Learning with Cascaded Variational Inference for Multi-Step Manipulation (CoRL 2019) :white_check_mark:
  - sequential task for robots, subgoals 
  - C : latent effect space (subgoals), Z: latent motion space (actions), ~N(0, 1)
  - two levels
    - high level task : subgoals -> target states(s_t) : p(s_t|s, c)
    - low level task : actions -> subgoals a~p(s_t|s, c, z) 
- learn from task-agnostic manipulations
- Reinforcement and Imitation Learning for Diverse Visuomotor Skills (2018)
- 

### Dynamic-based





## Multi-modal

- Making Sense of Vision and Touch: Self-Supervised Learning of Multimodal Representations for Contact-Rich Tasks (2019)
- 



### Generalized Learning / Intuitive Physics / Cognition

- MarioNette: Self-Supervised Sprite Learning (NIPS2021) :white_check_mark:
  - RGB scene decompose, sprite learning
- Grounding Physical Concepts of Objects and Events Through Dynamic Visual Reasoning (ICLR2021)
- iGibson 2.0: Object-Centric Simulation for Robot Learning of Everyday Household Tasks
- Point-Voxel CNN for Efficient 3D Deep Learning
- SFV: Reinforcement Learning of Physical Skills from Videos :white_check_mark:
  - video -> pose -> imitation for robot
- Modeling Expectation Violation in Intuitive Physics with Coarse Probabilistic Object Representations (2019 Josh Group)




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
