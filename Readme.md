# PAPER READING :cat:

* [PAPER READING <g-emoji class="g-emoji" alias="cat" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/1f431.png">🐱</g-emoji>](#paper-reading-cat)
   * [Vision](#vision)
      * [3D Perception / NeRF](#3d-perception--nerf)
         * [Spatial Transformer Networks<g-emoji class="g-emoji" alias="white_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2705.png">✅</g-emoji>](#spatial-transformer-networkswhite_check_mark)
         * [Point-Voxel CNN for Efficient 3D Deep Learning<g-emoji class="g-emoji" alias="white_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2705.png">✅</g-emoji>](#point-voxel-cnn-for-efficient-3d-deep-learningwhite_check_mark)
         * [GRF](#grf)
         * [NeX<g-emoji class="g-emoji" alias="white_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2705.png">✅</g-emoji>](#nexwhite_check_mark)
         * [pixel-NeRF<g-emoji class="g-emoji" alias="white_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2705.png">
✅</g-emoji>](#pixel-nerfwhite_check_mark)
         * [Plenoxels: Radiance Fields without Neural Networks <g-emoji class="g-emoji" alias="white_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2705.png">✅</g-emoji>](#plenoxels-radiance-fields-without-neural-networks-white_check_mark)
         * [D-NeRF: Neural Radiance Fields for Dynamic Scenes (CVPR2021)](#d-nerf-neural-radiance-fields-for-dynamic-scenes-cvpr2021)
         * [PlenOctTree <g-emoji class="g-emoji" alias="white_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2705.png">✅</g-emoji>](#plenocttree-white_check_mark)
      * [ViT pretrain](#vit-pretrain)
      * [Video Understanding / Motion Analysis](#video-understanding--motion-analysis)
         * [Learning Motion Priors for 4D Human Body Capture in 3D Scenes](#learning-motion-priors-for-4d-human-body-capture-in-3d-scenes)
         * [Object-Centric Learning with Slot Attention (NIPS 2020)](#object-centric-learning-with-slot-attention-nips-2020)
         * [Unsupervised Discovery of Object Radiance Fields (Jiajun group 2021)](#unsupervised-discovery-of-object-radiance-fields-jiajun-group-2021)
      * [Unsupervised Learning](#unsupervised-learning)
      * [GAN](#gan)
         * [GAN-Supervised Dense Visual Alignment (Junyan Zhu 2021.12 arxiv)](#gan-supervised-dense-visual-alignment-junyan-zhu-202112-arxiv)
   * [NLP](#nlp)
      * [Syntactic Learning](#syntactic-learning)
         * [Enhancing Machine Translation with Dependency-Aware Self-Attention <g-emoji class="g-emoji" alias="white_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2705.png">✅</g-emoji>](#enhancing-machine-translation-with-dependency-aware-self-attention-white_check_mark)
      * [Machine Translation](#machine-translation)
   * [Robot Learning](#robot-learning)
      * [RL-based](#rl-based)
         * [CURL: Contrastive Unsupervised Representations for Reinforcement Learning <g-emoji class="g-emoji" alias="white_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2705.png">✅</g-emoji>](#curl-contrastive-unsupervised-representations-for-reinforcement-learning-white_check_mark)
         * [Dynamics Learning with Cascaded Variational Inference for Multi-Step Manipulation (CoRL 2019) <g-emoji class="g-emoji" alias="white_check_mark" fallback-src="https://github.githubassets.com/images/icons/emoji/unicode/2705.png">✅</g-emoji>](#dynamics-learning-with-cascaded-variational-inference-for-multi-step-manipulation-corl-2019-white_check_mark)
      * [Dynamic-based](#dynamic-based)
         * [Learning to Simulate Complex Physics with Graph Networks](#learning-to-simulate-complex-physics-with-graph-networks)
   * [Multi-modal](#multi-modal)
      * [Generalized Learning / Intuitive Physics / Cognition](#generalized-learning--intuitive-physics--cognition)
* [Conference Tutorials / Lectures](#conference-tutorials--lectures)


## Vision

### 3D Perception / NeRF

#### Spatial Transformer Networks:white_check_mark:

#### Point-Voxel CNN for Efficient 3D Deep Learning:white_check_mark:

#### GRF

#### NeX:white_check_mark:

#### pixel-NeRF:white_check_mark:

#### Plenoxels: Radiance Fields without Neural Networks :white_check_mark:

  - No Neural Nets, 100 times faster on training 
  - Data Structure: Sparse Voxel = (Spherical Harmonic Basis + Opacity)
  - Rendering: Neural Rendering, Trilinear Interpolation
  - Optimization: reconstruction loss + total variation loss (smooth variation)
  - Key: the paper demonstrate that : the soa performance on novel view synthesis of NeRF lies on the Neural Rendering Reconstruction, but not on NN.

#### D-NeRF: Neural Radiance Fields for Dynamic Scenes (CVPR2021)

#### PlenOctTree :white_check_mark:

  - real-time nerf rendering, OCTree structure
  - using spherical harmonic basis to encode view-dependent effects, which is much faster than conventional NeRF rendering where NN inference is needed 
  - learn F: (x, y, z) -> (SHs, $\sigma$)

  



### ViT pretrain



### Video Understanding / Motion Analysis
#### Learning Motion Priors for 4D Human Body Capture in 3D Scenes
  - pose estimation, smoothing
#### Object-Centric Learning with Slot Attention (NIPS 2020)
#### Unsupervised Discovery of Object Radiance Fields (Jiajun group 2021)



### Unsupervised Learning





### GAN

#### GAN-Supervised Dense Visual Alignment (Junyan Zhu 2021.12 arxiv)
  -  Image/Video Align : learn a transformation to congeal a image 





## NLP

### Syntactic Learning
#### Enhancing Machine Translation with Dependency-Aware Self-Attention :white_check_mark:
  - NLP, syntactically-enhanced Transformer


### Machine Translation






## Robot Learning

### RL-based
#### CURL: Contrastive Unsupervised Representations for Reinforcement Learning :white_check_mark:
  -  RL from pixel, contrastive learning, moco style


#### Dynamics Learning with Cascaded Variational Inference for Multi-Step Manipulation (CoRL 2019) :white_check_mark:
  - sequential task for robots, subgoals 
  - C : latent effect space (subgoals), Z: latent motion space (actions), ~N(0, 1)
  - two levels
    - high level task : subgoals -> target states(s_t) : p(s_t|s, c)
    - low level task : actions -> subgoals a~p(s_t|s, c, z) 
- learn from task-agnostic manipulations
- Reinforcement and Imitation Learning for Diverse Visuomotor Skills (2018)
- 

### Dynamic-based

#### Learning to Simulate Complex Physics with Graph Networks





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
