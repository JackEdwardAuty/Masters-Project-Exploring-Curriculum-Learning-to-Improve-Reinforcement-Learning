# Masters-Project-Exploring-Curriculum-Learning-to-Improve-Reinforcement-Learning

Developed exploratory software applying Curriculum Learning, training agents on simpler foundational tasks before progressing to complex ones, to improve reinforcement learning. Tested in an OpenAI MuJoCo environment, this approach achieved significantly faster convergence and improved agent performance compared to traditional methods.

![Single Hider and Seeker Demo](https://github.com/JackEdwardAuty/Masters-Project-Exploring-Curriculum-Learning-to-Improve-Reinforcement-Learning/blob/4491fbb9a7a19187fd1507c2323ca2d7c6e514a9/Single-Hider-and-Seeker-Example-Lossless.webp)

Code base missing since the GitLab repository is unavailable and I no longer have access to my university GitLab profile.

Instead here is the main repository we built upon:

* OpenAI Multi-Agent Emergence Environments - https://github.com/openai/multi-agent-
  emergence-environments
* Other inspiration and requirements used listed at page end.



# Extract from Report

## Summary

The widely adopted approach to human education is organised in a hierarchical fashion, such

that ‘easier’ content - generally content considered to be logically prerequisite to later content

\- is taught initially, prior to more ‘difficult’ content. Within this system, over the course of

a students’ education within a field they are gradually exposed to its more complex content,

allowing them to apply previously learned concepts as required to build a more thorough and

complete understanding of that field. This approach to learning is often formalised under the

broader concept known as a ‘Curriculum’, and we will be adapting and applying this ‘Curriculum

Learning’ approach for use within the Reinforcement Learning paradigm. Specifically, we will

be exploring its use within a modified version of an OpenAI environment utilising the MuJoCo

physics engine \[1]. We found that through the application of curriculum learning techniques,

the speed to convergence, and performance of the agent improved massively over traditional

reinforcement learning approaches.



# Code Base

The aforementioned external resources we used as a basis for, and to inspire, our project are:

• OpenAI Multi-Agent Emergence Environments - https://github.com/openai/multi-agent-
emergence-environments
• TensorFlow https://www.tensorflow.org/
• MuJoCo - http://www.mujoco.org/
• Python interface for MuJoCo - https://github.com/openai/mujoco-py/tree/1.50.1.0
• MuJoCo Worldgen - https://github.com/openai/mujoco-worldgen
• OpenAI Baselines - https://github.com/openai/baselines
• Stable Baselines - https://github.com/hill-a/stable-baselines



