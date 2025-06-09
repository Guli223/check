# Loss of Plasticity in Deep Continual Learning


## Repository Contents
- [lop/algos](./lop/algos): All the algorithms used in the paper, including our new continual backpropagation algorithm.
- [lop/nets](./lop/nets): The network architectures used in the paper.
- [lop/imagenet](./lop/imagenet): Demonstration and mitigation of loss of plasticity in a task-incremental problem using ImageNet.
- [lop/incremental_cifar](./lop/incremental_cifar): Demonstration and mitigation of loss of plasticity in a class-incremental problem.
- [lop/slowly_changing_regression](./lop/slowly_changing_regression): A small problem for quick demonstration of loss of plasticity.
- [lop/rl](./lop/rl): Loss of plasticity in standard reinforcement learning problems using the PPO algorithm[1].

The README files in each subdirectory contains further information on the contents of the subdirectory.

## System Requirements

This package only requires a standard computed with sufficient RAM (8GB+) to reproduce the experimental results.
However, a GPU can significantly speed up experiments with larger networks such as the residual networks in [lop/incremental_cifar](./lop/incremental_cifar).
Internet connection is required to download many of the datasets and packages.

The package has been tested on Ubuntu 20.04 and python3.8. We expect this package to work on all machines that support all the packages listed in [`requirements.txt`](requirements.txt)

## Installation Guide

Create a virtual environment
```sh
mkdir ~/envs
virtualenv --no-download --python=/usr/bin/python3.8 ~/envs/lop
source ~/envs/lop/bin/activate
pip3 install --no-index --upgrade pip
```

Download the repository and install the requirements
```sh
git clone https://github.com/shibhansh/loss-of-plasticity.git
cd loss-of-plasticity
pip3 install -r requirements.txt
pip3 install -e .
```

Add this lines in your `~/.zshrc` or `~/.bashrc`
```sh
source ~/envs/lop/bin/activate
```

Installation on a normal laptop with good internet connection should only take a few minutes



上述为loss of plasticity原本readme，根据上述内容完成后将三个文件分别替换，lop/rl/run_ppo.py，，lop/algos/rl/buffer.py，lop/algos/rl/ppo.py
替换之后使用后续原本readme
# Loss of plasticity in reinforcement learning

This directory contains the code to demonstrate and mitigate loss of plasticity in reinforcement learning problems from the OpenAI [Gym](https://www.gymlibrary.dev/index.html).
The actor and critic networks are specified in [`../net/policies.py`](../nets/policies.py) and [`../net/valuesf.py`](../nets/valuefs.py) respectively.

The configurations for individual experiments can be found in [`cfg`](cfg). [`cfg/ant/std.yml`](cfg/ant/std.yml) specifies the parameters for _standard PPO_ in the _Ant-v3_ environment.
The following command can be used to perform one run for this configuration file. The `-s` parameter specifies the random seed for the experiment. 
A single run (for 50M time-steps) on a normal laptop takes about 24 CPU-hours.

```sh
python3.8 run_ppo.py -c cfg/ant/std.yml -s 0
```

Configuration files in [`cfg/ant/ns.yml`](cfg/ant/ns.yml), [`cfg/ant/l2.yml`](cfg/ant/l2.yml), and [`cfg/ant/cbp.yml`](cfg/ant/cbp.yml)specify the parameters for
PPO with proper Adam, PPO with L2 regularization, and PPO with continual backpropagation respectively.

After completing 30 runs for the four configuration files specified above, the commands below can be used to plot the left figure below.
The generated figures will be in the [`plots`](plots) directory.
```sh
cd plots/
python3.8 fig4a.py 
```

![](ant.png "Various algorithms on Ant-v3")

