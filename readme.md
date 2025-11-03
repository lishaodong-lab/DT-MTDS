# DT-MTDS

## The introduction and implementation of code

Here is the introduction of key files:

| envs | Introduction |
|  ----  | ----  |
|  **env_for_peg_in_hole.py** | The environment of peg-in-hole |
|  **env_for_push_reach.py**  | The environment of push-reach |
|  **env_for_two_cube_exchange.py**  | The environment of position exchange |
|  **env_for_stack.py**  | The environment of stack |



| pcrl | Introduction |
|  ----  | ----  |
|  **pcrl_two.py** |The PCRL used for peg-in-hole and push-reach |
|  **pcrl_four.py** | The PCRL used for position exchange |
|  **pcrl_eight.py** | The PCRL used for stack |

| dt-mtds | Introduction |
|  ----  | ----  |
|  **train.py** | Train teachers|
|  **extract.py** | Collect the teachers’ knowledge |
|  **task_classification.py** | Train the classifier|
|  **distillation.py** | Distill the teachers’ knowledge into the student |

The other folders contain the necessity files for training of our framework. 

Here is the guidance of code implementation
===========================
## Environment establishment
### Configuration
Operation System `Linux`

We recommend `python>=3.8` and `cuda == 1.18`
```shell
pip install -r requirements_dtmtds.txt
```
The environment setup of G-DINO refers to https://github.com/IDEA-Research


## Train according to the following steps (taking the push-reach as an example):
Import `env_for_push_reach.py` (in “envs” folder) and `pcrl_two.py` (in pcrl folder) into `train.py` (in dt-mtds folder)

Execute `train.py` (in dt-mtds folder)

Execute `extract.py` (in dt-mtds folder)

Execute `task_classification.py` (in dt-mtds folder)

Execute `distillation.py` (in dt-mtds folder)
