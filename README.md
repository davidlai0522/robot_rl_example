# Robot_RL_Example

## Description

This is a basic package to showcase how a robot RL simulation environment can be set up.

## Table of Contents

1. [Installation](#installation)
2. [Package Structure](#package-structure)
3. [Usage](#usage)
4. [Contact](#contact)
5. [Special Notes](#special-notes)
6. [Reference](#reference)

## Installation

**Step 1**: Clone the package:
```bash
git clone https://github.com/davidlai0522/robot_rl_example.git
```
**Step 2**: Install the dependencies:
- The main dependencies of this package are: `robotsuite` and `mujoco`:

- You can install the required dependencies by:
    ```bash
    pip install -r requirements.txt
    ```
- If somehow the installation fails, please refer to [install_robosuite](https://robosuite.ai/docs/installation.html) and [install_mujoco](https://github.com/google-deepmind/mujoco).
- This package is tested on `Ubuntu 22.04`, `Python 3.10.12`

## Package Structure
```markdown
/robot_rl_example
    /doc
    /environments:                  ->the RL environment where the reward function can be designed.
        door_env.py                 ->this is the only environment I have played with so far.
    /models:                        ->directory for trained RL agents.
    01_load_robotsuite_env.py:      ->minimal example to load a robosuite environment.
    02_panda_open_door_ppo.py       ->a real script to train and test a RL model.
```

## Usage

**Step 1**: Minimal exmple to load and render an environment:
```bash
python3 01_load_robotsuite_env.py 
```

**You should be able to see the window in `doc/example1.mp4`**

**Step 2**: Train a model (Panda robot opening a door):
```bash
python3 02_panda_open_door_ppo.py --mode train  --total_timesteps 2000000 --final_model_path ppo_panda_door_final
```

<i>Note: You may only see the view of the training in the begining only and the simulation window will be frozen, this is fine as long as the script in the terminal is still ongoing (training the model).</i>

**Step 3**: Test a model (After training for 2000000 steps):
```bash
python3 ./02_panda_open_door_ppo.py --mode test --final_model_path trained_model/ppo_panda_door_2000000_steps.zip 
```

**This is the results after the training: `doc/example_trained_model.mp4`**

## Contact

This package is prepared by David Lai (LinkedIn: [link](https://sg.linkedin.com/in/en-han-lai-1808a11a3))

## Special Notes
Please trasure hunt for all the hidden notes in the scripts that are formatted in:
```python
# -------------------------------------------------------------------------------
# NOTE : Explaination
# -------------------------------------------------------------------------------
```

In summary, for our project (RL part), we just need to modify some parts in the following files:
- environments/door_env.py
    - depends on what scenario we want, we can modify the existing environment provided by the library according to our needs.
    - we will have to design the rewards function based on the observable states.
- 02_panda_open_door_ppo.py
    - we will need to choose/experiment different RL algorithms to use.

## Reference:
- [robosuite](https://robosuite.ai/docs/demos.html)