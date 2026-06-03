# 41118 Project Team 6

## Resources
Setting up a virtual environment:  
https://www.w3schools.com/python/python_virtualenv.asp
More information on the original Gymnasium Environment 
https://gymnasium.farama.org/environments/box2d/bipedal_walker/
## Installation
Install Gymnasium Environment
```
pip install swig
pip install gymnasium[box2d]
```
Neccessary(I think) packages to run GPU training:
``` pip install gymnasium stable-baselines3 torch torchaudio torchvision numpy pandas matplotlib opencv-python pygame-ce pybullet Box2D ale-py sympy tqdm``` 
Tensor Board visualisation packages:
``` pip install tensorboard``` 
## Run Commands
To run the training program, run:
``` python train.py``` 
To watch your agent, run:
``` python watch.py``` 
To watch the tensor board, run:
``` tensorboard --logdir=./ppo_tensorboard/``` 
Important note:
Edit "hardcore=True" in both training file and watch file "env" definition, to test or watch model in flat or obstacle world.


