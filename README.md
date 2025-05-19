# AI Group Assignment - Parralel Parking Agent

## Group Information
### Group Compositon
-Benjamin Pedler  
-Connor Mawer   
-Isaac Poulson    

## Install Info  
```bash
  pip install -r requirements.txt
```

## Operation  
### Training
```bash
  python3 train_dqn.py
```
By default this will:  
 • run DQN for 10 000 steps  
 • log TensorBoard data under ./carpark_tensorboard/  
 • save the model as dqn_carpark_minimal.zip  

In a separate shell, launch TensorBoard to monitor training:
```bash
  tensorboard --logdir ./carpark_tensorboard/
```

Then point your browser at:
```bash
  http://localhost:6006
```
You’ll see “episode_reward” and “episode_success” curves.

## Test the learned policy:  
```bash
  python3 test_dqn.py
```
