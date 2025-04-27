# AI

  pip install gymnasium
  pip install numpy

the simulator
  pip install pybullet        
  
deep‐RL algorithm
  pip install torch
  pip install stable-baselines3

logging & visualization
  pip install tensorboard


Train the agent
  python3 train_dqn.py

By default this will:
 • run DQN for 10 000 steps
 • log TensorBoard data under ./carpark_tensorboard/
 • save the model as dqn_carpark_minimal.zip

In a separate shell, launch TensorBoard to monitor training:
tensorboard --logdir ./carpark_tensorboard/

Then point your browser at:
  http://localhost:6006
You’ll see “episode_reward” and “episode_success” curves.

# Once training is done, test/run the learned policy:
python3 test_dqn.py
