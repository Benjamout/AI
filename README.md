# AI Group Assignment - Parralel Parking Agent

To install all necessary libraries: 
  pip install -r requirements.txt

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
