"""
Deep Q-Learning Model
Main Pipeline
Implemented Without High Level Libraries
"""
from __future__ import annotations
import os
import copy

# Import numerical libraries for main pipeline
import math
import numpy as np

"""
Neural Network Class
"""
class NeuralNetwork:
    # NeuralNetwork initialiser
    # Initialise neural network weights, biases and Adam optimiser state
    def __init__(self,
                 input_size : int = 22,
                 # INPUTS
                 # Input layer contains 22 neurons
                 # danger_straight, danger_right, danger_left
                 # dir_l, dir_r, dir_u, dir_d
                 # fire_to_left, fire_to_right, fire_above, fire_below
                 # storm_to_left, storm_to_right, storm_above, storm_below
                 # battery_to_left, battery_to_right, battery_above, battery_below
                 # dist_norm, idx_norm, charge_norm
                
                 # HIDDEN LAYER
                 # Hidden layer contains 256 neurons with ReLU activation function
                 hidden_size: int = 256,

                 output_size: int = 3) -> None:
                 # OUTPUTS
                 # Output layer contains 3 neurons producing 3 Q-values for each agent action
                 # Q(s, straight), Q(s, turn right), Q(s, turn left)



        # XAVIER INITIALISATION
        # Initialise neural network weights with Xavier initialisation
        # Xavier initialisation prevents exploding gradients during training
        lim1 = math.sqrt(6 / (input_size + hidden_size))
        lim2 = math.sqrt(6 / (hidden_size + output_size))

        # Weight martrices and bias vectors
        self.W1 = np.random.uniform(-lim1, lim1, (hidden_size, input_size))
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.uniform(-lim2, lim2, (output_size, hidden_size))
        self.b2 = np.zeros((output_size, 1))

        # ADAM OPTIMISER
        # Initialise Adam optimiser moments for all parameters
        # Adam optimiser smooths and balances each neural network weight update
        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)
        self.t = 0

    # FORWARD PROPAGATION
    # Compute Q-values for a batch of states
    def forward(self,
                x: np.ndarray,
                *,
                cache: bool = True) -> np.ndarray:
        # Transpose inputs
        xT = x.T                                
        # First affine transform
        # Compute hidden pre-activations
        z1 = self.W1 @ xT + self.b1             
        # Apply ReLU activation
        a1 = np.maximum(0, z1)             
        # Second affine transform and transpose
        # Compute output pre-activations and transpose
        q  = (self.W2 @ a1 + self.b2).T         

        # Store for backward pass
        if cache:
            self.x, self.z1, self.a1 = xT, z1, a1

        return q

    # BACKPROPAGATION
    # Backpropagate loss gradient through neural network and update parameters with Adam optimiser
    def backward(self,
                 dQ: np.ndarray, # Gradient
                 # Adam optimiser hyperparameters
                 lr: float, # Learning rate (alpha)
                 beta1: float = 0.9, # Exponential decay rate of first moment
                 beta2: float = 0.999, # Exponential decay rate of second moment
                 eps: float = 1e-8) -> None: # Epsilon
    
        # Batch size
        B = dQ.shape[0] 

        # Gradients
        dz2 = dQ.T                               
        dW2 = dz2 @ self.a1.T               
        db2 = dz2.sum(axis=1, keepdims=True)     

        # Propagate back into hidden layer
        da1 = self.W2.T @ dz2  
        # Backprogagation through ReLU activation                  
        dz1 = da1 * (self.z1 > 0)              

        # Gradients
        dW1 = dz1 @ self.x.T                    
        db1 = dz1.sum(axis=1, keepdims=True)     

        # Adam optimiser timestep
        self.t += 1

        # Adam optimiser update rule
        def adam(param, grad, m, v):
            # First moment estimate
            m[:] = beta1 * m + (1 - beta1) * grad
            # Second moment estimate
            v[:] = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** self.t)
            v_hat = v / (1 - beta2 ** self.t)
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

        # Apply Adam optimiser to update neural network weights
        for (W, gW, mW, vW) in ((self.W1, dW1, self.mW1, self.vW1), (self.W2, dW2, self.mW2, self.vW2)):
            adam(W, gW, mW, vW)

        # Apply Adam optimiser to update neural network biases
        for (b, gb, mb, vb) in ((self.b1, db1, self.mb1, self.vb1), (self.b2, db2, self.mb2, self.vb2)):
            adam(b, gb, mb, vb)

    # Save neural network weights and biases
    def save(self, fname: str = "model.npz") -> None:
        os.makedirs("model", exist_ok=True)
        np.savez(os.path.join("model", fname), W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    # Load neural network weights and biases
    def load(self, path: str) -> None:
        d = np.load(path)
        self.W1[:] = d["W1"]
        self.b1[:] = d["b1"]
        self.W2[:] = d["W2"]
        self.b2[:] = d["b2"]

"""
Deep Q-network Trainer Class
"""
class DQNTrainer:
    
    # DQNTrainer initialiser
    # Initialise Deeq Q-network trainer
    def __init__(self,
                 model: NeuralNetwork, # Neural network to train
                 lr: float, # Learning rate (alpha)
                 gamma: float) -> None: # Discount factor (gamma)
        self.model = model
        self.target = copy.deepcopy(model)
        self.lr = lr
        self.gamma = gamma
        self.out_dim = 3
        self.sync_every = 1000
        self.step_cnt = 0

    # Stack inputs into 2D arrays for batch processing
    @staticmethod
    def _stack(v, dtype=np.float32, want_int=False) -> np.ndarray:
        arr = np.asarray(v, dtype=np.int32 if want_int else dtype)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)

        elif arr.dtype == object:
            arr = np.stack(v)

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    # Perform one training step
    def train_step(self,
                   state, # Stack state
                   action,
                   reward,
                   nxt,
                   done) -> float:
        
        # States
        S = self._stack(state)
        # Next states
        NXT = self._stack(nxt)
        # Actions
        A = self._stack(action, dtype=np.int32, want_int=True)
        # Rewards
        R = np.asarray(reward, dtype=np.float32).reshape(-1)
        done = np.asarray(done, dtype=bool).reshape(-1)

        # Batch size
        B = S.shape[0]      

        # Build target Q-values
        Q_next = self.target.forward(NXT, cache=False)
        target = self.model.forward(S,  cache=True)

        # Determine action indices for indexing
        act_idx = np.argmax(A, axis=1) if A.shape[1] == self.out_dim else A[:, 0]

        # BELLMAN EQUATION
        for i in range(B):
            q_new = R[i]
            if not done[i]:
                q_new += self.gamma * np.max(Q_next[i])
            target[i, act_idx[i]] = q_new

        # Compute gradient of Mean Squared Error (MSE) loss
        Q_pred = self.model.forward(S, cache=True)
        dQ = 2 * (Q_pred - target) / (B * self.out_dim)

        # Backpropagation and Adam optimiser
        self.model.backward(dQ, self.lr)

        # POLYAK AVERAGING
        # Soft update target network with Polyak Avergaing
        self.step_cnt += 1
        if self.step_cnt % self.sync_every == 0:
            tau = 0.005
            for a, b in ((self.model.W1, self.target.W1),
                         (self.model.b1, self.target.b1),
                         (self.model.W2, self.target.W2),
                         (self.model.b2, self.target.b2)):
                
                b[:] = tau * a + (1.0 - tau) * b
        
        # Return MSE loss
        return float(np.mean((Q_pred - target) ** 2))