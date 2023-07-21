import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from autoclicker_environment import *

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.update_target_freq = 10

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            # Exploration: choose a random action within the action space range
            action = torch.tensor([random.uniform(650, 1350), random.uniform(250, 500)], device=self.device)
        else:
            # Exploitation: choose the action with the highest Q-value from the model
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0)
                q_values = self.model(state)
                action = q_values.squeeze(0)

        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, device=self.device, dtype=torch.float)
        actions = torch.tensor(actions, device=self.device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1).long())

        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Create an instance of the environment
env = ScreenshotEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# Create an instance of the DQNAgent
agent = DQNAgent(state_size, action_size)

# Training loop
observation = env.reset()
done = False
for episode in range(20):  # Change the number of episodes as needed
    while not done:
        action = agent.act(observation)
        next_observation, reward, done, _ = env.step(action)
        print('action : ',action,'reward :',reward)
        agent.remember(observation, action, reward, next_observation, done)
        observation = next_observation
        agent.replay()
    
    agent.update_target_model()
    observation = env.reset()
    done = False

    if episode % 10 == 0:
        print(f"Episode: {episode}")
        print(f"Epsilon: {agent.epsilon}")

torch.save(agent.model.state_dict(), "agent_model.pth") # saving

# # Create a new instance of the agent
# agent = DQNAgent(state_size, action_size)

# # Load the saved model state dictionary
# agent.model.load_state_dict(torch.load("agent_model.pth")) # reloading


# # Use the trained agent for inference
# observation = env.reset()
# done = False
# while not done:
#     action = agent.act(observation)
#     next_observation, reward, done, _ = env.step(action)
#     env.render(next_observation)
#     observation = next_observation
