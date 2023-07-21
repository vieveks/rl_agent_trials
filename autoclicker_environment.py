import gymnasium as gym
from gymnasium import spaces
import pyautogui
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

class ScreenshotEnv(gym.Env):
    def __init__(self):
        
        super(ScreenshotEnv, self).__init__()
        self.input_size = 1000
        
        self.action_space = spaces.Box(low=np.array([650, 250]), high=np.array([1350,500]), dtype=np.uint8, shape=(2,))  # Replace with the number of actions
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.input_size, self.input_size, 3), dtype=np.uint8)  # Replace input_size with your desired image size

        # Define the region of interest (ROI)
        self.roi = (650, 250, 700, 350)  # (left, top, width, height)

    def reset(self):
        # Reset the environment
        self.current_step = 0
        return self._capture_state()

    def step(self, action):
        # Perform the action and return the new state, reward, and done flag

        # Perform action based on the action index
        # Replace this with your actual action execution code
        x,y = action
        pyautogui.click(x,y)
        pyautogui.click(850,500)
        # if action == 0:
        #     # Perform action 0
        #     pyautogui.click(x,y)
        #     pass
        # elif action == 1:
        #     # Perform action 1
        #     pass
        # elif action == 2:
        #     # Perform action 2
        #     pass
        # elif action == 3:
        #     # Perform action 3
        #     pass

        # Capture the new state
        new_state = self._capture_state()

        # Calculate the reward
        #reward = 1 if action == dataset_actions[current_step] else 0
        reward = 1

        # Update current step
        self.current_step += 1

        # Determine if the episode is done
        #done = current_step >= len(dataset_images) - 1
        done = False

        return new_state, reward, done, {}

    def _capture_state(self):
        # Capture the screen as the state
        screenshot = pyautogui.screenshot(region=self.roi)
        pil_image = screenshot.resize((self.input_size, self.input_size))
        state = np.array(pil_image)

        return state
    
    def render(self,obs):
        obs.show()

# Create an instance of the environment
env = ScreenshotEnv()

# # Use the environment like a standard Gym environment
# observation = env.reset()
# done = False
# # while not done:
# # running one episode
# for i in range(1):
#     # action = agent.select_action(observation)
#     action = env.action_space.sample()
#     print(action)
#     next_observation, reward, done, _ = env.step(action)
#     #print(observation)
#     env.render(next_observation)
#     observation = np.array(next_observation)
    

# # Continue with training or testing using the environment

# config = {
#     "policy": 'MlpPolicy',
#     "total_timesteps": 2000
# }
# env = gym.wrappers.RecordEpisodeStatistics(env)
# model = PPO(config['policy'], env, verbose=1)
# model.learn(total_timesteps=config['total_timesteps'])
# while True:
#     x, y = pyautogui.position()
#     print(f"Mouse position: X={x}, Y={y}")