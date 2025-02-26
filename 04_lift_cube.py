import robosuite
from robosuite.wrappers import GymWrapper
from environment.lift_env import Lift

if __name__ == "__main__":
    # Create a Panda robot lifting environment
    env = Lift(
        robots="Panda",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
    )

    env = GymWrapper(env)
    obs, _ = env.reset()

    print(f"The action space of Panda Robot is: {env.action_space}")
    print(f"The observation space of Panda Robot is: {env.observation_space}")
    
    for _ in range(1000):
        # Sample a random action
        random_action = env.action_space.sample()
        print("Random Action:", random_action)
        
        obs, reward, done, _, info = env.step(random_action)
        
        # Render the environment
        env.render()
        
        # Check if the episode is done
        if done:
            print(f"Task completed! Total rewards: {reward}")
            break
        
        # Print results
        print("Reward:", reward)
        print("Done:", done)
    env.close()
    exit()