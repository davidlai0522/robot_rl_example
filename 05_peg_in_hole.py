import robosuite
from robosuite.wrappers import GymWrapper
from environment.peg_in_hole_env import PegInHole

if __name__ == "__main__":
    # Create environment instance
    env = PegInHole(
        robots="Panda",  # Use Panda robot
        gripper_types="default",
        has_renderer=True,  # Enable visualization
        has_offscreen_renderer=False,  # Disable offscreen rendering
        use_camera_obs=False,  # Don't use camera observations
        reward_shaping=True,  # Enable reward shaping
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
