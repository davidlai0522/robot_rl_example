import robosuite
from robosuite.wrappers import GymWrapper
from environment.door_env import Door

if __name__ == "__main__":
    # Create a Panda robot lifting environment
    # env = robosuite.make(
    #     env_name="Lift",
    #     robots="Panda",
    #     has_renderer=True,
    #     has_offscreen_renderer=True,
    #     use_camera_obs=True,
    #     reward_shaping=True,
    # )
    env = Door(
            robots="Panda",
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
        )
    env = GymWrapper(env)
    obs, _ = env.reset()

    # -------------------------------------------------------------------------------
    # NOTE [CS5446]: You can see the action space here.
    # The printout will be: Box(-1.0, 1.0, (7,), float32)
    # This means that the lower bound of the action space is -1.0 and the upper bound of the action space is 1.0,
    # And there are 7 joints to control.
    # To find out how many controllable joints, you can refere to https://robosuite.ai/docs/modules/robots.html
    # -------------------------------------------------------------------------------
    print(f"The action space of Panda Robot is: {env.action_space}")
    
    # -------------------------------------------------------------------------------
    # NOTE [CS5446]: You can see the observation space here.
    # The printout will be: Box(-inf, inf, (196654,), float32)
    # This means that the lower bound of the observation space is -inf and the upper bound of the observation space is inf,
    # And there are 196654 observations to observe.
    # The reason it is so big is because image is also a part of the observation.
    # To find out how many observations, you can refere to https://robosuite.ai/docs/modules/environments.html
    # -------------------------------------------------------------------------------
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