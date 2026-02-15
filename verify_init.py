import torch
from mjlab_task.stand_env import stand_env_cfg
from mjlab.envs import ManagerBasedRlEnv
import gc


def main():
    # Setup environment
    cfg = stand_env_cfg()
    cfg.scene.num_envs = 10  # Standard for verification
    cfg.viewer.enabled = (
        True  # Enable viewer for debugging if needed, but here we just want data
    )

    # Enable viewer for debugging if needed, but here we just want data
    env = ManagerBasedRlEnv(cfg, device="cuda:0")

    # Reset
    print("Resetting environment...")
    obs, extras = env.reset()

    # Print robot info
    robot = env.scene.entities["robot"]

    # Check root position
    root_pos = robot.data.geom_pos_w[0].cpu().numpy()
    print(f"Robot root position: {root_pos}", flush=True)

    # Check foot positions
    # Need to find the site indices. Sites are in robot.data.site_pos_w.
    # We can use robot.site_names.
    foot_site_names = ["left_foot", "right_foot"]
    foot_indices = [robot.site_names.index(name) for name in foot_site_names]

    env_origins = env.scene.env_origins[0].cpu().numpy()
    print(f"Environment origin: {env_origins}", flush=True)
    for i in range(10):
        # Step once
        env.step(
            torch.ones(cfg.scene.num_envs, env.action_space.shape[1], device="cpu")
        )
        root_pos = robot.data.geom_pos_w[0].cpu().numpy()
        foot_positions = robot.data.site_pos_w[0, foot_indices].cpu().numpy()

        print(
            f"Step {i}: Root Z={root_pos[2]}, Foot Zs={foot_positions[:, 2]}",
            flush=True,
        )

    del env
    gc.collect()


if __name__ == "__main__":
    main()
