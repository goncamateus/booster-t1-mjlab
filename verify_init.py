from mjlab_task.stand_env import stand_env_cfg


def assert_true(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def main():
    train_cfg = stand_env_cfg(play=False)
    play_cfg = stand_env_cfg(play=True)

    print("=== T1 Stand Configuration Checks ===")

    reward_keys = set(train_cfg.rewards.keys())
    print(f"Reward terms: {sorted(reward_keys)}")
    assert_true(
        "track_linear_velocity" not in reward_keys,
        "track_linear_velocity must be disabled for pure standing.",
    )
    assert_true(
        "track_angular_velocity" not in reward_keys,
        "track_angular_velocity must be disabled for pure standing.",
    )
    assert_true(
        "stable_standing" in reward_keys,
        "stable_standing reward must be active.",
    )

    term_keys = set(train_cfg.terminations.keys())
    print(f"Termination terms: {sorted(term_keys)}")
    assert_true("time_out" in term_keys, "time_out termination must be defined.")
    assert_true(
        "stability_violation" in term_keys,
        "stability_violation termination must be defined.",
    )

    print(f"Train episode length: {train_cfg.episode_length_s}")
    assert_true(
        abs(float(train_cfg.episode_length_s) - 5.0) < 1e-6,
        "Training episode length must be exactly 5.0 seconds.",
    )

    print(f"Play episode length: {play_cfg.episode_length_s}")
    assert_true(
        abs(float(play_cfg.episode_length_s) - 20.0) < 1e-6,
        "Play episode length must remain 20.0 seconds.",
    )

    print("All stand-task checks passed.")


if __name__ == "__main__":
    main()
