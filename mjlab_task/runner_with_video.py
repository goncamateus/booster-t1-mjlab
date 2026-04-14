import os

import wandb

from mjlab.rl import MjlabOnPolicyRunner


class VideoOnPolicyRunner(MjlabOnPolicyRunner):
    """Runner that uploads recorded training videos to wandb."""

    def save(self, path: str, infos=None):
        super().save(path, infos)
        self._log_videos()

    def _log_videos(self):
        if self.logger_type != "wandb" or not self.log_dir:
            return

        video_folder = os.path.join(self.log_dir, "videos", "train")
        if not os.path.isdir(video_folder):
            return

        for video_name in os.listdir(video_folder):
            if not video_name.endswith(".mp4"):
                continue
            full_path = os.path.join(video_folder, video_name)
            if wandb.run is not None:
                try:
                    wandb.run.log({"video": wandb.Video(full_path)})
                    print(f"[INFO] Logged video to wandb: {video_name}")
                except Exception:
                    pass
