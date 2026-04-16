import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobotWinInputs(transforms.DataTransformFn):
    """Map RobotWin observations to the model input format."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation.images.cam_high"])
        left_wrist = _parse_image(data["observation.images.cam_left_wrist"])
        right_wrist = _parse_image(data["observation.images.cam_right_wrist"])

        if self.model_type == _model.ModelType.PI0_FAST:
            image = {
                "base_0_rgb": base_image,
                "base_1_rgb": left_wrist,
                "wrist_0_rgb": right_wrist,
            }
            image_mask = {k: np.True_ for k in image}
        else:
            image = {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            }
            image_mask = {k: np.True_ for k in image}

        inputs = {
            "state": np.asarray(data["observation.state"]),
            "image": image,
            "image_mask": image_mask,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class RobotWinOutputs(transforms.DataTransformFn):
    action_dim: int = 14

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
