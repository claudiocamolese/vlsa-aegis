import base64
import io

import numpy as np
import requests
from PIL import Image


class OpenVLAOFTPolicy:
    def __init__(self, server_url="http://127.0.0.1:8766/infer", **kwargs):
        self.server_url = server_url
        print(f"[OpenVLA-OFT client] server_url={server_url}")

    def _extract(self, element, keys):
        for key in keys:
            if key in element:
                return element[key]
        raise KeyError(f"Missing keys {keys}. Available keys: {list(element.keys())}")

    def _to_png_b64(self, image):
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            arr = np.asarray(image)

            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)

            img = Image.fromarray(arr).convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def infer(self, element):
        full_image = self._extract(
            element,
            [
                "observation/image",
                "observation/agentview_image",
                "full_image",
                "image",
            ],
        )

        wrist_image = self._extract(
            element,
            [
                "observation/wrist_image",
                "observation/robot0_eye_in_hand_image",
                "wrist_image",
            ],
        )

        state = self._extract(
            element,
            [
                "observation/state",
                "state",
                "proprio",
                "robot_state",
            ],
        )

        task_description = self._extract(
            element,
            [
                "prompt",
                "instruction",
                "task_description",
            ],
        )

        payload = {
            "full_image_png_b64": self._to_png_b64(full_image),
            "wrist_image_png_b64": self._to_png_b64(wrist_image),
            "state": np.asarray(state, dtype=np.float32).tolist(),
            "task_description": str(task_description),
        }

        r = requests.post(self.server_url, json=payload, timeout=300)
        r.raise_for_status()

        actions = np.asarray(r.json()["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]

        print(f"[OpenVLA-OFT client] actions shape={actions.shape}")
        return {"actions": actions}
