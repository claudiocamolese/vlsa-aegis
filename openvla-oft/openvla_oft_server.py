
import base64
import io
import traceback
from typing import List

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from experiments.robot.robot_utils import (
    invert_gripper_action,
    normalize_gripper_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


class InferRequest(BaseModel):
    full_image_png_b64: str
    wrist_image_png_b64: str
    state: List[float]
    task_description: str


def decode_image(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return arr


def fix_state(state) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    print(f"[OpenVLA-OFT server] raw state shape={state.shape}, state={state}")

    if state.shape[0] < PROPRIO_DIM:
        padded = np.zeros((PROPRIO_DIM,), dtype=np.float32)
        padded[: state.shape[0]] = state
        state = padded
        print(f"[OpenVLA-OFT server] padded state to {state.shape}: {state}")

    elif state.shape[0] > PROPRIO_DIM:
        state = state[:PROPRIO_DIM]
        print(f"[OpenVLA-OFT server] truncated state to {state.shape}: {state}")

    return state


print("[OpenVLA-OFT server] Loading model...")

cfg = GenerateConfig(
    pretrained_checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,
    use_proprio=True,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=NUM_ACTIONS_CHUNK,
    unnorm_key="libero_spatial_no_noops",
)

vla = get_vla(cfg)
processor = get_processor(cfg)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
proprio_projector = get_proprio_projector(
    cfg,
    llm_dim=vla.llm_dim,
    proprio_dim=PROPRIO_DIM,
)

print("[OpenVLA-OFT server] Ready.")

app = FastAPI()


@app.post("/infer")
def infer(req: InferRequest):
    try:
        full_image = decode_image(req.full_image_png_b64)
        wrist_image = decode_image(req.wrist_image_png_b64)
        state = fix_state(req.state)

        print(f"[OpenVLA-OFT server] full_image type={type(full_image)} shape={full_image.shape} dtype={full_image.dtype}")
        print(f"[OpenVLA-OFT server] wrist_image type={type(wrist_image)} shape={wrist_image.shape} dtype={wrist_image.dtype}")
        print(f"[OpenVLA-OFT server] task={req.task_description!r}")

        observation = {
            "full_image": full_image,
            "wrist_image": wrist_image,
            "state": state,
            "task_description": req.task_description,
        }

        actions = get_vla_action(
            cfg,
            vla,
            processor,
            observation,
            req.task_description,
            action_head,
            proprio_projector,
        )

        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]

        raw_gripper = actions[:, -1].copy()

        processed_actions = []
        for action in actions:
            # Official OpenVLA-OFT LIBERO post-processing.
            # Gripper action: [0, 1] -> [-1, +1], binarized.
            action = normalize_gripper_action(action, binarize=True)

            # Official OpenVLA behavior: flip gripper sign back for LIBERO env.
            if cfg.model_family == "openvla":
                action = invert_gripper_action(action)

            processed_actions.append(action)

        actions = np.asarray(processed_actions, dtype=np.float32)

        print(f"[OpenVLA-OFT server] actions shape={actions.shape}")
        print(f"[OpenVLA-OFT server] raw gripper={raw_gripper}")
        print(f"[OpenVLA-OFT server] processed gripper={actions[:, -1]}")

        return {"actions": actions.tolist()}

    except Exception:
        tb = traceback.format_exc()
        print("[OpenVLA-OFT server] ERROR:")
        print(tb)
        raise HTTPException(status_code=500, detail=tb)
