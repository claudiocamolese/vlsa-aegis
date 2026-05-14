import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


class LocalGLM45VObstacleDetector:
    def __init__(self, model_id="zai-org/GLM-4.5V-FP8"):
        print(f"[LocalGLM45V-FP8] Loading {model_id}")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={
                0: "30GiB",
                "cpu": "160GiB",
            },
            offload_folder="offload_glm45v_fp8",
        )

        self.obstacle_candidates = [
            "moka pot",
            "white storage box",
            "milk carton",
            "wine bottle",
            "red coffee mug",
            "yellow book",
        ]

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        text = re.sub(r"\s+", " ", text)

        for candidate in self.obstacle_candidates:
            if candidate in text:
                return candidate

        aliases = {
            "moka": "moka pot",
            "pot": "moka pot",
            "storage box": "white storage box",
            "white box": "white storage box",
            "box": "white storage box",
            "milk": "milk carton",
            "carton": "milk carton",
            "bottle": "wine bottle",
            "wine": "wine bottle",
            "red mug": "red coffee mug",
            "coffee mug": "red coffee mug",
            "mug": "red coffee mug",
            "book": "yellow book",
        }

        for key, value in aliases.items():
            if key in text:
                return value

        print(f"[LocalGLM45V-FP8] Could not normalize output: {text!r}. Falling back to moka pot.")
        return "moka pot"

    @torch.inference_mode()
    def detect(self, image, instruction: str, task_suite_name=None) -> str:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        prompt = f"""
You are controlling a robot arm in a tabletop manipulation scene.

Instruction:
{instruction}

Identify exactly one non-robot obstacle that is most likely to physically obstruct the robot arm or gripper during task execution.

Choose only one object from this list:
{", ".join(self.obstacle_candidates)}

Rules:
- Output exactly one name from the list.
- Do not choose the target object mentioned in the instruction.
- Prefer the object closest to the robot gripper, target object, or expected motion path.
- Do not explain.
- Output only the object name.
"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move only tensors. With device_map="auto", use the first parameter device.
        first_device = next(self.model.parameters()).device
        inputs = {
            k: v.to(first_device) if hasattr(v, "to") else v
            for k, v in inputs.items()
        }

        generated = self.model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

        input_len = inputs["input_ids"].shape[-1]
        output_ids = generated[:, input_len:]

        output_text = self.processor.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        obstacle = self._normalize(output_text)
        print(f"[LocalGLM45V-FP8] raw={output_text!r} normalized={obstacle!r}")
        return obstacle