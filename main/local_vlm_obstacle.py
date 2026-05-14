import re
import torch
from PIL import Image
from collections import Counter
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class LocalVLMObstacleDetector:
    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
        print(f"[LocalVLM] Loading {model_id}")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )

        self.processor = AutoProcessor.from_pretrained(model_id)

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
            "milk box": "milk carton",
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

        print(f"[LocalVLM] Could not normalize output: {text!r}. Falling back to moka pot.")
        return "moka pot"

    def _build_inputs(self, image, instruction: str):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        prompt = f"""
You are controlling a robot arm in a tabletop manipulation scene.

Instruction:
{instruction}

Choose exactly one object that is most likely to physically obstruct the robot arm or gripper during task execution.

Choose only one name from this list:
{", ".join(self.obstacle_candidates)}

Rules:
- Output exactly one object name from the list.
- Do not choose the target object mentioned in the instruction.
- Prefer the object closest to the robot gripper, target object, bowl, plate, or expected motion path.
- Ignore background objects and objects outside the robot workspace.
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

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        return inputs

    @torch.inference_mode()
    def detect(self, image, instruction: str, task_suite_name=None) -> str:
        inputs = self._build_inputs(image, instruction)

        votes = []

        # 1 deterministic + 2 low-temperature samples.
        generation_settings = [
            {"do_sample": False},
            {"do_sample": True, "temperature": 0.2, "top_p": 0.9},
            {"do_sample": True, "temperature": 0.2, "top_p": 0.9},
        ]

        for gen_kwargs in generation_settings:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=16,
                **gen_kwargs,
            )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            votes.append(self._normalize(output_text))

        obstacle = Counter(votes).most_common(1)[0][0]
        print(f"[LocalVLM] votes={votes} selected={obstacle!r}")
        return obstacle