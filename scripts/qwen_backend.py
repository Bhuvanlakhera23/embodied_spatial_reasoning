import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


class QwenVLMBackend:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("[QWEN] Loading model...")

        # ⚠️ Force FP16 for 6GB GPUs
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()

        # ⚠️ Clamp visual token budget for VRAM safety
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28
        )

        print("[QWEN] Model loaded.")

    def run(self, prompt, frame_paths):
        if not frame_paths:
            raise ValueError("No frames provided to VLM backend")

        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": f"file://{p}"} for p in frame_paths]
                )
            }
        ]

        messages[0]["content"] = list(messages[0]["content"]) + [
            {"type": "text", "text": prompt}
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text
