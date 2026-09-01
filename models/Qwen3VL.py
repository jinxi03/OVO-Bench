"""
Qwen3-VL Eval Code

Weight from:
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

Inference Code from:
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct (transformers >= 4.57, qwen-vl-utils >= 0.0.14)

Settings mirror models/QWen2VL.py so results stay comparable with the official
Qwen2-VL baseline: up to 64 uniformly sampled frames, max_pixels = 360 * 420,
max_new_tokens = 128.  Differences on purpose: flash_attention_2 (installed in the
env), greedy decoding (the checkpoint's generation_config defaults to sampling),
and the response is returned as a str, not a list (the scorer does `gt in response`
and `re.findall` on it, both of which misbehave on a list).

Inference Platform:
- 8B: 1 * RTX 6000 Ada 48GB
"""
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.OVOBench import OVOBenchOffline
from decord import VideoReader


def get_max_frames(video_file_name, max_frames):
    video = VideoReader(video_file_name)
    return min(max_frames, len(video) - 2)


class EvalQwen3VL(OVOBenchOffline):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.args = args
        self._model_init()

    def _model_init(self):
        model_path = self.args.model_path
        # config.json says dtype=bfloat16; pin it explicitly anyway (FA2 needs bf16/fp16).
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_path)

    def inference(self, video_file_name, prompt):
        frames_num = get_max_frames(video_file_name, max_frames=64)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_file_name,
                        "max_pixels": 360 * 420,
                        "nframes": frames_num,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Qwen3-VL uses 16x16 patches and writes per-frame timestamps into the prompt,
        # so the processor must receive the sampled frames together with their metadata.
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        videos, video_metadata = zip(*video_inputs)  # [(tensor, VideoMetadata), ...]
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=list(videos),
            video_metadata=list(video_metadata),
            do_resize=False,  # frames are already resized to the patch grid by qwen_vl_utils
            padding=True,
            return_tensors="pt",
            **video_kwargs,  # do_sample_frames=False: frames are already sampled
        )
        inputs = inputs.to(self.model.device)

        # Inference (greedy; the checkpoint's generation_config defaults to sampling)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
