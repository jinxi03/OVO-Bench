"""
Qwen3.5-architecture Eval Code (Qwen3.6, Qwen3.8)

Weight from:
- https://huggingface.co/Qwen/Qwen3.6-27B
- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/Qwen/Qwen3.6-35B-A3B

Inference Code from:
- the model cards above (transformers >= 5, qwen-vl-utils >= 0.0.14)

These checkpoints are natively multimodal Qwen3.5-architecture models (model_type qwen3_5 /
qwen3_5_moe: Gated DeltaNet + gated attention) and reuse Qwen3-VL's vision / video chain, so the
input side mirrors models/Qwen3VL.py: up to 64 uniformly sampled frames, max_pixels = 360 * 420,
16 px patches, per-frame timestamps from the video metadata, greedy decoding, max_new_tokens = 128,
str response.  Differences on purpose: sdpa attention (no flash-attn wheel for torch 2.14 on
glibc 2.28; the linear-attention layers use flash-linear-attention kernels) and thinking switched
off in the chat template, so the model answers straight away instead of spending the 128 tokens
on a <think> block.

Inference Platform:
- 27B dense / 35B-A3B: 2 * RTX 6000 Ada 48GB (52-67 GB bf16, spread over the visible GPUs by
  device_map="auto")
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.OVOBench import OVOBenchOffline
from decord import VideoReader


def get_max_frames(video_file_name, max_frames):
    video = VideoReader(video_file_name)
    return min(max_frames, len(video) - 2)


class EvalQwen3_5(OVOBenchOffline):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.args = args
        self._model_init()

    def _model_init(self):
        model_path = self.args.model_path
        # device_map="auto" spreads the weights over every visible GPU: pin CUDA_VISIBLE_DEVICES.
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
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

        # Preparation for inference; thinking off: the assistant turn opens with an empty think block
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        # Same video chain as Qwen3-VL: 16x16 patches, and the processor needs the frame metadata
        # to write per-frame timestamps into the prompt.
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
