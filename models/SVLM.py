"""
Bridge to svlm (src/svlm in the streaming_video superproject)

Any --model name registered in svlm.methods.METHODS runs through OVO-Bench's official loop: the
harness still picks the clip ([0:t], pre-chunked) and builds the prompt, while the frame sampler,
pixel budget and model backend all live in svlm, so the same recipe runs unchanged on other
benchmarks.  --model_path is optional (the registry has a default checkpoint per method).

Inference Platform:
- see the method's backend preset in svlm.backends.qwen (1 GPU for 8B, 2 GPUs for 27B / 35B-A3B)
"""
from utils.OVOBench import OVOBenchOffline
from svlm.methods import build


class EvalSVLM(OVOBenchOffline):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.args = args
        self.method = build(args.model, model_path=args.model_path)

    def inference(self, video_file_name, prompt):
        return self.method.answer(video_file_name, prompt)
