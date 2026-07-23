"""VoxCPM2 worker (runs in 'voxcpm2' venv, Python 3.11).

Verified API: from voxcpm import VoxCPM; VoxCPM.from_pretrained("openbmb/VoxCPM2").
generate(text, reference_wav_path, prompt_wav_path, prompt_text, cfg_value, inference_timesteps).
Apache-2.0 -> safe for commercial. Auto language detect. 48kHz output.

Speed:
  - standard PyTorch: RTF ~0.30 (verified, RTX 4090 24GB)
  - Nano-vLLM engine:  RTF ~0.13 (~2x faster) — enabled via env VOXCPM_VLLM=1
    (bf16-capable GPU only, e.g. L4/A10G/A100; NOT T4). Falls back to standard
    automatically if the engine isn't installed or the GPU can't do bf16.
"""
from __future__ import annotations
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from base_worker import BaseWorker, run_worker  # noqa: E402
from protocol import GenRequest  # noqa: E402


class VoxCPM2Worker(BaseWorker):
    model_id = "voxcpm2"

    def load_model(self):
        self._vllm = False
        # T4 is compute capability 7.5. Some Torch builds report BF16 support
        # optimistically, but VoxCPM's internal compile path then fails in Dynamo
        # / einops. Gate compile by actual CUDA capability, not only Torch's BF16
        # helper. Ampere/Ada/Hopper (sm80+) retain the L4 acceleration path.
        bf16_ok = False
        compile_ok = False
        try:
            import torch
            capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            compile_ok = torch.cuda.is_available() and capability[0] >= 8
            bf16_ok = compile_ok and torch.cuda.is_bf16_supported()
            torch._dynamo.config.suppress_errors = True
            if not compile_ok:
                os.environ["TORCHDYNAMO_DISABLE"] = "1"
                try:
                    torch.compiler.disable()
                except Exception:
                    pass
            else:
                os.environ.pop("TORCHDYNAMO_DISABLE", None)
        except Exception:
            pass
        want_vllm = os.environ.get("VOXCPM_VLLM", "0") == "1"
        if want_vllm and bf16_ok:
            try:
                # Nano-vLLM accelerated engine (optional, ~2x). Verified import per the
                # official OpenBMB/VoxCPM README: `pip install nano-vllm-voxcpm` ->
                # `from nanovllm_voxcpm import VoxCPM`. Falls back cleanly if absent.
                from nanovllm_voxcpm import VoxCPM as NanoVoxCPM  # type: ignore
                self._model = NanoVoxCPM.from_pretrained("openbmb/VoxCPM2")
                self._vllm = True
                print("[voxcpm2] Nano-vLLM engine enabled (~2x faster).", flush=True)
                return
            except Exception as e:
                print(f"[voxcpm2] Nano-vLLM unavailable ({e}); using standard engine.",
                      flush=True)
        from voxcpm import VoxCPM
        self._model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)

    def synthesize(self, req: GenRequest) -> float:
        import soundfile as sf
        import numpy as np
        p = req.preset or {}
        text = (req.text or "").strip()
        # The manual UI style hint was previously carried in emotion_tags but never
        # consumed by this worker. Apply it only when the adapted cue does not
        # already contain its own leading VoxCPM2 style prefix.
        if req.emotion_tags and not re.match(r"^\s*\([^\n)]{1,100}\)", text):
            hint = str(req.emotion_tags).strip().strip("()")
            if hint:
                text = f"({hint}) {text}"
        kw = dict(
            text=text,
            cfg_value=p.get("cfg_value", 2.0),
            inference_timesteps=p.get("inference_timesteps", 14),
            normalize=bool(p.get("normalize", False)),
            retry_badcase=True,
            # The library's default allows extremely long bad generations before
            # retrying. A conservative cap catches appended/noisy tails while still
            # allowing naturally slow narration.
            retry_badcase_ratio_threshold=3.0,
        )
        if req.reference_wav:
            # VoxCPM2 clone modes:
            # - Hi-Fi Clone: prompt_wav_path + prompt_text + reference_wav_path (max similarity, style prefix ignored)
            # - Controllable Clone: reference_wav_path + style prefix (emotion/style controllable)
            # - Hybrid: First cue uses Hi-Fi (if reference text provided), rest use Controllable
            has_style_prefix = bool(re.match(r"^\s*\([^\n)]{1,100}\)", text))
            ref_text = (req.reference_text or "").strip()
            clone_mode = getattr(req, "clone_mode", "hybrid")  # "hifi", "controllable", "hybrid"
            cue_index = getattr(req, "cue_index", 0)
            
            kw["reference_wav_path"] = req.reference_wav
            
            if clone_mode == "hifi":
                # Always Hi-Fi: use prompt_wav + prompt_text + reference_wav
                if ref_text:
                    kw["prompt_wav_path"] = req.reference_wav
                    kw["prompt_text"] = ref_text
            elif clone_mode == "controllable":
                # Always Controllable: reference_wav + style prefix (prompt_text ignored)
                pass  # reference_wav_path already set; style prefix in text is used
            else:  # hybrid (default)
                if cue_index == 0 and ref_text:
                    # First cue: Hi-Fi for maximum identity fidelity
                    kw["prompt_wav_path"] = req.reference_wav
                    kw["prompt_text"] = ref_text
                else:
                    # Subsequent cues: Controllable for style/emotion consistency
                    # Style prefix in text will be used automatically
                    pass
        wav = self._model.generate(**kw)
        wav = np.asarray(wav, dtype="float32")
        # sample rate: both engines expose it; default 48kHz for VoxCPM2
        sr = getattr(getattr(self._model, "tts_model", None), "sample_rate", None) \
            or getattr(self._model, "sample_rate", 48000)
        sf.write(req.out_path, wav, sr)
        return float(len(wav)) / float(sr)


if __name__ == "__main__":
    run_worker(VoxCPM2Worker())
