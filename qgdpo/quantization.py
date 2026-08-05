import torch

class QuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Quantized Log-Ratios.
    Allows forward pass quantization while letting exact gradients flow backward.
    """
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, scale: float = 127.0) -> torch.Tensor:
        ctx.scale = scale
        scaled = torch.clamp(input_tensor * scale, -128.0, 127.0)
        return torch.round(scaled) / scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Gradient passes straight through to input_tensor; scale is non-differentiable (None)
        return grad_output, None

class QuantizedReferenceCache:
    """
    Caches reference model log-probabilities in 8-bit unsigned integers 
    to drastically reduce VRAM usage on single/multi-GPU setups.
    """
    @staticmethod
    def compress(ref_logps: torch.Tensor, num_bits: int = 8):
        min_val = ref_logps.min()
        max_val = ref_logps.max()
        scale = (max_val - min_val) / (2**num_bits - 1)
        
        quantized = torch.clamp(
            torch.round((ref_logps - min_val) / (scale + 1e-8)), 0, 255
        ).to(torch.uint8)
        
        return quantized, min_val, scale

    @staticmethod
    def decompress(quantized: torch.Tensor, min_val: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (quantized.to(torch.float32) * scale) + min_val
