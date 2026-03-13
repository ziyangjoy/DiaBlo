"""
DiaBlo: Diagonal Block Linear module for parameter-efficient fine-tuning.

Replaces standard nn.Linear layers with block-diagonal adapters,
enabling efficient fine-tuning with significantly fewer trainable parameters.
"""

import torch
import torch.nn as nn
import math


class BlockLinear(nn.Module):
    """Block-diagonal linear adapter for parameter-efficient fine-tuning.

    Adds a block-diagonal update to a frozen linear layer.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        num_blocks: Number of diagonal blocks.
        bias: Whether the original linear layer has bias.
        drop_out: Dropout rate applied to input before adapter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_blocks: int,
        bias: bool = False,
        drop_out: float = 0.0,
    ):
        super().__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.in_features = in_features
        self.out_features = out_features

        self.block_size_in = math.ceil(in_features / num_blocks)
        self.block_size_out = math.ceil(out_features / num_blocks)

        N1 = math.ceil(in_features / self.block_size_in)
        N2 = math.ceil(out_features / self.block_size_out)
        self.num_blocks = max(N1, N2)

        self.register_parameter(
            "block_A",
            nn.Parameter(torch.zeros(self.num_blocks, self.block_size_in, self.block_size_out)),
        )
        self.in_diff = self.block_size_in * self.num_blocks - in_features

        self.dropout = nn.Dropout(drop_out)

    def block_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the block-diagonal forward pass."""
        if x.ndim == 2:
            if self.in_diff > 0:
                y = torch.zeros(x.shape[0], x.shape[1] + self.in_diff, device=x.device, dtype=x.dtype)
                y[:, : x.shape[1]] = x
                x = y
            x = x.view(x.size(0), self.num_blocks, self.block_size_in)
            outshape = (x.size(0), self.out_features)
            s1 = "bij,ijk->bik"
        elif x.ndim == 3:
            if self.in_diff > 0:
                y = torch.zeros(x.shape[0], x.shape[1], x.shape[2] + self.in_diff, device=x.device, dtype=x.dtype)
                y[:, :, : x.shape[2]] = x
                x = y
            x = x.view(x.size(0), x.size(1), self.num_blocks, self.block_size_in)
            outshape = (x.size(0), x.size(1), self.out_features)
            s1 = "blij,ijk->blik"
        else:
            raise ValueError(f"Input tensor must have 2 or 3 dimensions, got {x.ndim}.")

        result = torch.einsum(s1, x, self.block_A)
        result = result.flatten(start_dim=-2)[..., : self.out_features].reshape(outshape)
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.linear(x)
        x = self.dropout(x)
        result = result + self.block_forward(x)
        return result


def replace_linear_with_blocklinear(
    model,
    num_blocks: int = 64,
    drop_out: float = 0.0,
    target_modules: list = None,
):
    """Recursively replace nn.Linear layers with BlockLinear.

    Args:
        model: PyTorch model, module, or list of modules.
        num_blocks: Number of diagonal blocks.
        drop_out: Dropout rate.
        target_modules: List of module names to replace (e.g. ["q_proj", "v_proj"]).
            If None, replaces all Linear layers.
    """
    if isinstance(model, (list, nn.ModuleList)):
        for m in model:
            replace_linear_with_blocklinear(m, num_blocks, drop_out, target_modules)
        return

    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and (target_modules is None or name in target_modules):
            in_features = module.in_features
            out_features = module.out_features

            new_layer = BlockLinear(
                in_features=in_features,
                out_features=out_features,
                num_blocks=num_blocks,
                bias=module.bias is not None,
                drop_out=drop_out,
            )

            new_layer.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_layer.linear.bias.data = module.bias.data.clone()

            new_layer.to(module.weight.device).to(module.weight.dtype)
            setattr(model, name, new_layer)
        else:
            replace_linear_with_blocklinear(
                module,
                num_blocks=num_blocks,
                drop_out=drop_out,
                target_modules=target_modules,
            )


def replace_blocklinear_with_linear(model):
    """Merge BlockLinear adapters back into standard nn.Linear layers."""
    for name, module in model.named_children():
        if isinstance(module, BlockLinear):
            d1, d2 = module.block_size_in, module.block_size_out
            N = module.num_blocks
            D = torch.zeros(N, d1, N, d2, device=module.linear.weight.device, dtype=module.linear.weight.dtype)
            inds = torch.arange(N)
            D[inds, :, inds, :] = module.block_A.data
            D = torch.reshape(D, (N * d1, N * d2))[: module.in_features, : module.out_features]
            module.linear.weight.data = module.linear.weight.data + D.T

            setattr(model, name, module.linear)
        else:
            replace_blocklinear_with_linear(module)
