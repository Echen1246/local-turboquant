"""Paper-faithful TurboQuant Q_prod attention kernel (PyTorch level).

Computes scaled dot-product attention directly from the two-part Q_prod
compressed key representation instead of materializing k_hat_prod vectors.

The TurboQuant paper prescribes this for decode-time attention:

    logit[q, k] = <q, k_hat_mse> + sqrt(pi/2)/d * ||r|| * <Sq, qjl>

where:
    k_hat_mse = R^T @ centers[idx] * ||k||    (MSE reconstruction at b-1 bits)
    r         = k - k_hat_mse                  (residual, only norm stored)
    S         = random JL matrix               (shared across all keys in a layer)
    qjl       = sign(S @ r)                    (1-bit per dimension)

The first term is a standard inner product with the MSE-reconstructed key.
The second term is a scalar QJL correction that makes the combined logit
an UNBIASED estimate of <q, k>.  This unbiasedness prevents the systematic
attention drift that causes repetition loops during autoregressive decode.

For HuggingFace Transformers integration, the materialization approach
(returning k_hat_prod from cache.update()) gives mathematically identical
logits and works with standard F.scaled_dot_product_attention.  This module
provides the efficient direct-from-compressed path for custom inference
pipelines or future Triton/CUDA kernels.
"""

from __future__ import annotations

from math import pi, sqrt
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from turboquant.runtime.packed_qmse_cache import PackedMSELayer


def _repeat_kv(tensor: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return tensor
    return tensor.repeat_interleave(n_rep, dim=1)


def qprod_attention(
    query_states: torch.Tensor,
    packed_layer: "PackedMSELayer",
    new_key: torch.Tensor | None = None,
    new_value: torch.Tensor | None = None,
    n_kv_groups: int = 1,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute attention from the Q_prod compressed representation.

    Instead of materializing ``k_hat_prod`` and calling SDPA, this builds
    the attention logits in two parts:

    1. **MSE logits** — ``q @ k_hat_mse.T`` from the ``(b-1)``-bit MSE
       quantized keys (decoded on the fly from packed indices).
    2. **QJL logits** — ``sqrt(pi/2)/d * ||r|| * (q @ S^T) @ signs^T``
       from the 1-bit QJL sign vectors and stored residual norms.

    The two are summed, softmaxed, and multiplied by Q_mse-decoded values.

    Parameters
    ----------
    query_states : Tensor [B, Q_heads, Sq, D]
        Queries after projection and RoPE.
    packed_layer : PackedMSELayer
        Cache layer containing compressed keys (Q_prod) and values (Q_mse).
    new_key, new_value : Tensor [B, KV_heads, Sn, D] or None
        Current decode token(s) in full precision.
    n_kv_groups : int
        ``num_query_heads // num_kv_heads`` for Grouped Query Attention.
    attention_mask : Tensor [B, 1, Sq, Stotal] or None
        Additive attention mask (``0`` = attend, ``-inf`` = mask).

    Returns
    -------
    Tensor [B, Q_heads, Sq, D]
    """
    from turboquant.runtime.packed_qmse_cache import (
        PackedMSELayer,
        _unpack_qjl_signs,
    )

    B, Q, Sq, D = query_states.shape
    head_scale = 1.0 / sqrt(D)
    parts_logits: list[torch.Tensor] = []
    parts_values: list[torch.Tensor] = []

    # ── Compressed history (Q_prod keys, Q_mse values) ──────────────
    if packed_layer.keys_packed is not None and packed_layer._keys_qjl is not None:
        kp = packed_layer.keys_packed
        Sc = kp.original_shape[-2]
        KV = kp.original_shape[1]

        # 1a) MSE logits: q @ k_hat_mse^T
        key_mse = PackedMSELayer._decode_group(
            kp, packed_layer._key_rotation, packed_layer._key_centers,
        )
        key_mse = _repeat_kv(key_mse, n_kv_groups)
        logits_mse = query_states.float() @ key_mse.float().transpose(-2, -1)

        # 1b) QJL logits: sqrt(pi/2)/d * ||r|| * <Sq, qjl>
        S = packed_layer._qjl_matrix  # [D, D]
        q_proj = query_states.float() @ S.float().T  # [B, Q, Sq, D]

        num_vec = kp.num_vectors
        signs = _unpack_qjl_signs(
            packed_layer._keys_qjl, num_vec,
        )  # [N, D]  float32
        signs = signs.view(B, KV, Sc, D)
        signs = _repeat_kv(signs, n_kv_groups)

        logits_qjl = q_proj @ signs.transpose(-2, -1)  # [B, Q, Sq, Sc]

        r_norms = packed_layer._keys_qjl.residual_norms.float()  # [B, KV, Sc]
        r_norms = _repeat_kv(r_norms.unsqueeze(2), n_kv_groups).squeeze(2)
        qjl_scale = sqrt(pi / 2) / D
        logits_qjl = logits_qjl * (r_norms.unsqueeze(2) * qjl_scale)

        parts_logits.append((logits_mse + logits_qjl) * head_scale)

        val = packed_layer._decode_values_full()
        parts_values.append(_repeat_kv(val, n_kv_groups))

    elif packed_layer.keys_packed is not None:
        # Q_mse-only keys (no QJL) — standard materialized path
        key_decoded = packed_layer._decode_keys_full()
        key_decoded = _repeat_kv(key_decoded, n_kv_groups)
        parts_logits.append(
            (query_states.float() @ key_decoded.float().transpose(-2, -1)) * head_scale,
        )
        val = packed_layer._decode_values_full()
        parts_values.append(_repeat_kv(val, n_kv_groups))

    # ── Dense buffer (force_dense layers or dense-decode tokens) ────
    if packed_layer._dense_keys is not None:
        dk = _repeat_kv(packed_layer._dense_keys, n_kv_groups)
        parts_logits.append(
            (query_states.float() @ dk.float().transpose(-2, -1)) * head_scale,
        )
        dv = _repeat_kv(packed_layer._dense_values, n_kv_groups)
        parts_values.append(dv)

    # ── New token (current decode step, full precision) ─────────────
    if new_key is not None:
        nk = _repeat_kv(new_key, n_kv_groups)
        parts_logits.append(
            (query_states.float() @ nk.float().transpose(-2, -1)) * head_scale,
        )
        parts_values.append(_repeat_kv(new_value, n_kv_groups))

    # ── Combine, softmax, weighted sum ──────────────────────────────
    all_logits = torch.cat(parts_logits, dim=-1)
    all_values = torch.cat(parts_values, dim=-2)

    if attention_mask is not None:
        all_logits = all_logits + attention_mask.float()

    attn_weights = torch.softmax(all_logits, dim=-1, dtype=torch.float32)
    attn_weights = attn_weights.to(dtype=query_states.dtype)
    return attn_weights @ all_values
