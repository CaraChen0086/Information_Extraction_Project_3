import torch
import torch.nn.functional as F

def ctc_loss_from_logits(
    logits_btn: torch.Tensor,          # (B, T, N) unnormalized scores
    targets_bk: torch.Tensor,          # (B, K) int64 target labels
    input_lengths_b: torch.Tensor,     # (B,) lengths in frames (<= T)
    target_lengths_b: torch.Tensor,    # (B,) lengths in symbols (<= K)
    blank_id: int = 0,
    target_pad_id = None,  # if targets are padded, set this (e.g., blank_id)
    reduction: str = "mean",
    zero_infinity: bool = True,
):
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    log_probs = F.log_softmax(logits_btn, dim=-1)
    batch_size = logits_btn.size(0)
    losses = []
    neg_large = -1e8
    device = logits_btn.device

    for b in range(batch_size):
        input_len = int(input_lengths_b[b].item())
        target_len = int(target_lengths_b[b].item())

        if input_len <= 0:
            losses.append(logits_btn.new_tensor(0.0 if target_len == 0 else float("inf")))
            continue

        target = targets_bk[b, :target_len].long()
        if target_pad_id is not None:
            target = target[target != target_pad_id]
            target_len = int(target.numel())

        if target_len == 0:
            losses.append(-log_probs[b, :input_len, blank_id].sum())
            continue

        repeats = (target[1:] == target[:-1]).sum().item() if target_len > 1 else 0
        if input_len < target_len + repeats:
            losses.append(logits_btn.new_tensor(float("inf")))
            continue

        # Extended symbol sequence: blank s1 blank s2 blank ... sK blank
        ext_symbols = torch.full((2 * target_len + 1,), blank_id, dtype=torch.long, device=device)
        ext_symbols[1::2] = target
        S = ext_symbols.numel()

        # Precompute skip mask: can transition from s-2 to s?
        # Yes if s >= 2, ext_symbols[s] != blank, ext_symbols[s] != ext_symbols[s-2]
        skip_mask = torch.zeros(S, dtype=torch.bool, device=device)
        if S >= 3:
            skip_mask[2:] = (ext_symbols[2:] != blank_id) & (ext_symbols[2:] != ext_symbols[:-2])

        # Initialize: only states 0 and 1 are reachable at t=0
        alpha = logits_btn.new_full((S,), neg_large)
        alpha[0] = log_probs[b, 0, blank_id]
        if S > 1:
            alpha[1] = log_probs[b, 0, ext_symbols[1]]

        # Forward pass — vectorized over states, sequential over time
        neg_large_vec = logits_btn.new_full((S,), neg_large)

        for t in range(1, input_len):
            # Transition from same state
            t0 = alpha

            # Transition from s-1 (pad with -inf on the left)
            t1 = F.pad(alpha[:-1], (1, 0), value=neg_large)

            # Transition from s-2 (pad with two -inf on the left), only where skip is allowed
            t2_raw = F.pad(alpha[:-2], (2, 0), value=neg_large)
            t2 = torch.where(skip_mask, t2_raw, neg_large_vec)

            alpha = torch.logsumexp(torch.stack([t0, t1, t2], dim=0), dim=0) \
                    + log_probs[b, t, ext_symbols]

        log_likelihood = torch.logsumexp(alpha[-2:], dim=0)
        if log_likelihood.item() <= neg_large / 2:
            losses.append(logits_btn.new_tensor(float("inf")))
        else:
            losses.append(-log_likelihood)

    loss = torch.stack(losses)

    if zero_infinity:
        zero = logits_btn.sum() * 0.0
        loss = torch.where(torch.isfinite(loss), loss, zero.expand_as(loss))

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
