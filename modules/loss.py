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
    """
    Computes CTC loss from (B,T,N) logits + (B,K) padded targets.

    If your targets_bk includes padding, set target_pad_id and provide target_lengths_b,
    or omit padding by packing targets yourself.
    """

    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    log_probs = F.log_softmax(logits_btn, dim=-1)
    batch_size = logits_btn.size(0)
    losses = []
    neg_large = -1e8

    for b in range(batch_size):
        input_len = int(input_lengths_b[b].item())
        target_len = int(target_lengths_b[b].item())

        if input_len <= 0:
            loss_b = logits_btn.new_tensor(float("inf"))
            if target_len == 0:
                loss_b = logits_btn.new_tensor(0.0)
            losses.append(loss_b)
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

        ext_symbols = torch.full(
            (2 * target_len + 1,),
            blank_id,
            dtype=torch.long,
            device=logits_btn.device,
        )
        ext_symbols[1::2] = target
        num_states = ext_symbols.numel()

        alpha = logits_btn.new_full((input_len, num_states), neg_large)
        alpha[0, 0] = log_probs[b, 0, blank_id]
        alpha[0, 1] = log_probs[b, 0, ext_symbols[1]]

        for t in range(1, input_len):
            for s in range(num_states):
                candidates = [alpha[t - 1, s]]
                if s - 1 >= 0:
                    candidates.append(alpha[t - 1, s - 1])
                if (
                    s - 2 >= 0
                    and ext_symbols[s] != blank_id
                    and ext_symbols[s] != ext_symbols[s - 2]
                ):
                    candidates.append(alpha[t - 1, s - 2])

                prev_score = torch.logsumexp(torch.stack(candidates), dim=0)
                alpha[t, s] = prev_score + log_probs[b, t, ext_symbols[s]]

        log_likelihood = torch.logsumexp(alpha[input_len - 1, -2:], dim=0)
        if log_likelihood <= neg_large / 2:
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
