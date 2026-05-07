import math
import torch
import torch.nn.functional as F

from modules.loss import ctc_loss_from_logits


def _id_to_letter(dataset):
    return {i: ch for ch, i in dataset.letter2id.items()}


def _ids_to_string(ids, dataset, blank_id=0):
    id2letter = _id_to_letter(dataset)
    chars = []
    for idx in ids:
        idx = int(idx)
        if idx == blank_id:
            continue
        ch = id2letter.get(idx, "")
        if len(ch) == 1:
            chars.append(ch)
    return "".join(chars)


def _vocab_targets(dataset, device):
    unk = dataset.letter2id["<unk>"]
    rows = []
    words = []

    for word in dataset.scr2id.keys():
        if word == "<unk>":
            continue
        ids = [dataset.letter2id.get(ch, unk) for ch in f"|{word.lower()}|"]
        rows.append(torch.tensor(ids, dtype=torch.long, device=device))
        words.append(word)

    lengths = torch.tensor([row.numel() for row in rows], dtype=torch.long, device=device)
    max_len = int(lengths.max().item()) if rows else 0
    targets = torch.full((len(rows), max_len), 0, dtype=torch.long, device=device)

    for i, row in enumerate(rows):
        targets[i, : row.numel()] = row

    return words, targets, lengths


def _logadd(a, b):
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


# -----------------------------
# 1) Vocab-wide min CTC loss (classification)
# -----------------------------

@torch.no_grad()
def decode_batch_ctc_vocab_minloss(logits_btn, logit_lengths_b, dataset, reduction="none"):
    """
    Scores each utterance against each vocabulary word using CTC loss.
    Returns: LongTensor (B,) = best vocab row index (NOT scr2id id).
            If you want scr2id ids, map words->id yourself.
    """

    _, vocab_targets, vocab_lengths = _vocab_targets(dataset, logits_btn.device)
    batch_best = []

    for b in range(logits_btn.size(0)):
        num_words = vocab_targets.size(0)
        logits = logits_btn[b : b + 1].expand(num_words, -1, -1).contiguous()
        input_lengths = logit_lengths_b[b : b + 1].expand(num_words).contiguous()

        losses = ctc_loss_from_logits(
            logits_btn=logits,
            targets_bk=vocab_targets,
            input_lengths_b=input_lengths,
            target_lengths_b=vocab_lengths,
            blank_id=getattr(dataset, "ctc_blank_id", 0),
            target_pad_id=None,
            reduction="none",
            zero_infinity=False,
        )
        batch_best.append(torch.argmin(losses))

    return torch.stack(batch_best).long()


# -----------------------------
# 2) Greedy CTC decode (sequence)
# -----------------------------

@torch.no_grad()
def decode_batch_ctc_greedy(logits_btn, logit_lengths_b, dataset, blank_id=0):
    """
    Greedy argmax per frame, then CTC collapse repeats and remove blanks.
    Returns: list[str] length B (decoded letter sequences as strings).
    """

    pred_ids = logits_btn.argmax(dim=-1)
    out = []

    for b in range(logits_btn.size(0)):
        length = int(logit_lengths_b[b].item())
        collapsed = []
        prev = None
        for idx in pred_ids[b, :length].tolist():
            if idx != prev:
                if idx != blank_id:
                    collapsed.append(idx)
                prev = idx
            else:
                prev = idx
        out.append(_ids_to_string(collapsed, dataset, blank_id=blank_id))

    return out


# -----------------------------
# 3) Beam search CTC decode (sequence)
# -----------------------------

@torch.no_grad()
def decode_batch_ctc_beam(logits_btn, logit_lengths_b, dataset, beam=4, blank_id=0):
    """
    Simple prefix beam search CTC (no LM).
    Returns: list[str] length B.
    """

    log_probs = F.log_softmax(logits_btn, dim=-1)
    results = []

    for b in range(logits_btn.size(0)):
        length = int(logit_lengths_b[b].item())
        beams = {(): (0.0, float("-inf"))}  # prefix -> (log p_blank, log p_nonblank)

        for t in range(length):
            next_beams = {}
            frame = log_probs[b, t]

            for prefix, (p_blank, p_nonblank) in beams.items():
                for token_id in range(frame.numel()):
                    token_logp = float(frame[token_id].item())
                    old_blank, old_nonblank = next_beams.get(
                        prefix, (float("-inf"), float("-inf"))
                    )

                    if token_id == blank_id:
                        old_blank = _logadd(old_blank, p_blank + token_logp)
                        old_blank = _logadd(old_blank, p_nonblank + token_logp)
                        next_beams[prefix] = (old_blank, old_nonblank)
                        continue

                    last = prefix[-1] if prefix else None
                    new_prefix = prefix + (token_id,)

                    if token_id == last:
                        old_nonblank = _logadd(old_nonblank, p_nonblank + token_logp)
                        next_beams[prefix] = (old_blank, old_nonblank)

                        ext_blank, ext_nonblank = next_beams.get(
                            new_prefix, (float("-inf"), float("-inf"))
                        )
                        ext_nonblank = _logadd(ext_nonblank, p_blank + token_logp)
                        next_beams[new_prefix] = (ext_blank, ext_nonblank)
                    else:
                        ext_blank, ext_nonblank = next_beams.get(
                            new_prefix, (float("-inf"), float("-inf"))
                        )
                        ext_nonblank = _logadd(ext_nonblank, p_blank + token_logp)
                        ext_nonblank = _logadd(ext_nonblank, p_nonblank + token_logp)
                        next_beams[new_prefix] = (ext_blank, ext_nonblank)

            beams = dict(
                sorted(
                    next_beams.items(),
                    key=lambda item: _logadd(item[1][0], item[1][1]),
                    reverse=True,
                )[:beam]
            )

        best_prefix = max(beams.items(), key=lambda item: _logadd(item[1][0], item[1][1]))[0]
        results.append(_ids_to_string(best_prefix, dataset, blank_id=blank_id))

    return results
