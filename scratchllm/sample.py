"""Text generation with temperature and top-p sampling."""

import torch


@torch.no_grad()
def generate(
    model,
    ids: list[int],
    max_new_tokens: int,
    *,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    stop_id: int | None = None,
) -> list[int]:
    """Return only the new tokens. Stops before stop_id, which is not returned."""
    if not ids:
        raise ValueError("generate needs at least one starting token")
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    ids, new = list(ids), []
    try:
        for _ in range(max_new_tokens):
            window = torch.tensor([ids[-model.cfg.context :]], device=device)
            logits = model(window)[0][0, -1].float().cpu()
            if temperature == 0:
                token = int(logits.argmax())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                sorted_probs, order = probs.sort(descending=True)
                # Keep the smallest set of tokens whose probability reaches top_p.
                keep = sorted_probs.cumsum(0) - sorted_probs < top_p
                kept = sorted_probs * keep
                token = int(order[torch.multinomial(kept / kept.sum(), 1, generator=gen)])
            if token == stop_id:
                break
            ids.append(token)
            new.append(token)
    finally:
        model.train(was_training)
    return new
