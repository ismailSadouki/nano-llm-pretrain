
import torch
import torch.nn.functional as F

from models.kv_cache import KVCache




@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    seed=None,
    eos_token_id=None,
):
    """
    input_ids : [B, T]
    returns   : [B, T + max_new_tokens]
    """
    assert not (
        top_k is not None and top_p is not None
    ), "Choose either top_k or top_p, not both."
    assert temperature >= 0, "temperature must be non-negative"
    

    if seed is not None:
        torch.manual_seed(seed)
    model.eval()


    B = input_ids.size(0)

    device = input_ids.device
    dtype = model.tok_embeddings.weight.dtype

    caches = [ # len(caches) = number of decoder blocks
        KVCache(
            batch_size=B,
            max_seq_len=model.config.block_size,
            n_kv_heads=model.config.n_kv_heads,
            head_dim=model.config.d_model // model.config.n_heads,
            device=device,
            dtype=dtype,
        ) 
        for _ in range(model.config.n_layers)
    ]


    logits = model( # [B, S, V]
        input_ids,
        caches=caches,
        start_pos=0
    )

    generated = input_ids.clone() # [B, T]

    for step in range(max_new_tokens):
        logits = logits[:, -1, :] # [B, V]

    
        if temperature == 0:
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True
                ) # [B, 1]
        else:
            logits = logits / temperature
             
      

            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, top_k)

                logits = logits.masked_fill(
                    logits < v[:, [-1]],
                    float("-inf")
                )
                probs = F.softmax(logits, dim=-1)

                next_token = torch.multinomial(
                     probs,
                     num_samples = 1
                )
            elif top_p is not None:
                probs = F.softmax(logits, dim=-1) # [B, V]
                sorted_probs, sorted_indices = torch.sort(
                    probs,
                    descending=True,
                )

                cumulative_probs = torch.cumsum(
                     sorted_probs,
                     dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = \
                    sorted_indices_to_remove[..., :-1].clone()

                sorted_indices_to_remove[..., 0] = False


                indices_to_remove = torch.zeros_like(
                    sorted_indices_to_remove
                )

                indices_to_remove.scatter_(
                    dim=-1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove,
                )

                logits = logits.masked_fill(
                    indices_to_remove,
                    float("-inf"),
                )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probs,
                    num_samples=1,
                )
            else: # temp = 1 and no top_k or top_p
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                        probs,
                        num_samples=1,
                    )

                


        generated = torch.cat(
            [generated, next_token],
            dim=1
        ) # [B, T + 1]

        if generated.size(1) >= model.config.block_size:
            break

        # stop if eos_token_id is found
        if eos_token_id is not None:
            if (next_token == eos_token_id).all():
                break

        logits = model(
            next_token,
            caches=caches,
            start_pos=input_ids.size(1) + step
        )

    return generated
        
