from tokenizers import Tokenizer
import numpy as np

def load_tokenizer(path):
    return Tokenizer.from_file(path)


def pack_tokens(tokens, block_size, pad_id):
    """
    Pack tokens into blocks of size `block_size`.
    """
    input_ids = []
    labels = []
    loss_masks = []

    for i in range(0, len(tokens), block_size):
        block = tokens[i:i + block_size]

        # last block
        pad = block_size - len(block)

        if pad > 0:
            block = block + [pad_id] * pad

        label = block[1:] + [pad_id]

        mask = [1] * (block_size - pad) + [0] * pad 

        input_ids.append(block)
        labels.append(label)
        loss_masks.append(mask)

    return (
            np.array(input_ids, dtype=np.int32),
            np.array(labels, dtype=np.int32),
            np.array(loss_masks, dtype=np.uint8),
        )


