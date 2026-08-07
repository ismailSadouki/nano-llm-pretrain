import torch

def get_device_and_dtype(requested_dtype:str = "auto"):
    """
    Returns
    -------
    device : torch.device
    amp_dtype : torch.dtype | None
    use_scaler : bool
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # CPU
    if device.type == "cpu":
        return device, None, False

    # Auto selection
    if requested_dtype == "auto":
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            use_scaler = False
        else:
            amp_dtype = torch.float16
            use_scaler = True
    elif requested_dtype == "bf16" or requested_dtype == "bfloat16":
        amp_dtype = torch.bfloat16
        use_scaler = False
    elif requested_dtype == "fp16" or requested_dtype == "float16":
        amp_dtype = torch.float16
        use_scaler = True
    elif requested_dtype == "fp32" or requested_dtype == "float32":
        amp_dtype = None
        use_scaler = False
    else:
        raise ValueError(f"Unknown dtype: {requested_dtype}")

    return device, amp_dtype, use_scaler