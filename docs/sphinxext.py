def reset_torch(gallery_conf, fname):
    """Function to clean up torch between sphinx examples"""
    import gc

    import torch

    torch.cuda.empty_cache()
    gc.collect()
