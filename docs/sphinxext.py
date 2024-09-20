
def reset_torch(gallery_conf, fname):
    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()

