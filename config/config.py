class Config:
    lr = 3e-4
    batch_size = 32
    epochs = 5

    embed_dim = 256
    num_heads = 8
    num_layers = 4
    ff_dim = 512
    max_len = 128

    num_classes = 2

    device = "cuda"