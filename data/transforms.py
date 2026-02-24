import torch


def get_transformer(transformer, X, **kwargs):
    if transformer == "unit":
        return UnitTransformer(X, **kwargs)
    elif transformer == "unit_gaussian":
        return UnitGaussianNormalizer(X, **kwargs)
    else:
        raise ValueError(f"Unknown transformer: {transformer}")


class UnitTransformer:
    def __init__(self, X):
        self.mean = X.mean(dim=(0, 1), keepdim=True)
        self.std = X.std(dim=(0, 1), keepdim=True) + 1e-8

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x * self.std + self.mean

    def transform(self, X, inverse=True, component="all"):
        if component == "all" or "all-reduce":
            if inverse:
                orig_shape = X.shape
                return (X * (self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X - self.mean) / self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (
                    X * (self.std[:, component] - 1e-8) + self.mean[:, component]
                ).view(orig_shape)
            else:
                return (X - self.mean[:, component]) / self.std[:, component]


class UnitGaussianNormalizer(object):
    def __init__(self, x, dims=None, eps=1e-6, **kwargs):
        super(UnitGaussianNormalizer, self).__init__()
        self.eps = eps
        self.dims = dims
        self.mean = torch.mean(x, dims, keepdim=True) if dims else torch.mean(x)
        self.std = torch.std(x, dims, keepdim=True) if dims else torch.std(x)

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
