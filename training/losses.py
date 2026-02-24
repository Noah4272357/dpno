import torch


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


def rel_loss(x, y, p, reduction=True, size_average=False, time_average=False):
    # x, y: [b, c, t, h, w] or [b, c, t, n]
    batch_num = x.shape[0]
    frame_num = x.shape[2]

    if len(x.shape) == 5:
        h = x.shape[3]
        w = x.shape[4]
        n = h * w
    else:
        n = x.shape[-1]
    # x = rearrange(x, 'b c t h w -> (b t h w) c')
    # y = rearrange(y, 'b c t h w -> (b t h w) c')
    num_examples = x.shape[0]
    eps = 1e-6
    diff_norms = torch.norm(
        x.reshape(num_examples, -1) - y.reshape(num_examples, -1), p, 1
    )
    y_norms = torch.norm(y.reshape(num_examples, -1), p, 1) + eps

    loss = torch.sum(diff_norms / y_norms)
    if reduction:
        loss = loss / batch_num
        if size_average:
            loss /= n
        if time_average:
            loss /= frame_num

    return loss


def rel_l2norm_loss(x, y):
    #   x, y [b, c, t, n]
    eps = 1e-6
    y_norm = (y**2).mean(dim=-1) + eps
    diff = ((x - y) ** 2).mean(dim=-1)
    diff = diff / y_norm  # [b, c, t]
    diff = diff.sqrt().mean()
    return diff


def pointwise_rel_l2norm_loss(x, y):
    #   x, y [b, n, c]
    eps = 1e-6
    y_norm = (y**2).mean(dim=-2) + eps
    diff = ((x - y) ** 2).mean(dim=-2)
    diff = diff / y_norm  # [b, c]
    diff = diff.sqrt().mean()
    return diff
