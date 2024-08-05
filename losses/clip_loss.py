import torch


class ClipLoss(torch.nn.Module):
    def __init__(self):
        super(ClipLoss, self).__init__()
    @staticmethod
    def contrastive_loss(logits):
        return torch.nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def forward(self, similarity):
        caption_loss = self.contrastive_loss(similarity)
        image_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
