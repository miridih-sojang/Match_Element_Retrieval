import torch

from transformers import EfficientNetPreTrainedModel
from transformers.models.efficientnet.modeling_efficientnet import EfficientNetEmbeddings, EfficientNetEncoder


class EfficientNetDualModel(EfficientNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = EfficientNetEmbeddings(self.config)
        self.encoder = EfficientNetEncoder(self.config)

        # Final pooling layer
        if config.pooling_type == "mean":
            self.pooler = torch.nn.AvgPool2d(config.hidden_dim, ceil_mode=True)
        elif config.pooling_type == "max":
            self.pooler = torch.nn.MaxPool2d(config.hidden_dim, ceil_mode=True)
        else:
            raise ValueError(f"config.pooling must be one of ['mean', 'max'] got {config.pooling}")

        self.vision_embed_dim = config.hidden_dim
        self.projection_dim = config.projection_dim

        self.visual_projection = torch.nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = torch.nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.post_init()

    def forward(self,
                element_images,
                background_images
                ):

        element_images_embedding = self.embeddings(element_images)
        background_images_embedding = self.embeddings(background_images)

        element_images_output = self.encoder(element_images_embedding)
        background_images_output = self.encoder(background_images_embedding)

        element_last_hidden_state = element_images_output[0]
        element_pooled_output = self.pooler(element_last_hidden_state)
        element_pooled_output = element_pooled_output.reshape(element_pooled_output.shape[:2])
        element_image_embeds = self.visual_projection(element_pooled_output)

        background_last_hidden_state = background_images_output[0]
        background_pooled_output = self.pooler(background_last_hidden_state)
        background_pooled_output = background_pooled_output.reshape(background_pooled_output.shape[:2])
        background_image_embeds = self.visual_projection(background_pooled_output)

        # normalized features
        elements_image_embeds = element_image_embeds / element_image_embeds.norm(p=2, dim=-1, keepdim=True)
        background_image_embeds = background_image_embeds / background_image_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_elements = torch.matmul(elements_image_embeds, background_image_embeds.t().to(
            elements_image_embeds.device)) * logit_scale.to(elements_image_embeds.device)
        # logits_per_removed = logits_per_elements.t()
        return logits_per_elements

    def inference_image(self, image_tensor):
        with torch.no_grad():
            embedding_output = self.embeddings(image_tensor.unsqueeze(0))
            vision_output = self.encoder(embedding_output)
            last_hidden_state = vision_output[0]
            pooled_output = self.pooler(last_hidden_state).reshape(last_hidden_state.shape[0], -1)
            image_embedding = self.visual_projection(pooled_output)
            image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
            return image_embedding

    def inference(self,
                  element_images_tensor,
                  background_images_tensor):

        element_images_embedding = self.inference_image(element_images_tensor)
        background_images_embedding = self.inference_image(background_images_tensor)
        similarity = self.get_similarity(element_images_embedding, background_images_embedding)
        return similarity

    @staticmethod
    def get_similarity(query_image, candidate_image):
        similarity = torch.matmul(query_image, candidate_image.t().to(query_image.device)).item()
        return similarity
