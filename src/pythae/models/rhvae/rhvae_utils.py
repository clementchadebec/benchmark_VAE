import torch


def create_metric(model):
    def G(z):
        return torch.inverse(
            (
                model.M_tens.unsqueeze(0)
                * torch.exp(
                    -torch.norm(
                        model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (model.temperature ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1)
            + model.lbd * torch.eye(model.latent_dim).to(z.device)
        )

    return G


def create_inverse_metric(model):
    def G_inv(z):
        return (
            model.M_tens.unsqueeze(0)
            * torch.exp(
                -torch.norm(model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1)
                ** 2
                / (model.temperature ** 2)
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        ).sum(dim=1) + model.lbd * torch.eye(model.latent_dim).to(z.device)

    return G_inv
