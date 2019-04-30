"""Module defining gans."""
from onmt.gans.gan import MLP_D, MLP_G

str2gan = {"gan_g": MLP_G, "gan_d": MLP_D}

__all__ = ["MLP_G", "MLP_D", "str2gan"]