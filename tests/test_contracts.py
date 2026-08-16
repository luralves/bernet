import dataclasses

import pytest
import torch

from bernet.contracts import Direction, Head


class TestDirection:
    def test_required_fields(self):
        latent = torch.randn(4, 3)
        xi = torch.rand(4)
        direction = Direction(latent=latent, xi=xi)
        assert torch.equal(direction.latent, latent)
        assert torch.equal(direction.xi, xi)

    def test_optional_fields_default_to_none(self):
        direction = Direction(latent=torch.randn(4, 3), xi=torch.rand(4))
        assert direction.phi_0 is None
        assert direction.phi_1 is None
        assert direction.dphi_dxi_0 is None
        assert direction.dphi_dxi_1 is None

    def test_is_frozen(self):
        direction = Direction(latent=torch.randn(4, 3), xi=torch.rand(4))
        with pytest.raises(dataclasses.FrozenInstanceError):
            direction.xi = torch.rand(4)


class TestHead:
    def test_required_fields(self):
        head = Head(input_dim=3, output_dim=5)
        assert head.input_dim == 3
        assert head.output_dim == 5

    def test_optional_fields_default_to_none(self):
        head = Head(input_dim=3, output_dim=5)
        assert head.hidden_dims is None
        assert head.activation is None

    def test_is_frozen(self):
        head = Head(input_dim=3, output_dim=5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            head.output_dim = 10
