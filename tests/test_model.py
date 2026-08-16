import torch

from bernet import Bernstein, Direction, Head
from bernet.functions import bernstein_basis


class TestInit:
    def test_creates_one_network_per_head(self):
        model = Bernstein([Head(3, 4), Head(5, 2)])
        assert len(model.networks) == 2
        assert model.networks[0](torch.randn(2, 3)).shape == (2, 4)
        assert model.networks[1](torch.randn(2, 5)).shape == (2, 2)


class TestSingleDirection:
    def test_forward_shape(self):
        torch.manual_seed(0)
        model = Bernstein([Head(3, 4)])
        batch = 6
        direction = Direction(latent=torch.randn(batch, 3), xi=torch.rand(batch))
        phi = model([direction])
        assert phi.shape == (batch,)

    def test_matches_direct_bernstein_contraction(self):
        torch.manual_seed(0)
        model = Bernstein([Head(3, 4)])
        batch = 6
        latent = torch.randn(batch, 3)
        xi = torch.rand(batch)
        direction = Direction(latent=latent, xi=xi)

        with torch.no_grad():
            phi = model([direction])
            p = model.networks[0](latent)
            w = bernstein_basis(xi, p.shape[-1] - 1)
            expected = (p * w).sum(dim=-1)

        assert torch.allclose(phi, expected, atol=1e-6)

    def test_dirichlet_boundary_is_exact(self):
        torch.manual_seed(0)
        model = Bernstein([Head(3, 3)])
        batch = 10
        latent = torch.randn(1, 3).expand(batch, -1)
        xi = torch.linspace(0, 1, batch)
        direction = Direction(
            latent=latent, xi=xi,
            phi_0=torch.zeros(batch), phi_1=torch.ones(batch),
        )

        with torch.no_grad():
            phi = model([direction])

        assert torch.isclose(phi[0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(phi[-1], torch.tensor(1.0), atol=1e-6)

    def test_neumann_derivative_matches_prescribed(self):
        torch.manual_seed(0)
        model = Bernstein([Head(3, 4)])
        batch = 5
        latent = torch.randn(1, 3).expand(batch, -1)
        xi = torch.zeros(batch, requires_grad=True)
        derivative = torch.full((batch,), 2.5)
        direction = Direction(latent=latent, xi=xi, dphi_dxi_0=derivative)

        phi = model([direction])
        (grad,) = torch.autograd.grad(phi.sum(), xi)

        assert torch.allclose(grad, derivative, atol=1e-4)


class TestTwoDirections:
    def test_free_directions_factorize(self):
        torch.manual_seed(0)
        model = Bernstein([Head(3, 4), Head(3, 5)])
        batch = 6
        latent1 = torch.randn(batch, 3)
        latent2 = torch.randn(batch, 3)
        xi1 = torch.rand(batch)
        xi2 = torch.rand(batch)
        d1 = Direction(latent=latent1, xi=xi1)
        d2 = Direction(latent=latent2, xi=xi2)

        with torch.no_grad():
            phi = model([d1, d2])
            p1 = model.networks[0](latent1)
            p2 = model.networks[1](latent2)
            w1 = bernstein_basis(xi1, p1.shape[-1] - 1)
            w2 = bernstein_basis(xi2, p2.shape[-1] - 1)
            expected = (p1 * w1).sum(dim=-1) * (p2 * w2).sum(dim=-1)

        assert torch.allclose(phi, expected, atol=1e-5)

    def test_single_boundary_does_not_leak_into_free_direction(self):
        # Regression test: a boundary condition on one direction must hold
        # exactly along its whole edge, regardless of the other (fully
        # free) direction's coordinate.
        torch.manual_seed(0)
        model = Bernstein([Head(3, 3), Head(3, 5)])
        n = 12
        axis = torch.linspace(0, 1, n)
        xi1_grid, xi2_grid = torch.meshgrid(axis, axis, indexing="ij")
        xi1, xi2 = xi1_grid.reshape(-1), xi2_grid.reshape(-1)
        batch = xi1.shape[0]
        latent1 = torch.randn(1, 3).expand(batch, -1)
        latent2 = torch.randn(1, 3).expand(batch, -1)
        d1 = Direction(latent=latent1, xi=xi1, phi_1=torch.ones(batch))
        d2 = Direction(latent=latent2, xi=xi2)

        with torch.no_grad():
            phi = model([d1, d2]).reshape(n, n)

        edge = phi[-1, :]  # xi1 = 1
        assert torch.allclose(edge, torch.ones(n), atol=1e-5)

    def test_shared_corner_uses_last_listed_direction(self):
        # Known limitation: when two directions each fix the same shared
        # corner with incompatible values, the direction listed LAST in
        # `forward` wins there (see `Bernstein`'s docstring).
        torch.manual_seed(0)
        model = Bernstein([Head(3, 3), Head(3, 3)])
        n = 5
        axis = torch.linspace(0, 1, n)
        xi1_grid, xi2_grid = torch.meshgrid(axis, axis, indexing="ij")
        xi1, xi2 = xi1_grid.reshape(-1), xi2_grid.reshape(-1)
        batch = xi1.shape[0]
        latent1 = torch.randn(1, 3).expand(batch, -1)
        latent2 = torch.randn(1, 3).expand(batch, -1)
        d1 = Direction(latent=latent1, xi=xi1, phi_0=torch.full((batch,), 1.0))
        d2 = Direction(latent=latent2, xi=xi2, phi_0=torch.full((batch,), 5.0))

        with torch.no_grad():
            phi = model([d1, d2]).reshape(n, n)

        assert torch.isclose(phi[0, 0], torch.tensor(5.0), atol=1e-6)

    def test_compatible_shared_corner_has_no_ambiguity(self):
        torch.manual_seed(0)
        model = Bernstein([Head(3, 3), Head(3, 3)])
        n = 5
        axis = torch.linspace(0, 1, n)
        xi1_grid, xi2_grid = torch.meshgrid(axis, axis, indexing="ij")
        xi1, xi2 = xi1_grid.reshape(-1), xi2_grid.reshape(-1)
        batch = xi1.shape[0]
        latent1 = torch.randn(1, 3).expand(batch, -1)
        latent2 = torch.randn(1, 3).expand(batch, -1)
        d1 = Direction(latent=latent1, xi=xi1, phi_0=torch.full((batch,), 3.0))
        d2 = Direction(latent=latent2, xi=xi2, phi_0=torch.full((batch,), 3.0))

        with torch.no_grad():
            phi = model([d1, d2]).reshape(n, n)

        assert torch.isclose(phi[0, 0], torch.tensor(3.0), atol=1e-6)


class TestGradients:
    def test_gradients_flow_to_head_parameters(self):
        torch.manual_seed(0)
        model = Bernstein([Head(3, 4)])
        batch = 6
        direction = Direction(latent=torch.randn(batch, 3), xi=torch.rand(batch))

        phi = model([direction])
        phi.sum().backward()

        params = list(model.networks[0].parameters())
        assert len(params) > 0
        assert all(p.grad is not None for p in params)
