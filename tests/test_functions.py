import math

import pytest
import torch

from bernet.functions import (
    bernstein_basis,
    create_sequential_network,
    insert,
    outer_product,
    select,
)


class TestBernsteinBasis:
    def test_shape(self):
        xi = torch.rand(7)
        w = bernstein_basis(xi, n=4)
        assert w.shape == (7, 5)

    def test_partition_of_unity(self):
        xi = torch.rand(50)
        w = bernstein_basis(xi, n=6)
        assert torch.allclose(w.sum(dim=-1), torch.ones(50), atol=1e-6)

    def test_boundary_values(self):
        xi = torch.tensor([0.0, 1.0])
        w = bernstein_basis(xi, n=3)
        assert torch.allclose(w[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        assert torch.allclose(w[1], torch.tensor([0.0, 0.0, 0.0, 1.0]))

    def test_matches_closed_form_degree_2(self):
        xi = torch.linspace(0, 1, 11)
        w = bernstein_basis(xi, n=2)
        expected = torch.stack([(1 - xi) ** 2, 2 * xi * (1 - xi), xi**2], dim=-1)
        assert torch.allclose(w, expected, atol=1e-6)


class TestOuterProduct:
    def test_single_tensor_passthrough(self):
        t = torch.randn(4, 3)
        assert torch.equal(outer_product([t]), t)

    def test_two_tensors_matches_einsum(self):
        batch = 5
        a = torch.randn(batch, 3)
        b = torch.randn(batch, 4)
        result = outer_product([a, b])
        expected = torch.einsum("bi,bj->bij", a, b)
        assert torch.allclose(result, expected)

    def test_three_tensors_matches_einsum(self):
        batch = 5
        a = torch.randn(batch, 3)
        b = torch.randn(batch, 4)
        c = torch.randn(batch, 2)
        result = outer_product([a, b, c])
        expected = torch.einsum("bi,bj,bk->bijk", a, b, c)
        assert torch.allclose(result, expected)

    def test_preserves_batch_dimension(self):
        # Regression test: batch size must not collide with a feature size
        # via naive broadcasting.
        batch = 3
        a = torch.randn(batch, 6)
        b = torch.randn(batch, 6)
        result = outer_product([a, b])
        assert result.shape == (batch, 6, 6)


class TestInsert:
    def test_grows_dimension_by_one(self):
        t = torch.randn(2, 5)
        out = insert(t, dim=1, index=2)
        assert out.shape == (2, 6)

    def test_inserted_slice_is_ones(self):
        t = torch.randn(2, 5)
        out = insert(t, dim=1, index=2)
        assert torch.equal(out[:, 2], torch.ones(2))

    def test_preserves_surrounding_values(self):
        t = torch.randn(2, 5)
        out = insert(t, dim=1, index=2)
        assert torch.equal(out[:, :2], t[:, :2])
        assert torch.equal(out[:, 3:], t[:, 2:])

    def test_insert_at_front_and_back(self):
        t = torch.arange(6).reshape(2, 3).float()
        front = insert(t, dim=1, index=0)
        assert torch.equal(front[:, 0], torch.ones(2))
        assert torch.equal(front[:, 1:], t)

        back = insert(t, dim=1, index=t.shape[1])
        assert torch.equal(back[:, -1], torch.ones(2))
        assert torch.equal(back[:, :-1], t)

    def test_negative_dim(self):
        t = torch.randn(2, 5)
        assert torch.equal(insert(t, dim=-1, index=1), insert(t, dim=1, index=1))


class TestSelect:
    def test_returns_correct_slice(self):
        t = torch.arange(24).reshape(2, 3, 4).float()
        assert torch.equal(select(t, dim=1, index=1), t[:, 1, :])

    def test_returns_a_mutable_view(self):
        t = torch.ones(3, 4)
        select(t, dim=1, index=2).mul_(5.0)
        assert torch.equal(t[:, 2], torch.full((3,), 5.0))
        assert torch.equal(t[:, 0], torch.ones(3))

    def test_copy_writes_through(self):
        t = torch.zeros(3, 4)
        select(t, dim=1, index=1).copy_(torch.arange(3).float())
        assert torch.equal(t[:, 1], torch.arange(3).float())


class TestCreateSequentialNetwork:
    def test_no_hidden_layers(self):
        net = create_sequential_network(3, 5, hidden_dims=None, activation=None)
        assert len(net) == 1
        assert isinstance(net[0], torch.nn.Linear)
        assert net(torch.randn(2, 3)).shape == (2, 5)

    def test_with_hidden_layers(self):
        net = create_sequential_network(3, 5, hidden_dims=[8, 8], activation=None)
        assert net(torch.randn(2, 3)).shape == (2, 5)

    def test_default_activation_is_tanh(self):
        net = create_sequential_network(3, 5, hidden_dims=[8], activation=None)
        activations = [m for m in net if isinstance(m, torch.nn.Tanh)]
        assert len(activations) >= 1

    def test_custom_activation(self):
        net = create_sequential_network(3, 5, hidden_dims=[8], activation=torch.nn.ReLU)
        assert any(isinstance(m, torch.nn.ReLU) for m in net)
