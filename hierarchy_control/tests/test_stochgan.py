"""StochGAN in the simulator: the JAX generator matches the PyTorch layer definitions,
and the c4g / ceg stages feed its output through rolling shutter and erosion."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skydreamer import gan

from ctba import env as E, tasks

SHAPES = ([(32, 2, 7, 7), (64, 32, 3, 3), (128, 64, 3, 3)] + [(128, 128, 3, 3)] * 12
          + [(128, 64, 3, 3), (64, 32, 3, 3), (1, 32, 7, 7)])


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    rng = np.random.default_rng(0)
    arrays = {}
    for i, s in enumerate(SHAPES):
        fan_in = s[1] * s[2] * s[3] if i not in gan.UPSAMPLE else s[0] * s[2] * s[3]
        arrays[f"conv{i}.w"] = (rng.standard_normal(s) / np.sqrt(fan_in)).astype(np.float32)
        out = s[1] if i in gan.UPSAMPLE else s[0]
        arrays[f"conv{i}.b"] = (0.01 * rng.standard_normal(out)).astype(np.float32)
    path = tmp_path_factory.mktemp("gan") / "G.npz"
    np.savez(path, **arrays)
    return path


def _conv_transpose_ref(x, w, b, stride=2, pad=1, out_pad=1):
    """torch.nn.ConvTranspose2d by definition: scatter every input pixel times the kernel."""
    cin, h, wd = x.shape
    _, cout, k, _ = w.shape
    full = np.zeros((cout, (h - 1) * stride + k, (wd - 1) * stride + k))
    for i in range(h):
        for j in range(wd):
            full[:, i * stride:i * stride + k, j * stride:j * stride + k] += np.einsum("c,cokl->okl", x[:, i, j], w)
    ho, wo = (h - 1) * stride - 2 * pad + k + out_pad, (wd - 1) * stride - 2 * pad + k + out_pad
    full = np.pad(full, ((0, 0), (0, out_pad), (0, out_pad)))
    return full[:, pad:pad + ho, pad:pad + wo] + b[:, None, None]


def test_transposed_conv_is_pytorchs():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((4, 5, 6)).astype(np.float32)
    w = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    b = rng.standard_normal(3).astype(np.float32)
    got = np.asarray(gan._conv_t(jnp.asarray(x), jnp.asarray(w), jnp.asarray(b)))
    want = _conv_transpose_ref(x, w, b)
    assert got.shape == (3, 10, 12) == want.shape
    np.testing.assert_allclose(got, want, atol=1e-4)


def test_generator_shapes_and_range(weights):
    p = gan.load(weights)
    y = gan.generator(p, jnp.zeros((2, 64, 64)) - 1.0)
    assert y.shape == (1, 64, 64)
    assert float(jnp.abs(y).max()) <= 1.0
    m = gan.translate(p, jnp.zeros((64, 64)).at[20:40, 20:40].set(1.0), jax.random.key(0))
    assert m.shape == (64, 64) and set(np.unique(np.asarray(m))) <= {0.0, 1.0}


def test_noise_channel_changes_the_output(weights):
    p = gan.load(weights)
    m = jnp.zeros((64, 64)).at[10:50, 10:50].set(1.0)
    a = gan.translate(p, m, jax.random.key(0), binary=False)
    b = gan.translate(p, m, jax.random.key(1), binary=False)
    assert float(jnp.abs(a - b).max()) > 1e-4


def test_stages_need_weights(monkeypatch):
    monkeypatch.delenv(gan.ENV_VAR, raising=False)
    with pytest.raises(RuntimeError):
        tasks.make_config("c4g_team44up")


def test_stochgan_stage_steps(weights, monkeypatch):
    monkeypatch.setenv(gan.ENV_VAR, str(weights))
    cfg = tasks.make_config("c4g_team44up")
    assert cfg.stochgan and not tasks.make_config("c4_team44up").stochgan
    reset_b, step_b = E.batched(cfg)
    st, obs = reset_b(jax.random.split(jax.random.key(0), 2))
    for _ in range(3):
        st, obs, *_ = step_b(st, jnp.zeros((2, 4)))
    m = np.asarray(st.mask_buf)
    assert m.shape[0] == 2 and np.isfinite(m).all() and m.min() >= 0.0 and m.max() <= 1.0
