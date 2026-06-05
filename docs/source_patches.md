# Source Patches

The remote runner archives Evo-RL from the local source tree:

```text
E:\Evo-RL-main\Evo-RL-main
```

During the PuSH-T pilot on PyTorch 2.2.1, checkpoint saving failed because
`lerobot.utils.random_utils.serialize_numpy_rng_state()` passed NumPy's
`uint32` MT19937 state directly to `torch.tensor(..., dtype=torch.int64)`.
PyTorch does not implicitly convert this dtype.

Apply this patch to the Evo-RL source tree before running checkpointed jobs:

```text
patches/evorl-rng-uint32.patch
```

The current local Evo-RL source tree has already been patched.
