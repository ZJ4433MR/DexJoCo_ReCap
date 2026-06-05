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

The value-model pilot also uses a tiny BERT checkpoint as the language
backbone. Some tiny Hugging Face test checkpoints include task-head weights
that are reported as `unexpected_keys` when loaded through `AutoModel`. Evo-RL's
strict loading check treats that as a failure even when the backbone has no
missing or mismatched weights. For the simulation pilot, apply this patch to
allow unexpected-only keys for the `AutoModel` language backbone while still
failing on missing or mismatched weights:

```text
patches/evorl-pistar06-allow-automodel-heads.patch
```

The current local Evo-RL source tree has already been patched with both source
patches.
