# Third-party attribution

The following adaptations are based on DSCNet by Yaolei Qi: modules/dsconv_fix.py,
modules/dscnet_plus.py, modules/unet_same_skeleton.py, train_clean.py,
batch_inference.py, performance_cache.py and experiments/e1_whole_network/e1_bug.py.
They retain the upstream notice in their headers. The upstream baseline is imported
at runtime from a separately downloaded checkout, not redistributed here.
DSCNet source and data provenance: https://github.com/YaoleiQi/DSCNet/tree/0e35b2b6ea748a5e8ce7557ea3e580b9951471d6
Upstream license: https://github.com/YaoleiQi/DSCNet/blob/0e35b2b6ea748a5e8ce7557ea3e580b9951471d6/LICENSE

## DSCNet MIT notice (retained in full)

MIT License

Copyright (c) 2025 Yaolei Qi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

SGL source/data distribution: https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation
Mamba2 implementation: https://github.com/state-spaces/mamba (Apache-2.0).
causal-conv1d: https://github.com/Dao-AILab/causal-conv1d (BSD-3-Clause).
These external packages and audit targets are credited to their respective authors;
their complete sources, weights and datasets are not part of this distribution.

The whole-network E1 script/results were supplied by Claude in the project;
the synthetic E1 characterization and code audit were prepared by Codex.
Publication integration, adaptations and repository copyright: Heng Zhang.
