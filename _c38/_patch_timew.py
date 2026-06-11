f = "models/diffusers/multi_asset/ma_gaussian_diffusion.py"
s = open(f).read()
head = s.split("class ")[0]
if "import os" not in head:
    s = s.replace("import numpy as np\nimport torch",
                  "import os\nimport numpy as np\nimport torch", 1)
old = (
    '    def _mse_loss_per_asset(self, noise_t, noise_true):\n'
    '        """L2 norm of noise residual per (batch, asset). Returns (B, N)."""\n'
    '        diff = noise_t - noise_true  # (B, N, K, F)\n'
    '        return torch.norm(diff, p=2, dim=[2, 3])'
)
new = (
    '    def _mse_loss_per_asset(self, noise_t, noise_true):\n'
    '        """L2 norm of noise residual per (batch, asset). Returns (B, N).\n'
    '\n'
    '        TIME_LOSS_W>1 upweights the time channel (idx 0) of the epsilon error\n'
    '        to attack the inter_arrival weakness (NEXT_WORK_ZH.md task 2.2).\n'
    '        """\n'
    '        diff = noise_t - noise_true  # (B, N, K, F)\n'
    '        tw = float(os.environ.get("TIME_LOSS_W", "1.0"))\n'
    '        if tw != 1.0:\n'
    '            w = torch.ones(diff.shape[-1], device=diff.device, dtype=diff.dtype)\n'
    '            w[0] = tw ** 0.5  # squared inside L2 norm -> effective weight tw\n'
    '            diff = diff * w\n'
    '        return torch.norm(diff, p=2, dim=[2, 3])'
)
assert old in s, "anchor not found"
s = s.replace(old, new)
open(f, "w").write(s)
print("patched; import os in head:", "import os" in s.split("class ")[0])
