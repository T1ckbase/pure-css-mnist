import numpy as np
from tinygrad import Tensor


def fmt(x: float) -> str:
    return f'{x:.10f}'.rstrip('0').rstrip('.')


def prop(name: str) -> str:
    return f'@property --{name} {{ syntax: "<number>"; inherits: true; initial-value: 0; }}'


def var(name: str) -> str:
    return f'var(--{name})'


def conv_relu_pool(w: np.ndarray, in_h: int, in_w: int, in_pfx: str, out_pfx: str) -> list[str]:
    """Conv2d (no bias) + ReLU + MaxPool(2x2), fused into max(0, conv@each_pool_position)."""
    c_out, c_in, kh, kw = w.shape
    pad, out_h, out_w = kh // 2, in_h // 2, in_w // 2
    css = []
    for co in range(c_out):
        for py in range(out_h):
            for px in range(out_w):
                pool = ['0']
                for dy in range(2):
                    for dx in range(2):
                        y, x = py * 2 + dy, px * 2 + dx
                        terms = []
                        for ci in range(c_in):
                            for ky in range(kh):
                                for kx in range(kw):
                                    iy, ix = y - pad + ky, x - pad + kx
                                    if 0 <= iy < in_h and 0 <= ix < in_w:
                                        src = (
                                            f'{in_pfx}-{iy * in_w + ix}'
                                            if c_in == 1
                                            else f'{in_pfx}-{ci}-{iy * in_w + ix}'
                                        )
                                        terms.append(f'{var(src)} * {fmt(w[co, ci, ky, kx])}')
                        if terms:
                            pool.append(f'calc({" + ".join(terms)})')
                name = f'{out_pfx}-{co}-{py * out_w + px}'
                css += [prop(name), f':root {{ --{name}: max({", ".join(pool)}); }}']
    return css


def generate(state_dict: dict[str, Tensor]) -> str:
    l1, l2, l3 = (state_dict[f'l{i}.weight'].numpy() for i in (1, 2, 3))
    css: list[str] = []

    # input: 28x28 binary pixels
    css += [prop(f'in-{i}') for i in range(784)]

    # conv(1->6, 3x3, pad=1) + relu + maxpool(2x2): 28x28 -> 6x14x14
    css += conv_relu_pool(l1, 28, 28, 'in', 'p1')

    # conv(6->20, 3x3, pad=1) + relu + maxpool(2x2): 14x14 -> 20x7x7
    css += conv_relu_pool(l2, 14, 14, 'p1', 'p2')

    # global average pooling: 20x7x7 -> 20 (swapped before conv3 since 1x1 conv and mean commute)
    for ch in range(l3.shape[1]):
        terms = ' + '.join(var(f'p2-{ch}-{i}') for i in range(49))
        css += [prop(f'avg-{ch}'), f':root {{ --avg-{ch}: calc(({terms}) / 49); }}']

    # conv(20->10, 1x1): channel mixing -> 10 logits
    for ch in range(l3.shape[0]):
        terms = ' + '.join(f'{var(f"avg-{ci}")} * {fmt(l3[ch, ci, 0, 0])}' for ci in range(l3.shape[1]))
        css += [prop(f'logit-{ch}'), f':root {{ --logit-{ch}: calc({terms}); }}']

    # softmax
    for i in range(10):
        css += [prop(f'exp-{i}'), f':root {{ --exp-{i}: exp({var(f"logit-{i}")}); }}']
    css += [prop('exp-sum'), f':root {{ --exp-sum: calc({" + ".join(var(f"exp-{i}") for i in range(10))}); }}']
    for i in range(10):
        css += [prop(f'prob-{i}'), f':root {{ --prob-{i}: calc({var(f"exp-{i}")} / {var("exp-sum")}); }}']

    return '\n'.join(css)
