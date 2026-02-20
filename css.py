from tinygrad import Tensor


def fmt(x: float) -> str:
    return f'{x:.10f}'.rstrip('0').rstrip('.')


def prop(name: str) -> str:
    return f'@property --{name} {{ syntax: "<number>"; inherits: true; initial-value: 0; }}'


def var(name: str) -> str:
    return f'var(--{name})'


def generate(state_dict: dict[str, Tensor]) -> str:
    l1 = state_dict['l1.weight'].numpy()  # (4, 1, 5, 5)
    l2 = state_dict['l2.weight'].numpy()  # (10, 4x6x6)
    css: list[str] = []

    c_out, c_in, kh, kw = l1.shape
    stride = 4
    in_h, in_w = 28, 28
    out_h = (in_h - kh) // stride + 1  # 6
    out_w = (in_w - kw) // stride + 1  # 6

    # input: 28x28 binary pixels
    css += [prop(f'in-{i}') for i in range(784)]

    # conv(1->4, 5x5, stride=4, no pad) + leaky_relu(0.01): 28x28 -> 4x6x6
    for co in range(c_out):
        for oy in range(out_h):
            for ox in range(out_w):
                terms = []
                for ci in range(c_in):
                    for ky in range(kh):
                        for kx in range(kw):
                            iy, ix = oy * stride + ky, ox * stride + kx
                            src = f'in-{iy * in_w + ix}'
                            terms.append(f'{var(src)} * {fmt(l1[co, ci, ky, kx])}')
                conv = f'calc({" + ".join(terms)})'
                name = f'c-{co}-{oy * out_w + ox}'
                css += [prop(name), f':root {{ --{name}: max(calc(0.01 * {conv}), {conv}); }}']

    # flatten + linear(4x6x6->10, no bias): 4x6x6 -> 10 logits
    n_out, n_in = l2.shape
    for i in range(n_out):
        terms = []
        for j in range(n_in):
            co, pos = divmod(j, out_h * out_w)
            src = f'c-{co}-{pos}'
            terms.append(f'{var(src)} * {fmt(l2[i, j])}')
        css += [prop(f'logit-{i}'), f':root {{ --logit-{i}: calc({" + ".join(terms)}); }}']

    # softmax
    css += [prop('logit-max'), f':root {{ --logit-max: max({", ".join(var(f"logit-{i}") for i in range(n_out))}); }}']
    for i in range(n_out):
        css += [prop(f'exp-{i}'), f':root {{ --exp-{i}: exp(calc({var(f"logit-{i}")} - {var("logit-max")})); }}']
    css += [prop('exp-sum'), f':root {{ --exp-sum: calc({" + ".join(var(f"exp-{i}") for i in range(n_out))}); }}']
    for i in range(n_out):
        css += [prop(f'prob-{i}'), f':root {{ --prob-{i}: calc({var(f"exp-{i}")} / {var("exp-sum")}); }}']

    # https://github.com/oven-sh/bun/issues/27117
    css.sort(key=lambda x: not x.startswith('@'))

    return '\n'.join(css)
