# Pure CSS MNIST

An MNIST demo that runs entirely in CSS.

<img width="392" height="650" alt="MNIST Demo" src="https://github.com/user-attachments/assets/9315fa87-6ef3-4c2e-8f44-0ece8d4ecb93" />

### Architecture

```
Input (28×28 binary) → Conv2d(1→4, 5×5, stride=4) → LeakyReLU(0.01) → Linear(144→10) → Softmax
```

## Usage

### Train the model

```bash
uv run main.py
```

This trains the network and generates `model.css`.

### Build the frontend

```bash
bun run index.tsx
```

Outputs static HTML to `dist/`.

## How Drawing Works

Drawing uses a clever CSS-only trick:

```css
:root { transition: --in-X 1s 999999s; }
:root:has(.board>.cell:active):has(.board>.cell-X:hover) { 
  --in-X: 1; 
  transition: --in-X 0s; 
}
```

When a cell is hovered while any cell is active, the input variable is set to 1 immediately and persists for ~11.5 days.
