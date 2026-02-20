import { rm } from 'node:fs/promises';
import type { FC } from 'hono/jsx';

const PRODUCTION = Bun.env.NODE_ENV === 'production';

const H = 28;
const W = 28;

const Board: FC<{ width: number; height: number }> = ({ width, height }) => {
  const cells = Array.from({ length: width * height }, (_, i) => <div class={`cell cell-${i}`}></div>);
  return (
    <div class='board' style={{ width: `calc(var(--cell-size) * ${width})` }}>
      {cells}
    </div>
  );
};

function generateBoardCSS(cells: number): string {
  // const inf = '@property --inf { syntax: "<time>"; inherits: true; initial-value: 999999s; }';
  const transitions: string[] = [];
  const boardCells: string[] = [];
  const activedCells: string[] = [];
  for (let i = 0; i < cells; i++) {
    transitions.push(`--in-${i} 1s 999999s`);
    boardCells.push(`.board > .cell-${i} { background-color: hsl(0 0% calc(var(--in-${i}) * 100%)); }`);
    activedCells.push(
      `:root:has(.board > .cell:active):has(.board > .cell-${i}:hover) { --in-${i}: 1; transition: ${Array.from({ length: cells }, (_, j) => (j === i ? `--in-${j} 0s` : `--in-${j} 1s 999999s`)).join(',')} }`,
    );
  }
  return [`:root { transition: ${transitions.join(',')}; }`, boardCells.join('\n'), activedCells.join('\n')].join('\n');
}

const jsxElement = (
  <html lang='en'>
    <head>
      <meta charset='UTF-8' />
      <meta name='viewport' content='width=device-width, initial-scale=1.0' />
      <link rel='icon' href='data:,' />
      <meta name='color-scheme' content='dark' />
      <meta name='description' content='Pure CSS implementation of a CNN for MNIST digit recognition' />
      <meta
        name='keywords'
        content='CSS, machine learning, handwritten digit recognition, neural network, front-end AI, web development, MNIST, CNN'
      />
      <title>Pure CSS MNIST</title>
      <link rel='stylesheet' href='./model.css' />
      <link rel='stylesheet' href='./board.css' />
      <link rel='stylesheet' href='./main.css' />
    </head>
    <body>
      <header>
        <h1>Pure CSS MNIST</h1>
        <a href='https://github.com/T1ckbase/css-handwritten-digit-recognition'>GITHUB</a>
      </header>
      <div>Draw a digit in the box.</div>
      <Board width={W} height={H} />
      <button type='button' class='clear'>
        clear
      </button>
      <div class='prediction-results'>
        {Array.from({ length: 10 }, (_, i) => (
          <div class='bar-row'>
            <span>{i}</span>
            <div class='track'>
              <div class='fill' style={`--p: var(--prob-${i})`}></div>
            </div>
          </div>
        ))}
      </div>
    </body>
  </html>
);
const html = `<!DOCTYPE html>${jsxElement.toString()}`;

await rm('./dist', { recursive: true, force: true });
await Bun.build({
  entrypoints: ['./index.html'],
  files: {
    './index.html': html,
    './board.css': generateBoardCSS(H * W),
  },
  minify: PRODUCTION,
  outdir: './dist',
  publicPath: PRODUCTION ? 'https://t1ckbase.github.io/pure-css-mnist/' : undefined,
});

// https://github.com/oven-sh/bun/issues/16920
// Remove js
await rm(new Bun.Glob('./dist/*.js').scanSync().next().value);
Bun.write(
  './dist/index.html',
  new HTMLRewriter()
    .on('script', {
      element(element) {
        element.remove();
      },
    })
    .on('link', {
      element(element) {
        element.removeAttribute('crossorigin');
      },
    })
    .transform(new Response(Bun.file('./dist/index.html'))),
);
