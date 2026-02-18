from tinygrad import Tensor, TinyJit, dtypes, nn
from tinygrad.nn.datasets import mnist

from css import generate


class TinyConvNet:
    def __init__(self):
        self.l1 = nn.Conv2d(1, 6, kernel_size=3, padding=1, bias=False)
        self.l2 = nn.Conv2d(6, 20, kernel_size=3, padding=1, bias=False)
        self.l3 = nn.Conv2d(20, 10, kernel_size=1, padding=0, bias=False)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu().max_pool2d((2, 2))  # 28 -> 14
        x = self.l2(x).relu().max_pool2d((2, 2))  # 14 -> 7
        x = x.dropout(0.01)
        x = self.l3(x)
        return x.mean((2, 3))


def print_param_summary(model):
    total = 0
    print(f'{"Layer":<20} | {"Shape":<20} | {"Count":<10}')
    print('-' * 55)
    for name, tensor in nn.state.get_state_dict(model).items():
        n = tensor.numel()
        total += n
        print(f'{name:<20} | {str(tensor.shape):<20} | {n:<10,}')
    print('-' * 55)
    print(f'{"Total":<20} | {"":<20} | {total:<10,}\n')


X_train, Y_train, X_test, Y_test = mnist()

X_train = (X_train >= 127.5).cast(dtypes.float32)
X_test = (X_test >= 127.5).cast(dtypes.float32)

model = TinyConvNet()
print_param_summary(model)

optim = nn.optim.AdamW(nn.state.get_parameters(model), lr=0.002)
batch_size = 128


@TinyJit
@Tensor.train()
def train_step():
    samples = Tensor.randint(batch_size, high=X_train.shape[0])

    X, Y = X_train[samples], Y_train[samples]

    optim.zero_grad()
    loss = model(X).sparse_categorical_crossentropy(Y).backward()
    optim.step()
    return loss.realize()


for i in range(40000):
    loss = train_step()
    if i % 100 == 0:
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f'step {i:5d}, loss {loss.item():.2f}, acc {acc * 100.0:.2f}%')

state_dict = nn.state.get_state_dict(model)

with open('model.css', 'w', encoding='utf-8', newline='\n') as f:
    f.write(generate(state_dict))
