import torch as th

from models.mlp import MLP


def main():
    model = MLP(5, [10, 30], is_classification=True)
    x = th.rand(1, 5)
    y = model.forward(x)
    print(y)


if __name__ == "__main__":
    main()
