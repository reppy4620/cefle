def test1():
    from module.dataset import load_dataset

    ds = load_dataset('../../dataset/anime/portraits', 3)
    print('Loaded ds')
    x = next(ds)
    print(x.dtype, x.shape)


def test2():
    from train import State

    state = State.load('./out/n_0000010.ckpt')
    print(type(state.params['score_estimator/conv2_d']['b']))


if __name__ == '__main__':
    test2()
