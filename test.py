def test1():
    from module.dataset import load_dataset

    ds = load_dataset('../../dataset/anime/portraits', 3)
    print('Loaded ds')
    x = next(ds)
    print(x.dtype, x.shape)


if __name__ == '__main__':
    test1()
