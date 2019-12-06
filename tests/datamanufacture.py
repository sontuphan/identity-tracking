from src.datamanufacture import DataManufacture

def generate_small_data():
    batch_size = 16
    dm = DataManufacture("MOT17-05", 8, batch_size)
    pipeline = dm.input_pipeline()

    dataset = pipeline.shuffle(32).batch(batch_size, drop_remainder=True)
    rnn_inputs, cnn_inputs, labels = next(iter(dataset))
    steps = dm.steps_per_epoch
    print('Steps per epoch: {}'.format(steps))
    print(rnn_inputs.shape, cnn_inputs.shape, labels.shape)

def generate_data():
    batch_size = 64
    dm = DataManufacture("MOT17-05", 32, batch_size)
    pipeline = dm.input_pipeline()

    dataset = pipeline.shuffle(128).batch(batch_size, drop_remainder=True)
    rnn_inputs, cnn_inputs, labels = next(iter(dataset))
    steps = dm.steps_per_epoch
    print('Steps per epoch: {}'.format(steps))
    print(rnn_inputs.shape, cnn_inputs.shape, labels.shape)


def review_source():
    dm = DataManufacture("MOT17-05", 16, 32)
    dm.review_source()
