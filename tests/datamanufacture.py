from src.datamanufacture import DataManufacture


def generate_data():
    dm = DataManufacture("MOT17-05", 32)
    pipeline = dm.input_pipeline()

    pipeline_batch = pipeline.shuffle(64).batch(1).take(1)
    print(pipeline_batch)
    a, b, c = next(iter(pipeline_batch))
    print(a)


def review_source():
    dm = DataManufacture("MOT17-05", 32)
    dm.review_source()
