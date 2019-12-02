from src.datamanufacture import DataManufacture


def process_data():
    dm = DataManufacture("MOT17-05")
    dm.process_data()


def load_data():
    dm = DataManufacture("MOT17-05")
    data = dm.load_data()
    print(data)


def review_data():
    dm = DataManufacture("MOT17-05")
    dm.review_data()


def generate_data():
    dm = DataManufacture("MOT17-05")
    dm.generate_data()
