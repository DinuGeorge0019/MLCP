

# standard library imports
import argparse
import pypandoc

# local application/library specific imports
from app_src import CodeforcesWebScrapper
from app_src import DatasetFactory

def main():
    try:
        version = pypandoc.get_pandoc_version()
        print(f'Pandoc is installed. Version: {version}')
    except OSError:
        pypandoc.download_pandoc()

    parser = argparse.ArgumentParser()
    webScrapper = CodeforcesWebScrapper()
    datasetFactory = DatasetFactory()
    arguments = [
        ("fetch_codeforces_data", webScrapper.fetch_contests, "Fetch codeforces contests links."),
        ("build_codeforces_dataset", webScrapper.build_dataset, "Build codeforces dataset."),
        ("build_raw_dataset", datasetFactory.build_raw_dataset, "Build raw dataset."),
        ("build_train_test_dataset", datasetFactory.build_train_test_dataset, "Build datasets (train / test / val)."),
        ("generate_dataset_overview", datasetFactory.generate_dataset_overview, "Generate dataset overview.")
    ]

    for arg, _, description in arguments:
        if arg == "build_train_test_dataset":
            parser.add_argument(f'--{arg}', type=int, help=description, nargs=1, metavar='TOP_N')
        else:
            parser.add_argument(f'--{arg}', action='store_true', help=description)

    params = parser.parse_args()
    for arg, fun, _ in arguments:
        if hasattr(params, arg) and getattr(params, arg):
            print(f"Executing {arg}")            
            if arg == "build_train_test_dataset":
                top_n = params.build_train_test_dataset[0]
                fun(top_n)
            else:
                fun()


if __name__ == '__main__':
    main()