

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
        ("update_codeforces_dataset", webScrapper.update_dataset, "Update codeforces dataset."),
        ("build_raw_dataset", datasetFactory.build_raw_dataset, "Build raw dataset."),
        ("backward_update_json_files", datasetFactory.backward_update_json_files, "Backward update json files."),
        ("build_base_train_test_dataset", datasetFactory.build_base_train_test_dataset, "Build base train / test datasets."),
        ("build_train_test_dataset", datasetFactory.build_train_test_dataset, "Build datasets (train / test / val)."),
        ("generate_dataset_overview", datasetFactory.generate_dataset_overview, "Generate dataset overview."),
        ("build_outside_train_test_dataset", datasetFactory.build_outside_train_test_dataset, "Build outside train / test datasets."),
        ("build_outside_nli_dataset", datasetFactory.build_outside_nli_dataset, "Build outside NLI dataset."),
        ('check_nli_dataset', datasetFactory.check_nli_dataset, 'Check NLI dataset.'),
        ('augument_train_data', datasetFactory.augument_train_data, 'Augument train data.'),
        ('build_nli_dataset', datasetFactory.build_nli_dataset, 'Build NLI dataset.'),
        ('build_train_test_dataset_without_tag_encoding', datasetFactory.build_train_test_dataset_without_tag_encoding, 'Build train / test dataset without tag encoding.'),
        ('build_outside_train_test_dataset_without_tag_encoding', datasetFactory.build_outside_train_test_dataset_without_tag_encoding, 'Build outside train / test dataset without tag encoding.'),
    ]

    for arg, _, description in arguments:
        if arg == "build_train_test_dataset"  or arg == "build_outside_train_test_dataset" or arg == "build_outside_nli_dataset" or arg == 'check_nli_dataset' or arg == 'augument_train_data' or arg == 'build_nli_dataset' or arg == 'build_train_test_dataset_without_tag_encoding' or arg == 'build_outside_train_test_dataset_without_tag_encoding':
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
            elif arg == "build_outside_train_test_dataset":
                top_n = params.build_outside_train_test_dataset[0]
                fun(top_n)
            elif arg == "build_outside_nli_dataset":
                top_n = params.build_outside_nli_dataset[0]
                fun(top_n)
            elif arg == 'check_nli_dataset':
                top_n = params.check_nli_dataset[0]
                fun(top_n)
            elif arg == 'augument_train_data':
                top_n = params.augument_train_data[0]
                fun(top_n)
            elif arg == 'build_nli_dataset':
                top_n = params.build_nli_dataset[0]
                fun(top_n)
            elif arg == 'build_train_test_dataset_without_tag_encoding':
                top_n = params.build_train_test_dataset_without_tag_encoding[0]
                fun(top_n)
            elif arg == 'build_outside_train_test_dataset_without_tag_encoding':
                top_n = params.build_outside_train_test_dataset_without_tag_encoding[0]
                fun(top_n)
            else:
                fun()

if __name__ == '__main__':
    main()

