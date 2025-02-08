

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
        ("check_filtered_dataset", datasetFactory.check_filtered_dataset, "Check filtered dataset."),
        ("build_base_train_test_dataset", datasetFactory.build_base_train_test_dataset, "Build base train / test datasets."),
        ("build_train_test_dataset", datasetFactory.build_train_test_dataset, "Build datasets (train / test / val)."),
        ("build_balanced_train_dataset", datasetFactory.build_balanced_train_dataset, "Build balanced train dataset."),
        ("build_augument_tag_train_dataset", datasetFactory.build_augument_tag_train_dataset, "Build augmented tag train dataset."),
        ("build_augument_train_with_editorials_dataset", datasetFactory.build_augument_train_with_editorials_dataset, "Build augmented train with editorials dataset."),
        ("generate_dataset_overview", datasetFactory.generate_dataset_overview, "Generate dataset overview."),
        ("build_nli_dataset", datasetFactory.build_nli_dataset, "Build NLI dataset.")
    ]

    for arg, _, description in arguments:
        if arg == "build_base_train_test_dataset" or arg == "build_train_test_dataset" or arg == "build_balanced_train_dataset"or arg == "build_augument_tag_train_dataset" or arg == "build_augument_train_with_editorials_dataset" or arg == "build_nli_dataset":
            parser.add_argument(f'--{arg}', type=int, help=description, nargs=1, metavar='TOP_N')
        else:
            parser.add_argument(f'--{arg}', action='store_true', help=description)

    params = parser.parse_args()
    for arg, fun, _ in arguments:
        if hasattr(params, arg) and getattr(params, arg):
            print(f"Executing {arg}")            
            if arg == "build_base_train_test_dataset":
                top_n = params.build_train_test_dataset[0]
                fun(top_n)
            elif arg == "build_train_test_dataset":
                top_n = params.build_train_test_dataset[0]
                fun(top_n)
            elif arg == "build_balanced_train_dataset":
                top_n = params.build_balanced_train_dataset[0]
                fun(top_n)
            elif arg == "build_augument_tag_train_dataset":
                top_n = params.build_augument_tag_train_dataset[0]
                fun(top_n)
            elif arg == "build_augument_train_with_editorials_dataset":
                top_n = params.build_augument_train_with_editorials_dataset[0]
                fun(top_n)
            elif arg == "build_nli_dataset":
                top_n = params.build_nli_dataset[0]
                fun(top_n)
            else:
                fun()

if __name__ == '__main__':
    main()

