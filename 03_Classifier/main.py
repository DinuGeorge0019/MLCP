

#standard library imports
import argparse

# local application/library specific imports
from app_src import CustomEncoder
from app_src import ClassifierChainWrapper

from sklearn.ensemble import RandomForestClassifier
   
   
def main():
    
    from app_src import ClassifierChainWrapper
    from sklearn.ensemble import RandomForestClassifier

    classifierChainWrapper = ClassifierChainWrapper(RandomForestClassifier(), "bert-base-uncased", 5)
    classifierChainWrapper.fit()
    metrics_results = classifierChainWrapper.predict()

    for metric_name, metric_value in metrics_results.items():
        print(f"{metric_name}: {metric_value}")
    
    # parser = argparse.ArgumentParser()
    # classifier = Classifier()
    # arguments = [
    #     ("func", classifier.func, " description "),
    # ]

    # for arg, _, description in arguments:
    #     if arg == "build_filtered_dataset":
    #         parser.add_argument(f'--{arg}', type=int, help=description, nargs=1, metavar='TOP_N')
    #     elif arg == "build_base_train_test_dataset":
    #         parser.add_argument(f'--{arg}', type=int, help=description, nargs=1, metavar='TOP_N')
    #     elif arg == "build_train_test_dataset":
    #         parser.add_argument(f'--{arg}', type=int, help=description, nargs=1, metavar='TOP_N')
    #     else:
    #         parser.add_argument(f'--{arg}', action='store_true', help=description)

    # params = parser.parse_args()
    # for arg, fun, _ in arguments:
    #     if hasattr(params, arg) and getattr(params, arg):
    #         print(f"Executing {arg}")            
    #         if arg == "build_filtered_dataset":
    #             top_n = params.build_filtered_dataset[0]
    #             fun(top_n)
    #         if arg == "build_base_train_test_dataset":
    #             top_n = params.build_base_train_test_dataset[0]
    #             fun(top_n)
    #         if arg == "build_train_test_dataset":
    #             top_n = params.build_train_test_dataset[0]
    #             fun(top_n)
    #         else:
    #             fun()


if __name__ == '__main__':
    main()