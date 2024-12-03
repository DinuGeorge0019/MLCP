

# standard library imports
import os
import time
import requests
import json

# related third-party
from bs4 import BeautifulSoup

# local application/library specific imports
from app_config import AppConfig
from .AlgoProblem import AlgoProblem
from .Contest import Contest

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = AppConfig(working_dir)

# get configuration
CONFIG = configProxy.return_config()

# get headers configuration
HEADERS = configProxy.return_request_headers()


class CodeforcesWebScrapper:
    def __init__(self):
        self.contests_list = []

    def fetch_contests(self):
        """
        Create a request to the codeforces api to retrieve all contests links and saves to the specified file afterwards.
        Args:
            None
        Returns:
            True if the operation succeeded, False otherwise
        """
        codeforces_data_path = os.path.join(CONFIG["WORKING_DIR"], "00_CODEFORCES_DATA")
        if not os.path.isdir(codeforces_data_path):
            os.mkdir(codeforces_data_path)

        def write_line(file, contest_id) -> None:
            file.write(CONFIG["CODEFORCES_BASE_CONSTEST_LINK"] + str(contest_id) + "\n")

        request_answer = requests.get(CONFIG["CODEFORCES_GET_CONTEST_LIST_REQUEST_LINK"])
        request_answer = request_answer.json()
        if request_answer['status'] == 'OK':

            answer_result_data = request_answer['result']

            div1_contests_file = open(CONFIG["CODEFORCES_DIV1_FILE"], "w")
            div1_2_contests_file = open(CONFIG["CODEFORCES_DIV1&2_FILE"], "w")
            div2_contests_file = open(CONFIG["CODEFORCES_DIV2_FILE"], "w")
            div3_contests_file = open(CONFIG["CODEFORCES_DIV3_FILE"], "w")
            div4_contests_file = open(CONFIG["CODEFORCES_DIV4_FILE"], "w")
            educational_contests_file = open(CONFIG["CODEFORCES_EDUCATIONAL_FILE"], "w")

            for contest_data in answer_result_data:
                if 'Educational' in contest_data['name']:
                    write_line(educational_contests_file, contest_data['id'])
                elif 'Div. 1' in contest_data['name'] and "Div. 2" in contest_data['name']:
                    write_line(div1_2_contests_file, contest_data['id'])
                elif 'Div. 1' in contest_data['name']:
                    write_line(div1_contests_file, contest_data['id'])
                elif 'Div. 2' in contest_data['name']:
                    write_line(div2_contests_file, contest_data['id'])
                elif 'Div. 3' in contest_data['name']:
                    write_line(div3_contests_file, contest_data['id'])
                elif 'Div. 4' in contest_data['name']:
                    write_line(div4_contests_file, contest_data['id'])

            div1_contests_file.close()
            div1_2_contests_file.close()
            div2_contests_file.close()
            div3_contests_file.close()
            div4_contests_file.close()
            educational_contests_file.close()
            return True
        else:
            print('Codeforces api servers are down')
            return False

    def __read_contests_list(self, contest_file_path) -> None:
        """
        Read the list of contests links from the specified file
        Args:
            contest_file_path (str): The file path to the contest file.
        Returns:
            None
        """
        with open(contest_file_path) as file:
            lines = file.readlines()
            self.contests_list = [line.rstrip() for line in lines]


    def __proces_contests_list(self, contest_category):
        """
        Process the list of contests for a given contest category and generates the dataset file for each problem.

        Args:
            contest_category (str): The contest category for which the contests list is to be processed.
        
        Returns:
            None
            
        http://codeforces.com/contest/1599
        """
        EXCLUDED_CONTESTS = []

        dataset_path = os.path.join(CONFIG["DATASET_DESTINATION"], contest_category)
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)

        for contest_link in self.contests_list:
            contest_id = contest_link.split("/")[-1]
            if contest_id in EXCLUDED_CONTESTS:
                continue
            
            contest = Contest(contest_link, contest_category)
            contest.process_contest()

    def build_dataset(self):
        """
        Fetch all algorithmic problems from the contests retrieved.
        Args:
            None
        Returns:
            None
        """
        print("Processing EDUCATIONAL contests")
        self.__read_contests_list(CONFIG["CODEFORCES_EDUCATIONAL_FILE"])
        self.__proces_contests_list("EDUCATIONAL")
        # print("Processing DIV3 contests")
        # self.__read_contests_list(CONFIG["CODEFORCES_DIV3_FILE"])
        # self.__proces_contests_list("DIV3")
        # print("Processing DIV4 contests")
        # self.__read_contests_list(CONFIG["CODEFORCES_DIV4_FILE"])
        # self.__proces_contests_list("DIV4")
        # print("Processing DIV1&2 contests")
        # self.__read_contests_list(CONFIG["CODEFORCES_DIV1&2_FILE"])
        # self.__proces_contests_list("DIV1&2")
        # print("Processing DIV1 contests")
        # self.__read_contests_list(CONFIG["CODEFORCES_DIV1_FILE"])
        # self.__proces_contests_list("DIV1")
        # print("Processing DIV2 contests")
        # self.__read_contests_list(CONFIG["CODEFORCES_DIV2_FILE"])
        # self.__proces_contests_list("DIV2")

