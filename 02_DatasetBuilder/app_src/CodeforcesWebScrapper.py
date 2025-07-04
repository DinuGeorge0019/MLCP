

# standard library imports
import os
from time import sleep
import requests
import json

# related third-party
import undetected_chromedriver as uc
import cloudscraper

# local application/library specific imports
from app_config import AppConfig
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
        self.EXCLUDED_CONTESTS = [
            'http://codeforces.com/contest/2010',           # No editorials
            'http://codeforces.com/contest/1599',           # Not secured
            'http://codeforces.com/contest/1423',           # Not secured
            'http://codeforces.com/contest/1218',           # Not secured
            'http://codeforces.com/contest/1045',           # Not secured
            'http://codeforces.com/contest/286',            # Not secured
            'http://codeforces.com/contest/240',            # No editorials
            'http://codeforces.com/contest/83',             # Russian language
            'http://codeforces.com/contest/1600',           # Not secured
            'http://codeforces.com/contest/1424',           # No editorials
            'http://codeforces.com/contest/1219',           # Not secured
            'http://codeforces.com/contest/1046',           # Not secured
            'http://codeforces.com/contest/854',            # No editorials
            'http://codeforces.com/contest/801',            # No editorials
            'http://codeforces.com/contest/709',            # No editorials
            'http://codeforces.com/contest/697',            # No editorials
            'http://codeforces.com/contest/672',            # No editorials
            'http://codeforces.com/contest/651',            # No editorials
            'http://codeforces.com/contest/588',            # No editorials
            'http://codeforces.com/contest/534',            # No editorials
            'http://codeforces.com/contest/394',            # No editorials
            'http://codeforces.com/contest/352',            # No editorials
            'http://codeforces.com/contest/287',            # Not secured
            'http://codeforces.com/contest/234',            # No editorials
            'http://codeforces.com/contest/203',            # No editorials
            'http://codeforces.com/contest/180',            # No editorials
            'http://codeforces.com/contest/84',             # No editorials
            'http://codeforces.com/contest/49',             # No editorials
            'http://codeforces.com/contest/35',             # No editorials
            'http://codeforces.com/contest/31',             # No editorials
            'http://codeforces.com/contest/4',              # No editorials
            'http://codeforces.com/contest/399',             # No solutions
            'http://codeforces.com/contest/398',            # No solutions
            'http://codeforces.com/contest/397',            # No solutions
            'http://codeforces.com/contest/396',            # No solutions
            'http://codeforces.com/contest/393',            # No solutions
            'http://codeforces.com/contest/392',            # No solutions
            'http://codeforces.com/contest/390',             # No solutions
            'http://codeforces.com/contest/2059',            # No contest
            'http://codeforces.com/contest/598',             # Russian language
            'http://codeforces.com/contest/2062',            # No contest
            'http://codeforces.com/contest/2059',            # No contest
            'http://codeforces.com/contest/631'              # NULL ??
        ]
        
        self.problems_collection = []

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

    def __update_contests_list(self, contest_category, scrapper, driver) -> None:
        """
        Update the list of contests links from the specified file
        Args:
            contest_file_path (str): The file path to the contest file.
            new_contests (list): The list of new contests to add to the file.
        Returns:
            None
        """
        
        dataset_path = os.path.join(CONFIG["DATASET_DESTINATION"], contest_category)
        
        for filename in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, filename), "r", encoding="utf-8") as file:
                data = json.load(file)
                if 'file_name' not in data:
                    # Add file_name as a field in data
                    data['file_name'] = os.path.join(dataset_path, filename)
                if 'editorial_link' not in data:
                    # Add editorial_link as a field in data
                    data['editorial_link'] = None
                if 'editorial' not in data:
                    # Add editorial as a field in data
                    data['editorial'] = None
                if 'hint' not in data:
                    # Add hint as a field in data
                    data['hint'] = None
                    
                self.problems_collection.append(data)
        
        for contest_link in self.contests_list:
            if contest_link in self.EXCLUDED_CONTESTS:
                continue
            
            contest = Contest(contest_link, contest_category, scrapper, driver)
            contest.update_contest(problems_collection=self.problems_collection)
        
        
    
    def __proces_contests_list(self, contest_category, scrapper, driver) -> None:
        """
        Process the list of contests for a given contest category and generates the dataset file for each problem.

        Args:
            contest_category (str): The contest category for which the contests list is to be processed.
        
        Returns:
            None
            
        """

        dataset_path = os.path.join(CONFIG["DATASET_DESTINATION"], contest_category)
        if not os.path.isdir(dataset_path):
            os.mkdir(dataset_path)

        for contest_link in self.contests_list:
            if contest_link in self.EXCLUDED_CONTESTS:
                continue
            
            contest = Contest(contest_link, contest_category, scrapper, driver)
            contest.process_contest()

    def init_driver(self, USE_COOKIES=True):
        if USE_COOKIES:
            # Load the cookies from a file
            with open("codeforces_cookies.json", "r") as file:
                cookies = json.load(file)

        # Set up undetected Chrome
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-application-cache")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--incognito")
        options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36")
        options.add_argument("--headless")
        driver = uc.Chrome(options=options)

        # Open Codeforces and load cookies
        driver.get("https://codeforces.com")
        
        if USE_COOKIES:
            # Clear browser data
            driver.execute_script("window.localStorage.clear();")
            driver.execute_script("window.sessionStorage.clear();")
            
            for cookie in cookies:
                if 'expiry' in cookie:
                    cookie['expiry'] = int(cookie['expirationDate'])  # Rename and ensure expiry is an integer
                if 'expirationDate' in cookie:
                    cookie.pop('expirationDate')  # Remove 'expirationDate' if present
                if 'sameSite' in cookie:
                    cookie.pop('sameSite')  # Remove 'sameSite' because Selenium does not support it
                driver.add_cookie(cookie)
            driver.refresh()
        
        return driver

    def build_dataset(self):
        """
        Fetch all algorithmic problems from the contests retrieved.
        Args:
            None
        Returns:
            None
        """
        # Init the cloudscraper
        scrapper = cloudscraper.create_scraper()
        
        # Init the driver
        driver = self.init_driver()
        
        print("Processing EDUCATIONAL contests")
        self.__read_contests_list(CONFIG["CODEFORCES_EDUCATIONAL_FILE"])
        self.__proces_contests_list("EDUCATIONAL", scrapper, driver)
        print("Processing DIV3 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV3_FILE"])
        self.__proces_contests_list("DIV3", scrapper, driver)
        print("Processing DIV4 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV4_FILE"])
        self.__proces_contests_list("DIV4", scrapper, driver)
        print("Processing DIV1&2 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV1&2_FILE"])
        self.__proces_contests_list("DIV1&2", scrapper, driver)
        print("Processing DIV1 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV1_FILE"])
        self.__proces_contests_list("DIV1", scrapper, driver)
        print("Processing DIV2 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV2_FILE"])
        self.__proces_contests_list("DIV2", scrapper, driver)

    def update_dataset(self):
        """
        Update the dataset with the latest contests.
        Args:
            None
        Returns:
            None
        """
        UPDATE_SOURCE_CODE = False
        
        # Init the cloudscraper
        scrapper = cloudscraper.create_scraper()
        
        if UPDATE_SOURCE_CODE:
            # Init the driver
            driver = self.init_driver(USE_COOKIES=True)
        else:
            driver = self.init_driver(USE_COOKIES=False)
        # Start updating the content of the dataset
        
        print("Processing EDUCATIONAL contests")
        self.__read_contests_list(CONFIG["CODEFORCES_EDUCATIONAL_FILE"])
        self.__update_contests_list("EDUCATIONAL", scrapper, driver)
        print("Processing DIV3 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV3_FILE"])
        self.__update_contests_list("DIV3", scrapper, driver)
        print("Processing DIV4 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV4_FILE"])
        self.__update_contests_list("DIV4", scrapper, driver)
        print("Processing DIV1&2 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV1&2_FILE"])
        self.__update_contests_list("DIV1&2", scrapper, driver)
        print("Processing DIV1 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV1_FILE"])
        self.__update_contests_list("DIV1", scrapper, driver)
        print("Processing DIV2 contests")
        self.__read_contests_list(CONFIG["CODEFORCES_DIV2_FILE"])
        self.__update_contests_list("DIV2", scrapper, driver)