

# standard library imports
import os
import requests
import json
from collections import defaultdict
import random

# related third-party
from bs4 import BeautifulSoup

# local application/library specific imports
from app_config import AppConfig
from .AlgoProblem import AlgoProblem

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = AppConfig(working_dir)

# get configuration
CONFIG = configProxy.return_config()

# get headers configuration
HEADERS = configProxy.return_request_headers()


class Contest:
    def __init__(self, contest_link: str, contest_category: str):
        self.contest_link = contest_link
        self.contest_id = contest_link.split("/")[-1]
        self.contest_category = contest_category
        self.problems = []
        self.top_contestants = []
        self.problem_submission = defaultdict(list)
        
    def __get_problems_links(self):
        """
        Returns the links of the problems from the given contest page.
        Args:
            contest_page_link (str): The link to the contest page, expected to be a valid URL.
        Returns:
            list: A list of strings, where each string represents a link to a problem page.
        """
        req = requests.get(self.contest_link, HEADERS)
        soup = BeautifulSoup(req.content, 'html5lib')
        problems_table = soup.find("table", class_="problems")
        problems = problems_table.find_all("a", href=lambda href: href and href.startswith(
            "/contest/") and "/problem/" in href)
        list_of_problems = []
        for problem_link in problems:
            full_link = CONFIG["CODEFORCES_LINK"] + problem_link.get("href")
            if full_link not in list_of_problems:
                list_of_problems.append(full_link)
        return list(list_of_problems)    
    
    def __get_top_contestants(self):
        req = requests.get(f"{CONFIG['CODEFORCES_LINK']}contest/{self.contest_id}/standings", HEADERS)
        soup = BeautifulSoup(req.content, 'html5lib')
        
        contestants = soup.find_all("td", class_="contestant-cell")[:20]
        for i in range(len(contestants)):
            self.top_contestants.append(contestants[i].getText().lstrip().rstrip())
            
    def __get_problem_submission(self):
        for contestant in self.top_contestants:
            request_answer = requests.get(f"https://codeforces.com/api/contest.status?contestId={self.contest_id}&handle={contestant}")
            request_answer = request_answer.json()
            if request_answer['status'] == 'FAILED':
                continue
            else:
                data = request_answer['result']
                for submission in data:
                    if 'C++' in submission['programmingLanguage'] and submission['verdict'] == 'OK':
                        self.problem_submission[submission['problem']['index']].append(submission['id'])
        
        return len(self.problem_submission) > 0

    def __get_editorial_link(self):
        """
        Return the link of the editorial for the given contest page.
        Args:
            contest_page_link (str): The link of the contest page, expected to be a valid URL.
        Returns:
            None
        """
        req = requests.get(self.contest_link, HEADERS)
        soup = BeautifulSoup(req.content, 'html5lib')
        contests_materials = soup.find_all("a", href=lambda href: href and "/blog/entry/" in href)

        print(self.contest_link)
        print(soup)
        
        tutorial_content_list = []
        for content in contests_materials:
            if "Tutorial" in content.text or "Editorial" in content.text or "T (en)" in content.text or "E (en)" in content.text:
                if "codeforces.com" not in content.get("href"):
                    tutorial_content_list.append(CONFIG["CODEFORCES_LINK"] + content.get("href"))
                else:
                    tutorial_content_list.append(content.get("href"))

        if len(tutorial_content_list) >= 1:
            return tutorial_content_list[0]

        return None

    def __get_problem_editorial(self):
        editorial_link = self.__get_editorial_link()
        # if editorial_link is None:
        #     print(f"No editorial found for contest {self.contest_id}")
            
    
    def process_contest(self):
        """
        Process the contest page and retrieves the problems links.
        Args:
            None
        Returns:
            None
            
            933, 1632
        """
        
        # self.__get_top_contestants()
        # response = self.__get_problem_submission()
        
        # if response == False:
        #     return
        
        self.__get_problem_editorial()
        
        # problems_links = self.__get_problems_links()
        # for problem_link in problems_links:
            
        #     print(f'{problem_link}')
            
        #     problem = AlgoProblem(problem_link, self.problem_submission)
            
        #     if(problem.interactive):
        #         print(f"Interactive problem found: {problem_link}")
        #         continue
                        
        #     problem_file_path = os.path.join(CONFIG["DATASET_DESTINATION"], self.contest_category, problem.name + ".json")
            
        #     if os.path.isfile(problem_file_path):
        #         problem_file_path = os.path.join(CONFIG["DATASET_DESTINATION"], self.contest_category, problem.name + str(random.randint(1, 100000)) + ".json")
            
        #     output_file = open(problem_file_path, "w")
        #     output_file.write(json.dumps(problem.__dict__))
        #     output_file.close()
