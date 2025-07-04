

# standard library imports
import os
from time import sleep
import re
import string

# related third-party
from time import sleep
import random
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pypandoc
import cloudscraper


# local application/library specific imports
from app_config import AppConfig

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = AppConfig(working_dir)

# get configuration
CONFIG = configProxy.return_config()

# get headers configuration
HEADERS = configProxy.return_request_headers()


class AlgoProblem:
    def __init__(self, problem_link: str, problem_submissions, editorial_link):
        self.link = problem_link
        self.problemId = ""
        self.problem_idx = problem_link.split("/")[-1]
        split_link = problem_link.split("/")
        self.shortId = split_link[-3] + split_link[-1]
        self.contest_number = split_link[-3]
        self.problem_submissions = problem_submissions
        self.editorial_link = editorial_link
        self.editorial = ""
        self.name = ""
        self.statement = ""
        self.solutions = []
        self.input = ""
        self.output = ""
        self.tags = []
        self.dificulty = ""
        self.interactive = False
        self.file_name = ""
        self.__get_problem_initial_data()

    def set_problem_file_name(self, file_name):
        self.file_name = file_name
    
    def __remove_page_element_tags(self, html_content, content_list):
        """
        Remove HTML tags from the specified list of page elements in the given HTML content.
        Args:
            html_content (bs4.BeautifulSoup): The BeautifulSoup object representing the HTML content.
            content_list (list): A list of page elements to remove from the HTML content.
        Returns:
            bs4.BeautifulSoup: The modified BeautifulSoup object with the specified page elements removed.
        """
        for data in html_content(content_list):
            # Remove tags
            data.decompose()

        # return data by retrieving the tag content
        return html_content
    
    def __get_problem_tags(self, soup):
        possible_problem_tag_boxes = soup.find_all("div", class_="roundbox sidebox borderTopRound")
        problem_tags_found = False
        for possible_tag_box in possible_problem_tag_boxes:
            if "Problem tags" in possible_tag_box.getText():
                tags = possible_tag_box.find_all("span", class_="tag-box")
                for tag in tags:
                    title = tag.get('title')
                    if "Difficulty" in title:
                        self.dificulty = re.sub(r'\D', '', tag.getText())
                    else:
                        self.tags.append(tag.getText().lstrip().rstrip().lower())
                problem_tags_found = True
                break
        if not problem_tags_found:
            print(f"No problem tags found for problem {self.link}")
            return False
        return True

    def __convert_latex_to_plain_text(self, latex):
        latex = latex.replace('$$$', '')
        # Replace backslashes with double backslashes
        latex = latex.replace('\\', '\\\\')
        # Replace curly braces with escaped curly braces
        latex = latex.replace('{', '\\{')
        latex = latex.replace('}', '\\}')
        latex = latex.replace('#', '')
        latex = re.sub(f'[^{re.escape(string.printable)}]', ' ', latex)
        return pypandoc.convert_text(latex, 'plain', format='latex')

    def __get_source_code_solution(self, scraper):
        submission = self.problem_submissions[self.problem_idx][0]
        response = scraper.get(f"https://codeforces.com/contest/{self.contest_number}/submission/{submission}")
        soup = BeautifulSoup(response.content, 'html5lib')
        source_code = soup.find(id='program-source-text')
        if source_code:
            self.solutions.append(source_code.getText()) 
            return True
        else:
            print(f"No submission https://codeforces.com/contest/{self.contest_number}/submission/{submission}")
            return False

    def __get_problem_initial_data(self, get_input_flag = False, get_output_flag = False):
        """
        Get the problem initial data: name, statement, solutions
        Args:
            None
        Returns:
            None
        """
        scrapper = cloudscraper.create_scraper()
        response = scrapper.get(self.link)
        soup = BeautifulSoup(response.content, 'html5lib')    
              
        # Get problem tags
        tag_response = self.__get_problem_tags(soup)

        if not tag_response:
            self.interactive = True
            return

        # Get problem statement and input / output
        soup = self.__remove_page_element_tags(soup, ['style', 'span', 'script', 'meta'])
        
        self.problemId = soup.find("input", {"name": "problemId"})['value']

        problem_statement = soup.find("div", class_="problem-statement")

        self.name = problem_statement.findChild("div", class_="title")
        if not self.name:
            self.interactive = True
            return
        
        self.name = self.name.getText()

        special_chars = "!#$%^&*()/?:\"\'"
        for char in special_chars:
            self.name = self.name.replace(char, ' ')

        self.statement = problem_statement.findChild("div", class_=None)
        if self.statement is None:
            self.statement = problem_statement.findChild("div", class_="legend")
        
        if not self.statement:
            self.interactive = True
            return
        
        self.statement = self.statement.getText()
        
        self.statement = self.__convert_latex_to_plain_text(self.statement)
        
        if get_input_flag:
            input_paragraphs = problem_statement.findChild("div", class_="input-specification")
            if input_paragraphs:
                input_paragraphs = input_paragraphs.contents[1:]
                self.input = ' '.join([data.text for data in input_paragraphs])
            else:
                self.input = None

        if get_output_flag:
            output_paragraphs = problem_statement.findChild("div", class_="output-specification")
            if output_paragraphs:
                output_paragraphs = output_paragraphs.contents[1:]
                self.output = ' '.join([data.text for data in output_paragraphs])
            else:
                self.output = None

    def get_problem_solution(self, driver):
        # Get problem solutions
        for submission in self.problem_submissions[self.problem_idx]:
            
            sleep(random.uniform(20, 30))  # Random sleep between requests
            
            print(f"https://codeforces.com/contest/{self.contest_number}/submission/{submission}")
            
            # Now navigate to the desired page
            driver.get(f"https://codeforces.com/contest/{self.contest_number}/submission/{submission}")
            
            WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.ID, "program-source-text"))
            )

            soup = BeautifulSoup(driver.page_source, 'html5lib')
            source_code = soup.find(id='program-source-text')
            if source_code:
                self.solutions.append(source_code.getText()) 
                break
            else:
                print(f"No solution found for problem https://codeforces.com/contest/{self.contest_number}/submission/{submission}")
                
        if len(self.solutions) == 0:
            self.interactive = True
            return

