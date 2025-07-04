# standard library imports
import os
import requests
import json
from collections import defaultdict
import random
from time import sleep
import re
import string

# related third-party
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import pypandoc

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
    def __init__(self, contest_link: str, contest_category: str, scraper, driver):
        self.contest_link = contest_link
        self.contest_id = contest_link.split("/")[-1]
        self.contest_category = contest_category
        self.problems = []
        self.top_contestants = []
        self.problem_submissions = defaultdict(list)
        self.editorial_link = None
        self.editorial_content_soup = None
        self.problem_links = []
        self.scraper = scraper
        self.driver = driver
        
    def __get_problems_links(self):
        """
        Returns the links of the problems from the given contest page.
        Args:
            contest_page_link (str): The link to the contest page, expected to be a valid URL.
        Returns:
            list: A list of strings, where each string represents a link to a problem page.
        """
        response = self.scraper.get(self.contest_link)
        soup = BeautifulSoup(response.content, 'html5lib')
        problems_table = soup.find("table", class_="problems")
        problems = problems_table.find_all("a", href=lambda href: href and href.startswith("/contest/") and "/problem/" in href)
        list_of_problems = set()
        for problem_link in problems:
            full_link = CONFIG["CODEFORCES_LINK"] + problem_link.get("href")
            list_of_problems.add(full_link)
        self.problem_links = list(list_of_problems)
        
        if len(self.problem_links) > 0:
            return True
        
        return False
    
    def __get_top_contestants(self):
        response = self.scraper.get(f"{CONFIG['CODEFORCES_LINK']}contest/{self.contest_id}/standings")
        soup = BeautifulSoup(response.content, 'html5lib')
                
        contestants = soup.find_all("td", class_="contestant-cell")[:20]
        for i in range(len(contestants)):
            self.top_contestants.append(contestants[i].getText().lstrip().rstrip())
        
        if len(self.top_contestants) >= 1:
            # print(self.top_contestants)
            return True
        
        return False
            
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
                        self.problem_submissions[submission['problem']['index']].append(submission['id'])
        
        if len(self.problem_submissions) > 0:
            # print(self.problem_submissions.keys())
            return True
        
        return False

    def __get_editorial_link(self):
        """
        Return the link of the editorial for the given contest page.
        Args:
            contest_page_link (str): The link of the contest page, expected to be a valid URL.
        Returns:
            None
        """
        response = self.scraper.get(self.contest_link)
        soup = BeautifulSoup(response.content, 'html5lib')
        contests_materials = soup.find_all("a", href=lambda href: href and "/blog/entry/" in href)

        # print(contests_materials)
        
        def is_relevant_content(content):
            exceptions = ["RCC 2014 WarmUp Analysis", "Round 55 (DIV 2)", "Round 32"]
            text_conditions = ["tutorial", "editorial", "Разбор", "Official tutorial", "Tutorial", "Editorial", "T (en)", "E (en)"]
            text_conditions.extend(exceptions)
            
            if 'Video' in content.text or 'video' in content.text:
                return False
            
            if any(cond in content.text for cond in text_conditions):
                return True
            if 'title' in content.attrs:
                title = content['title'].rstrip(' ')
                if "Editorial" in title or (title.isdigit() and title in content.text):
                    return True
            return False
        
        tutorial_content_list = []
        for content in contests_materials:
            if is_relevant_content(content):
                href = content.get("href")
                if ".ru" not in href and "=ru" not in href:
                    if "codeforces.com" not in href:
                        tutorial_content_list.append(CONFIG["CODEFORCES_LINK"] + href)
                    else:
                        tutorial_content_list.append(href)
        
        # print(tutorial_content_list)

        # sleep(0.3)

        if len(tutorial_content_list) >= 1:
            
            return tutorial_content_list[0]

        return None

    def __start_editorial_link_retrival(self):
        print(f"Editorial retrival started for contest {self.contest_link}")
        while True:
            self.editorial_link = self.__get_editorial_link()
            if self.editorial_link is None:
                # Try again after a pause
                sleep(0.5)
                print(f"Editorial not found for contest {self.contest_link}")
                continue
            break

    def __start_problem_link_retrival(self):
        if len(self.problem_links) == 0:
            while True:
                problem_links_retrived = self.__get_problems_links()
                if problem_links_retrived == False:
                    # Try again after a pause
                    sleep(0.5)
                    print(f"Problems not found for contest {self.contest_link}")
                    continue
                break
    
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
    
    def __convert_latex_to_plain_text(self, latex):
        latex = latex.replace('$$$', '')
        # Replace backslashes with double backslashes
        latex = latex.replace('\\', '\\\\')
        # Replace curly braces with escaped curly braces
        latex = latex.replace('{', '\\{')
        latex = latex.replace('}', '\\}')
        latex = latex.replace('#', '')
        latex = re.sub(f'[^{re.escape(string.printable)}]', ' ', latex)
        
        try:
            plain_text = pypandoc.convert_text(latex, 'plain', format='latex')
        except RuntimeError as e:
            print(f"Error converting LaTeX to plain text: {e}")
            plain_text = latex  # Fallback to the original text if conversion fails
            
        return plain_text
    
    def __get_hints(self, target_problem):
        """Return a **list of unique Hint spoilers** for one Codeforces task.

        * We identify the task section by locating the first `<a>` whose
          visible text starts with the task code (e.g. "2093B –").
        * Only spoiler blocks whose normalised title matches
          ``^hint\d*$`` are collected – the wrapper titled "Hints" is
          ignored.
        * Duplicate hints are removed while preserving order.
        * If no hints are found an empty list is returned (never *None*).
        """

        import re

        contest = target_problem['contest_number']
        idx = target_problem['problem_idx']
        code = f"{contest}{idx}"  # e.g. 2093B

        # 1) locate the section header (first anchor with text starting with code)
        anchor = self.editorial_content_soup.find(
            'a', string=lambda s: s and s.strip().startswith(code)
        )
        if not anchor:
            warn = f"Hint header not found for → {target_problem['link']}"
            print("[WARN]", warn)
            return []
        header = anchor.parent

        # helper: next problem header (anchor for another letter)
        def is_next_problem(tag):
            a = tag.find('a', string=lambda s: s and s.strip().startswith(str(contest)))
            return bool(a and not a.string.strip().startswith(code))

        hints: list[str] = []
        seen: set[str] = set()

        next_header = header.find_next(is_next_problem)
        ptr = header.find_next()
        while ptr and ptr != next_header:
            if ptr.name == 'div' and 'spoiler' in ptr.get('class', []):
                title_el = ptr.find(class_='spoiler-title')
                raw_title = title_el.get_text(strip=True) if title_el else ptr.get_text(strip=True).split('\n', 1)[0]
                norm = re.sub(r'[^a-z]', '', raw_title.lower())
                if re.fullmatch(r'hint\d*', norm):
                    txt = ptr.get_text(" ", strip=True)
                    if txt not in seen:
                        seen.add(txt)
                        hints.append(txt)
            ptr = ptr.find_next()

        return hints

    
    def __get_editorial(self, target_problem):
        
        href = f"contest/{target_problem['contest_number']}/problem/{target_problem['problem_idx']}"
        link = None
        
        for h3 in self.editorial_content_soup.find_all('h3'):
            a_tag = h3.find('a', href=lambda h: h and href in h)
            if a_tag:
                link = a_tag
                break
        
        if not link:
            for h4 in self.editorial_content_soup.find_all('h4'):
                a_tag = h4.find('a', href=lambda h: h and href in h)
                if a_tag:
                    link = a_tag
                    break
                
        if not link:
            for h2 in self.editorial_content_soup.find_all('h2'):
                a_tag = h2.find('a', href=lambda h: h and href in h)
                if a_tag:
                    link = a_tag
                    break
                
        if not link:
            for h1 in self.editorial_content_soup.find_all('h1'):
                a_tag = h1.find('a', href=lambda h: h and href in h)
                if a_tag:
                    link = a_tag
                    break
        if not link:
            for p in self.editorial_content_soup.find_all('p'):
                a_tag = p.find('a', href=lambda h: h and href in h)
                if a_tag:
                    link = a_tag
                    break

        # print(link)

        if link:
            problem_statement_div = link.find_next('div', class_='problem-statement')
            # print(problem_statement_div)
            if problem_statement_div:
                editorial = problem_statement_div.getText()
                editorial = self.__convert_latex_to_plain_text(editorial)
                    
                # print(editorial)

                return editorial
            else:
                paragraphs = []
                next_element = link.find_next()
                while next_element and next_element.name not in ['h2', 'h3', 'h4']:
                    if next_element.name == 'p':
                        # Stop if the paragraph contains a new problem link
                        if next_element.find('a', href=lambda h: 'contest' in h and 'problem' in h):
                            break
                        # Skip "Idea" sections
                        if "Idea:" in next_element.getText() or "Idea :" in next_element.getText():
                            next_element = next_element.find_next('div', class_='spoiler')
                            continue
                        # Skip "Writer" sections
                        if "Writer: " in next_element.getText():
                            next_element = next_element.find_next('div', class_='spoiler')
                            continue
                        if "First to solve:" in next_element.getText():
                            next_element = next_element.find_next('div', class_='spoiler')
                            continue
                        if "Authors:" in next_element.getText() or "Author:" in next_element.getText() or "Authors :" in next_element.getText() or "Author :" in next_element.getText():
                            next_element = next_element.find_next('div', class_='spoiler')
                            continue
                        paragraphs.append(next_element.getText())
                    elif next_element.name == 'div' and 'spoiler' in next_element.get('class', []):
                        # Skip "Hint" sections
                        if "Hint" in next_element.getText():
                            next_element = next_element.find_next('div', class_='spoiler')
                            continue
                        if "General idea" in next_element.getText():
                            next_element = next_element.find_next('div', class_='spoiler')
                            continue
                        if "Video Editorial" in next_element.getText():
                            next_element = next_element.find_next('div', class_='spoiler')
                            continue
                        paragraphs.append(next_element.getText())
                    if next_element.name == 'div':
                        break
                        
                    next_element = next_element.find_next()
                
                if paragraphs:
                    editorial = ' '.join(paragraphs)
                    editorial = self.__convert_latex_to_plain_text(editorial)
                    
                    # print(editorial)
                    
                    return editorial

        with open("logs.txt", "a+", encoding="utf-8") as f:
            f.write(f"Editorial not found for problem {target_problem['link']} \n")
            f.write(target_problem['file_name'] + "\n")
        
        print(f"Editorial not found for problem {target_problem['link']}")
        print(target_problem['file_name']+ "\n")
        return None
    
    def update_contest(self, problems_collection):
        """
        Update the contest data.
        Args:
            None
        Returns:
            None
        """
        
        print(self.contest_link)

        # PROBLEM RETRIVAL START
        
        self.__start_problem_link_retrival()

        # PROBLEM RETRIVAL END
        
        problems_collection_links = [data['link'] for data in problems_collection]
        
        # If at least one problem_link from self.problem_links in the problems_collection_links than check just for update of the problems
        if any(link in problems_collection_links for link in self.problem_links):

            
            for problem_link in self.problem_links:
                if problem_link in problems_collection_links:
                    # Retrieve the problem from the collection
                    target_problem = problems_collection[problems_collection_links.index(problem_link)]

                    # Update editorial link if missing
                    editorial_link_updated = self._update_editorial_link(target_problem)
                    
                    # Check if editorial or hint needs updating
                    needs_editorial_update = (target_problem['editorial'] == "" or target_problem['editorial'] == None)
                    needs_hint_update = (target_problem['hint'] == [] or target_problem['hint'] is None)
                    
                    # If either editorial or hint needs updating, ensure we have the editorial content soup
                    if needs_editorial_update or needs_hint_update:
                        self._ensure_editorial_content_soup(target_problem['editorial_link'])
                    
                    # Update editorial separately
                    if needs_editorial_update:
                        print(f"Updating editorial for problem {target_problem['file_name']}")
                        self._update_editorial(target_problem)
                    
                    # Update hint separately
                    if needs_hint_update:
                        # print(f"Updating hint for problem {target_problem['file_name']}")
                        self._update_hint(target_problem)
                    
                    # Save the problem back to the dataset path
                    with open(target_problem['file_name'], 'w', encoding='utf-8') as f:
                        json.dump(target_problem, f, ensure_ascii=False, indent=4)

        # If no problem_link from self.problem_links in the problems_collection_links than process the whole contest
        else:
            print(f"Contest {self.contest_link} is going to be processed.")
            if self.driver is not None:
                self.process_contest()
            else:
                print("No driver provided.")
                
    def process_contest(self):
        """
        Process the contest page and retrieves the problems links.
        Args:
            None
        Returns:
            None
        """
        
        print(self.contest_link)

        # SUBMISSION RETRIVAL START
        
        while True:
            top_contestants_retrived = self.__get_top_contestants()
            if top_contestants_retrived == False:
                # Try again after a pause
                sleep(0.5)
                print(f"Top contestants not found for contest {self.contest_link}")
                continue
            break
        
        submissions_retrived = self.__get_problem_submission()
        if submissions_retrived == False:
            print(f"Submissions not found for contest {self.contest_link}")
            return
        
        # SUBMISSION RETRIVAL END
        
        # EDITORIAL RETRIVAL START
        
        self.__start_editorial_link_retrival()

        # EDITORIAL RETRIVAL END
        
        # PROBLEM RETRIVAL START        
        
        self.__start_problem_link_retrival()

        for problem_link in self.problem_links:
            
            print(f'{problem_link}')
            
            problem = AlgoProblem(problem_link, self.problem_submissions, self.editorial_link)
            problem.get_problem_solution(self.driver)

            if(problem.interactive):
                print(f"Interactive problem found: {problem_link}")
                continue

            problem_file_path = os.path.join(CONFIG["DATASET_DESTINATION"], self.contest_category, problem.name + ".json")
            
            if os.path.isfile(problem_file_path):
                problem_file_path = os.path.join(CONFIG["DATASET_DESTINATION"], self.contest_category, problem.name + str(random.randint(1, 100000)) + ".json")
            
            problem.set_problem_file_name(problem_file_path)
            
            output_file = open(problem.file_name, "w")
            output_file.write(json.dumps(problem.__dict__))
            output_file.close()
        
        # PROBLEM RETRIVAL END
    
    def _ensure_editorial_content_soup(self, editorial_link):
        """
        Ensure that the editorial content soup is loaded and available.
        Args:
            editorial_link (str): The link to the editorial page
        Returns:
            None
        """
        if self.editorial_content_soup is None and editorial_link:
            self.editorial_link = editorial_link
            self.driver.get(self.editorial_link)
            sleep(0.5)
            self.editorial_content_soup = BeautifulSoup(self.driver.page_source, 'html5lib')
            self.editorial_content_soup = self.__remove_page_element_tags(
                self.editorial_content_soup, ['style', 'span', 'script', 'meta']
            )

    def _update_editorial_link(self, target_problem):
        """
        Update the editorial link for a problem if it's missing.
        Args:
            target_problem (dict): The problem dictionary to update
        Returns:
            bool: True if the editorial link was updated, False otherwise
        """
        if target_problem['editorial_link'] == "" or target_problem['editorial_link'] == None:
            self.__start_editorial_link_retrival()
            target_problem['editorial_link'] = self.editorial_link
            return True
        return False

    def _update_editorial(self, target_problem):
        """
        Update the editorial content for a problem if it's missing.
        Args:
            target_problem (dict): The problem dictionary to update
        Returns:
            bool: True if the editorial was updated, False otherwise
        """
        if target_problem['editorial'] == "" or target_problem['editorial'] == None:
            target_problem['editorial'] = self.__get_editorial(target_problem=target_problem)
            return True
        return False

    def _update_hint(self, target_problem):
        """
        Update the hint content for a problem if it's missing.
        Args:
            target_problem (dict): The problem dictionary to update
        Returns:
            bool: True if the hint was updated, False otherwise
        """
        if target_problem['hint'] == [] or target_problem['hint'] == None:
            target_problem['hint'] = self.__get_hints(target_problem=target_problem)
            if target_problem['hint']:
                print(f"Hint updated for problem {target_problem['file_name']}")
            return True
        return False
