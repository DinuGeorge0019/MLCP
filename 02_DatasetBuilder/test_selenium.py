import undetected_chromedriver as uc
import json
from time import sleep
import random
from bs4 import BeautifulSoup
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

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
# options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.111 Safari/537.36")
# options.add_argument("--headless")
driver = uc.Chrome(options=options)

# Open Codeforces and load cookies
driver.get("https://codeforces.com")

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


data = [
    "https://mirror.codeforces.com/contest/2043/submission/298219869",
    "https://mirror.codeforces.com/contest/2043/submission/298216982",
    "https://mirror.codeforces.com/contest/2043/submission/298222571",
    "https://mirror.codeforces.com/contest/2043/submission/298218218",
    "https://mirror.codeforces.com/contest/2043/submission/298213410",
    "https://mirror.codeforces.com/contest/2043/submission/298224781",
    "https://mirror.codeforces.com/contest/2043/submission/298213619",
    "https://mirror.codeforces.com/contest/2043/submission/298219577",
    "https://mirror.codeforces.com/contest/2043/submission/298216318",
    "https://mirror.codeforces.com/contest/2043/submission/298221470",
    "https://mirror.codeforces.com/contest/2043/submission/298226089",
    "https://mirror.codeforces.com/contest/2043/submission/298219311",
    "https://mirror.codeforces.com/contest/2043/submission/298219518",
    "https://mirror.codeforces.com/contest/2043/submission/298230143",
    "https://mirror.codeforces.com/contest/2043/submission/298226137",
    "https://mirror.codeforces.com/contest/2043/submission/298226579",
    "https://mirror.codeforces.com/contest/2043/submission/298229601",
    "https://mirror.codeforces.com/contest/2043/submission/298224236",
    "https://mirror.codeforces.com/contest/2043/submission/298231826"
]

for link in data:
    # Now navigate to the desired page
    driver.get(link)

    pre_element = WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.ID, "program-source-text"))
    )

    soup = BeautifulSoup(driver.page_source, 'html5lib')
    source_code = soup.find(id='program-source-text')
    print(source_code)
    print("Waiting for the page to be loaded...")
    
    sleep(random.uniform(10, 20))  # Random sleep between requests


sleep(5)

# for link in data:
# Now navigate to the desired page
# driver.get("https://codeforces.com/contest/2043/submission/35476620")

# pre_element = WebDriverWait(driver, 20).until(
# EC.visibility_of_element_located((By.ID, "program-source-text"))
# )

# soup = BeautifulSoup(driver.page_source, 'html5lib')
# source_code = soup.find(id='program-source-text')
# print(source_code)
# print("Waiting for the page to be loaded...")
# sleep(2)


sleep(1000)

print("Successfully bypassed CAPTCHA and logged in!")
