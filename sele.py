from selenium import webdriver
from selenium.webdriver.support.ui import Select
import selenium
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

browser  = webdriver.Edge(executable_path='msedgedriver.exe')
browser.get("https://ultimate-t3.herokuapp.com/local-game")

current = 0
def write(xPath):
    global current
    WebDriverWait(browser, 20).until(EC.element_to_be_clickable((By.XPATH, xPath))).click()
    sleep(1)
    current += 2

def read():
    global current
    OcurrentLocate = browser.find_element_by_xpath("//*[@id='history-table']/tbody/tr["+str(current)+"]/td[2]").text
    return str(OcurrentLocate)

# Click New game
newgame = browser.find_element_by_id("new-game")
newgame.click()

# Select difficulty
difficulty = Select(browser.find_element_by_id("difficulty"))
difficulty.select_by_visible_text("Impossible")

sleep(1)

while True:
    try:
        # WebDriverWait(browser, 20).until(EC.element_to_be_clickable((By.XPATH, 
        #     "//*[@id='game']/table/tr[3]/td[1]/table/tr[1]/td[1]"))).click()
        # sleep(1)
        # WebDriverWait(browser, 20).until(EC.element_to_be_clickable((By.XPATH, 
        #     "//*[@id='game']/table/tr[2]/td[2]/table/tr[1]/td[1]"))).click()
        # sleep(1)

        # O_current = browser.find_element_by_xpath("//*[@id='history-table']/tbody/tr[2]/td[2]").text
        # print(O_current)
        write("//*[@id='game']/table/tr[3]/td[1]/table/tr[1]/td[1]")
        print(read())
        

    except:
        browser.quit()

browser.close()
