from selenium import webdriver
import selenium
from time import sleep
browser  = webdriver.Edge(executable_path='msedgedriver.exe')

browser.get("https://ultimate-t3.herokuapp.com/local-game")

newgame = browser.find_element_by_id("new-game")
newgame.click()

NWBoard = browser.find_element_by_class_name("smallBoard NW enabled")[0]

browser.find_element_by_xpath()

