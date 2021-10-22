from state import State, State_2, UltimateTTT_Move
import time
from importlib import import_module
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import selenium
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

browser  = webdriver.Edge(executable_path='msedgedriver.exe')
browser.get("https://ultimate-t3.herokuapp.com/local-game")

def toStateLocate(locate):
    move = UltimateTTT_Move(0, 0, 0, -1)
    post, pre = "", ""
    for i in range(len(locate)):
        if (locate[i] != '/'):
            post += locate[i]
        else:
            pre = locate[i+1:len(locate)]
            break
    arrLocate = ['NW', 'N', 'NE', 'W', 'C', 'E', 'SW', 'S', 'SE']
    move.index_local_board = arrLocate.index(post)
    if (pre == "NW"):
        move.x, move.y = 0, 0
    elif (pre == "N"):
        move.x, move.y = 0, 1
    elif (pre == "NE"):
        move.x, move.y = 0, 2
    elif (pre == "W"):
        move.x, move.y = 1, 0
    elif (pre == "C"):
        move.x, move.y = 1, 1
    elif (pre == "E"):
        move.x, move.y = 1, 2
    elif (pre == "SW"):
        move.x, move.y = 2, 0
    elif (pre == "S"):
        move.x, move.y = 2, 1
    elif (pre == "SE"):
        move.x, move.y = 2, 2
    
    return move

def toWebLocate(locate: UltimateTTT_Move):
    '''
        Form: //*[@id='game']/table/tr[3]/td[1]/table/tr[1]/td[1]
    '''
    post, pre = "", ""
    
    if (locate.index_local_board == 0):
        post = "tr[1]/td[1]"
    elif (locate.index_local_board == 1):
        post = "tr[1]/td[2]"
    elif (locate.index_local_board == 2):
        post = "tr[1]/td[3]"
    elif (locate.index_local_board == 3):
        post = "tr[2]/td[1]"
    elif (locate.index_local_board == 4):
        post = "tr[2]/td[2]"
    elif (locate.index_local_board == 5):
        post = "tr[2]/td[3]"
    elif (locate.index_local_board == 6):
        post = "tr[3]/td[1]"
    elif (locate.index_local_board == 7):
        post = "tr[3]/td[2]"
    elif (locate.index_local_board == 8):
        post = "tr[3]/td[3]"
    ########################
    if (locate.x == 0 and locate.y == 0):
        pre = "tr[1]/td[1]"
    elif (locate.x == 0 and locate.y == 1):
        pre = "tr[1]/td[2]"
    elif (locate.x == 0 and locate.y == 2):
        pre = "tr[1]/td[3]"
    elif (locate.x == 1 and locate.y == 0):
        pre = "tr[2]/td[1]"
    elif (locate.x == 1 and locate.y == 1):
        pre = "tr[2]/td[2]"
    elif (locate.x == 1 and locate.y == 2):
        pre = "tr[2]/td[3]"
    elif (locate.x == 2 and locate.y == 0):
        pre = "tr[3]/td[1]"
    elif (locate.x == 2 and locate.y == 1):
        pre = "tr[3]/td[2]"
    elif (locate.x == 2 and locate.y == 2):
        pre = "tr[3]/td[3]"
    encodeStr = "//*[@id='game']/table/" + post + "/table/" + pre
    return encodeStr


def write(xPath):
    global current
    WebDriverWait(browser, 20).until(EC.element_to_be_clickable((By.XPATH, xPath))).click()
    sleep(1)


def read(current):
    OcurrentLocate = browser.find_element_by_xpath("//*[@id='history-table']/tbody/tr["+str(current)+"]/td[2]").text
    return str(OcurrentLocate)

def goFirst(first):
    First = Select(browser.find_element_by_id("first"))
    if (first):
        First.select_by_visible_text("Player")
    else:
        First.select_by_visible_text("Computer")

Difficulty = ["Piece of cake", "Medium", "Hard", "Very hard", "Extremely hard", "Expert", "Grandmaster", "Impossible"]
def difficultSelect(level):
    # Select difficulty
    difficulty = Select(browser.find_element_by_id("difficulty"))
    difficulty.select_by_visible_text(Difficulty[level])


def generate(player, whoGoFirst = True, rule = 2):
    # Click New game
    newgame = browser.find_element_by_id("new-game")
    newgame.click()
    dict_player = {1: 'X', -1: 'O'}
    if rule == 1:
        cur_state = State()
    else:
        cur_state = State_2()
    turn = 1    

    limit = 81
    remain_time_X = 120
    remain_time_O = 120
    
    player_1 = import_module(player)
    current: int
    if (whoGoFirst):
        current = 0
    else:
        current = 1
    while turn <= limit:
        caption =  browser.find_element_by_xpath("//*[@id='game-caption']").text
        if (caption == "O wins!"):
            print("winner: O")
            print("X:", cur_state.count_X)
            print("O:", cur_state.count_O + 1)
            return -1
        elif (caption == "X wins!"):
            print("winner: X")
            print("X:", cur_state.count_X)
            print("O:", cur_state.count_O + 1)
            return 1
        elif (caption == "It's a tie!"):
            print("Draw")
            print("X:", cur_state.count_X)
            print("O:", cur_state.count_O + 1)
            return 0

        start_time = time.time()
        if (whoGoFirst):
            if cur_state.player_to_move == 1:
                new_move = player_1.select_move(cur_state, remain_time_X)
                write(toWebLocate(new_move))
                elapsed_time = time.time() - start_time
                remain_time_X -= elapsed_time
                current += 2
            else:
                curOLocate = read(current) # read string O
                new_move = toStateLocate(curOLocate)
                elapsed_time = time.time() - start_time
                remain_time_O -= elapsed_time
        else:
            if cur_state.player_to_move == 1:
                curOLocate = read(current) # read string O
                new_move = toStateLocate(curOLocate)
                elapsed_time = time.time() - start_time
                remain_time_O -= elapsed_time
            else:
                new_move = player_1.select_move(cur_state, remain_time_X)
                write(toWebLocate(new_move))
                elapsed_time = time.time() - start_time
                remain_time_X -= elapsed_time
                current += 2

        if new_move == None:
            break
        
        if remain_time_X < 0 or remain_time_O < 0:
            print("out of time")
            print("winner:", dict_player[cur_state.player_to_move * -1])
            break
                
        if elapsed_time > 10.0:
            print("elapsed time:", elapsed_time)
            print("winner: ", dict_player[cur_state.player_to_move * -1])
            break
        
        cur_state.act_move(new_move)
        # print(cur_state)
        turn += 1

def main(player, whoGoFirst=True, level=0, round=1):
    count_O_win, count_X_win, count_Draw = 0, 0, 0
    goFirst(whoGoFirst)
    difficultSelect(level)
    for i in range(round):
        result = generate(player, whoGoFirst, rule=2)
        if (result == -1):
            count_O_win += 1
        elif (result == 1):
            count_X_win += 1
        else:
            count_Draw += 1
    print("\n O win: {0} \t X win: {1} \t Draw: {2}".format(count_O_win, count_X_win, count_Draw))

'''
    Parameter 1: file
    Parameter 2:
        + True: Go FIRST
    Parameter 3: difficulty
        + 0 : "Piece of cake" 
        + 1 : "Medium"
        + 2 : "Hard"
        + 3 : "Very hard"
        + 4 : "Extremely hard"
        + 5 : "Expert"
        + 6 : "Grandmaster"
        + 7 : "Impossible"
    Parameter 4: number of round.
'''
main('_MSSV', True, 0, 5)
