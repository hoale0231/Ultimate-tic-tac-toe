def computePerBlock(block):
        # 3 | 2 | 3
        # ---------
        # 2 | 4 | 2
        # ---------
        # 3 | 2 | 3
        
        # f = f(A) - f(B) + f(C_thread) - f(D_advantage)
    score = 0

    # block = block.reshape(3, 3)

    row_sum = np.sum(block, 1)
    col_sum = np.sum(block, 0)
    diagional_Left = block[0][0] + block[1][1] + block[2][2]
    diagional_Right = block[0][2] + block[1][1] + block[2][0]

    MatrixScore = np.array([[3,2,3],
                            [2,4,2],
                            [3,2,3]])

    score -= np.sum(block*MatrixScore)

    print('score: ' + str(score))

    X2 = 0
    O2 = 0

    X2 += any(row_sum == 2) + any(col_sum == 2) + diagional_Left == 2 + diagional_Right == 2 
    O2 += any(row_sum == -2) + any(col_sum == -2) + diagional_Left == -2 + diagional_Right == -2

    return score - O2*6 + X2*6

def evalFunction(cur_state: State):
    scoreState = 0

    board = cur_state.blocks
    row_sum = [0,0,0]
    col_sum = [0,0,0]
    diagional_Left, diagional_Right = 0,0
    for i in range(3):
        if (i==0):
            if (cur_state.game_result(board[3*i+0]) != None):
                row_sum[0] += cur_state.game_result(board[3*i+0])
                col_sum[0] += cur_state.game_result(board[3*i+0])
                diagional_Left += cur_state.game_result(board[3*i+0])
            elif (cur_state.game_result(board[3*i+1]) != None):
                row_sum[0] += cur_state.game_result(board[3*i+1])
                col_sum[1] += cur_state.game_result(board[3*i+1])
            elif (cur_state.game_result(board[3*i+2]) != None):
                row_sum[0] += cur_state.game_result(board[3*i+2])
                col_sum[2] += cur_state.game_result(board[3*i+2])
                diagional_Right += cur_state.game_result(board[3*i+2])
        elif (i==1):
            if (cur_state.game_result(board[3*i+0]) != None):
                row_sum[1] += cur_state.game_result(board[3*i+0])
                col_sum[0] += cur_state.game_result(board[3*i+0])
            elif (cur_state.game_result(board[3*i+1]) != None):
                row_sum[1] += cur_state.game_result(board[3*i+1])
                col_sum[1] += cur_state.game_result(board[3*i+1])
                diagional_Left += cur_state.game_result(board[3*i+1])
                diagional_Right += cur_state.game_result(board[3*i+1])
            elif (cur_state.game_result(board[3*i+2]) != None):
                row_sum[1] += cur_state.game_result(board[3*i+2])
                col_sum[2] += cur_state.game_result(board[3*i+2])
        elif (i==2):
            if (cur_state.game_result(board[3*i+0]) != None):
                row_sum[2] += cur_state.game_result(board[3*i+0])
                col_sum[0] += cur_state.game_result(board[3*i+0])
                diagional_Right += cur_state.game_result(board[3*i+0])
            elif (cur_state.game_result(board[3*i+1]) != None):
                row_sum[2] += cur_state.game_result(board[3*i+1])
                col_sum[1] += cur_state.game_result(board[3*i+1])
            elif (cur_state.game_result(board[3*i+2]) != None):
                row_sum[2] += cur_state.game_result(board[3*i+2])
                col_sum[2] += cur_state.game_result(board[3*i+2])
                diagional_Left += cur_state.game_result(board[3*i+2])

    for i in range(9):
        if (cur_state.game_result(board[i]) == None):
            scoreState += computePerBlock(board[i])
            # print("score block " + str(i) + ': ' + str(computePerBlock(board[i])))
    
    X2 = 0
    O2 = 0

    X2 += row_sum[0] == 2 + row_sum[1] == 2 + row_sum[2] == 2 + col_sum[0] == 2 + col_sum[1] == 2 + col_sum[2] == 2 + diagional_Left == 2 + diagional_Right == 2 
    O2 += row_sum[0] == -2 + row_sum[1] == -2 + row_sum[2] == -2 + col_sum[0] == -2 + col_sum[1] == -2 + col_sum[2] == -2 + diagional_Left == -2 + diagional_Right == -2
    return scoreState - O2*6 + X2*6

