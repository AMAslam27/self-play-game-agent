from games.tictactoe.rules import EMPTY, winner


def score_position(board, turn, bot):
    result = winner(board)
    if result != EMPTY:
        return 1 if result == bot else -1

    legal = [i for i, cell in enumerate(board) if cell == EMPTY]
    if not legal:
        return 0

    scores = []
    for action in legal:
        next_board = board.copy()
        next_board[action] = turn
        scores.append(score_position(next_board, -turn, bot))

    return max(scores) if turn == bot else min(scores)


def minimax_policy(game):
    if game.is_terminal():
        raise ValueError("Cannot choose a move after the game ends")

    bot = game.current_player
    best_action = None
    best_score = -2

    for action in game.legal_actions():
        next_board = game.board.copy()
        next_board[action] = bot
        score = score_position(next_board, -bot, bot)

        if score > best_score:
            best_score = score
            best_action = action

    return best_action
