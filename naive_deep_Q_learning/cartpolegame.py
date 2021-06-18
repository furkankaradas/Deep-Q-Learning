from training import TrainGame

game = TrainGame(game_name='CartPole-v1', n_games=10_000)
game.training()
