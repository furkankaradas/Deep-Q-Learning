from training import TrainGame

game = TrainGame(game_name='CartPole-v1', n_games=10_000, layer_dims=[128, 128])
game.training()
