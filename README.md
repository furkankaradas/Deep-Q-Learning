# Deep Q Learning

To implement Naive Deep Q Learning one basic example.

You can run this: 
```bash
$ python3 -m venv env
$ source env/bin/activate
$ pip3 install -r requirements.txt
$ cd naive_deep_Q_learning
$ python3 cartpolegame.py 
```

or you can use docker:
```bash
$ docker build -t rl-example .
$ docker run rl-example
```

If you want to change game in gym library, you should change "game_name":

```python
from training import TrainGame

game = TrainGame(game_name='CartPole-v1', n_games=10_000)
game.training()
```
