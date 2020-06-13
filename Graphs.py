from MapsGenerator import ai_board
import numpy as np
from MinimaxPlayer import MinimaxPlayer
from AlphaBetaPlayer import AlphaBetaPlayer
from OrderedAlphaBetaPlayer import OrderedAlphaBetaPlayer
from LiteAlphaBetaPlayer import LiteAlphaBetaPlayer
from HeavyAlphaBetaPlayer import HeavyAlphaBetaPlayer
import matplotlib.pyplot as plt

# Configurations
# names = ['Minimax', 'AlphaBeta','OrderedAlphaBeta']
names = ['HeavyAlphaBeta','LiteAlphaBeta']
colors = ['r', 'g', 'b']
# player_classes = [MinimaxPlayer, AlphaBetaPlayer, OrderedAlphaBetaPlayer]
player_classes = [HeavyAlphaBetaPlayer, LiteAlphaBetaPlayer]
min_seconds = 1
max_seconds = 15
num_samples = 15


fig = plt.figure()
ax1 = fig.add_subplot(111)
for name, color, player_class in zip(names, colors, player_classes):
    times = []
    depths = []
    for t in np.linspace(min_seconds, max_seconds, num_samples):
        player = player_class()
        player.set_game_params(ai_board.copy())
        d = player.make_move(t)
        times.append(t)
        depths.append(d)
    ax1.plot(times, depths, c=color, label=name)
plt.xlabel('Time')
plt.ylabel('Depth')
plt.legend(loc='upper left')
plt.show()
# times = []
# depths = []
# for t in np.linspace(0.1, 3, 50):
#     player = MinimaxPlayer()
#     player.set_game_params(ai_board.copy())
#     d = player.make_move(t)
#     times.append(t)
#     depths.append(d)
#
# print(len(times),len(depths))
# plt.scatter(times, depths)
# plt.show()