import numpy as np
import time
from visdom import Visdom

class line(object):
    def __init__(self, title, port=8080):
        self.vis = Visdom(port=port)
        self.windows = {}
        self.title = title

    def register_line(self, name, xlabel, ylabel):
        win = self.vis.line(
            X=np.zeros(3),
            Y=np.zeros(3),
            opts=dict(title=self.title, markersize=5, xlabel=xlabel, ylabel=ylabel)
        )
        self.windows[name] = win

    def update_line(self, name, x, *args):
        axis = ()
        for _ in range(len(args)):
            axis += (x+1,)

        self.vis.line(
            X=np.array([axis]),
            Y=np.array([args]),
            win=self.windows[name],
            update='append'
        )



if __name__ == '__main__':
    line = line('loss', port=8097)
    line.register_line('loss', 'iter', 'loss')

    for x in range(5):
        line.update_line('loss', x, 10, 20, 30)
