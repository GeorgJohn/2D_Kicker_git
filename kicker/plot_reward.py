import matplotlib.pyplot as plt


class RewardPlot:

    def __init__(self):
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([], [], '-')

        self.ax.set_autoscaley_on(True)
        self.ax.grid()

    def update(self, episodes, rewards):
        self.lines.set_xdata(episodes)
        self.lines.set_ydata(rewards)

        self.ax.relim()
        self.ax.autoscale_view()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
