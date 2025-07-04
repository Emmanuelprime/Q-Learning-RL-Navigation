import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox,QTextEdit  
)
from PyQt6.QtCore import QTimer
from matplotlib.animation import FuncAnimation

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- Constants and Hyperparameters ---
GRID_SIZE = 10
START = (0, 0)
GOAL = (9, 9)
DEFAULT_OBSTACLES = [(2, 2), (3, 2), (4, 2), (5, 5), (6, 5), (7, 5), (8, 5), (5, 6), (5, 7), (5, 8)]
EPISODES = 2000
ACTIONS = {
    0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
    4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)
}
ALPHA = 0.1
GAMMA = 0.9
EPSILON_DECAY = 0.999
MAX_STEPS = 200
FRAME_SKIP = 5

# --- Simulation GUI Class ---
class RLGui(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RL Pathfinding Simulator (PyQt6)")
        self.setGeometry(100, 100, 1000, 800)

        self.grid_size = GRID_SIZE
        self.start = START
        self.goal = GOAL
        self.obstacles = DEFAULT_OBSTACLES.copy()
        self.episodes = EPISODES
        self.epsilon = 1.0

        self.frames = []
        self.robot_positions = []
        self.explored_paths = []

        self.initUI()  # ✅ Call the UI initializer

    def log(self, message):
        self.log_box.append(message)

    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)

        # --- Top-level layout ---
        main_layout = QHBoxLayout()

        # --- Left: Grid World Plot ---
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Grid World"))
        left_layout.addWidget(self.canvas)

        # --- Right: Controls + Reward Plot ---
        right_layout = QVBoxLayout()

        # --- Input Grid ---
        grid = QGridLayout()
        self.start_input = QLineEdit("0,0")
        self.goal_input = QLineEdit("9,9")
        self.episodes_input = QLineEdit("2000")
        self.obstacle_count_input = QLineEdit("10")

        grid.addWidget(QLabel("Start (x,y):"), 0, 0)
        grid.addWidget(self.start_input, 0, 1)
        grid.addWidget(QLabel("Goal (x,y):"), 0, 2)
        grid.addWidget(self.goal_input, 0, 3)
        grid.addWidget(QLabel("Episodes:"), 1, 0)
        grid.addWidget(self.episodes_input, 1, 1)
        grid.addWidget(QLabel("Obstacle Count:"), 1, 2)
        grid.addWidget(self.obstacle_count_input, 1, 3)

        # --- Buttons ---
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)

        self.gen_obs_btn = QPushButton("Generate Random Obstacles")
        self.gen_obs_btn.clicked.connect(self.generate_random_obstacles)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset)

        grid.addWidget(self.run_btn, 2, 0)
        grid.addWidget(self.gen_obs_btn, 2, 1)
        grid.addWidget(self.reset_btn, 2, 2)

        right_layout.addLayout(grid)

        # --- Reward Plot ---
        self.reward_fig, self.reward_ax = plt.subplots()
        self.reward_canvas = FigureCanvas(self.reward_fig)
        right_layout.addWidget(QLabel("Reward Plot"))
        right_layout.addWidget(self.reward_canvas)

        # --- Add both sides to main layout ---
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=2)

                # --- Log Output ---
        self.log_output = QLabel("Log:")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)

        right_layout.addWidget(self.log_output)
        right_layout.addWidget(self.log_box)

        # Add Legend
        legend_label = QLabel("Legend:\n🟩 Start  🟥 Goal  ⬛ Obstacle  🔵 Path  🟣 Robot  🟢 Optimal Path")
        legend_label.setStyleSheet("font-weight: bold; padding: 5px;")
        left_layout.addWidget(legend_label)


        central.setLayout(main_layout)

        self.update_visualization()


    def reset(self):
        self.start_input.setText("0,0")
        self.goal_input.setText("9,9")
        self.episodes_input.setText("2000")
        self.obstacle_count_input.setText("10")
        self.obstacles = DEFAULT_OBSTACLES.copy()
        self.update_visualization()

    def update_visualization(self):
        self.ax.clear()
        self.ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal')
        self.ax.grid(True)

        for obs in self.obstacles:
            rect = plt.Rectangle((obs[1], obs[0]), 1, 1, facecolor='gray')
            self.ax.add_patch(rect)

        self.ax.add_patch(plt.Circle((self.start[1]+0.5, self.start[0]+0.5), 0.4, color='green'))
        self.ax.add_patch(plt.Circle((self.goal[1]+0.5, self.goal[0]+0.5), 0.4, color='red'))
        self.canvas.draw()

    def generate_random_obstacles(self):
        try:
            count = int(self.obstacle_count_input.text())
        except:
            count = 10
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                         if (x, y) != self.start and (x, y) != self.goal]
        for _ in range(100):
            sample = random.sample(all_positions, min(count, len(all_positions)))
            self.obstacles = sample
            if self.path_exists():
                break
        self.update_visualization()

    def path_exists(self):
        visited = set()
        queue = [self.start]
        while queue:
            curr = queue.pop(0)
            if curr == self.goal:
                return True
            for dx, dy in ACTIONS.values():
                nxt = (curr[0]+dx, curr[1]+dy)
                if (0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE
                        and nxt not in self.obstacles and nxt not in visited):
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def run_simulation(self):
        try:
            self.start = tuple(map(int, self.start_input.text().split(',')))
            self.goal = tuple(map(int, self.goal_input.text().split(',')))
            self.episodes = int(self.episodes_input.text())
        except:
            QMessageBox.warning(self, "Error", "Invalid input values.")
            return

        self.update_visualization()
        self.log("🔁 Training...")
        Q = self.train_robot()
        self.log("✅ Training completed.")
        self.animate(Q)

    def train_robot(self):
        self.frames = []
        self.robot_positions = []
        self.explored_paths = []
        self.epsilon = 1.0
        Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

        success_episodes = []
        rewards = []

        for episode in range(self.episodes):
            state = self.start
            path = [state]
            steps = 0
            episode_frames = []
            total_reward = 0
            done = False

            while not done and steps < MAX_STEPS:
                action = self.choose_action(state, Q)
                next_state = self.get_next(state, action)

                reward = self.get_reward(next_state)
                self.update_q(Q, state, action, next_state, reward)

                state = next_state
                path.append(state)
                total_reward += reward

                if steps % FRAME_SKIP == 0:
                    episode_frames.append((list(path), state))

                if state == self.goal:
                    done = True
                    success_episodes.append(episode_frames)

                steps += 1

            self.epsilon = max(0.01, self.epsilon * EPSILON_DECAY)
            rewards.append(total_reward)

        # Save frames from a few episodes
        selected = random.sample(success_episodes, min(10, len(success_episodes)))
        for ep in selected:
            for f in ep:
                self.frames.append(f)
                self.robot_positions.append(f[1])
                self.explored_paths.append(f[0])

        self.plot_rewards(rewards)
        self.log(f"📈 Training complete. Avg reward: {np.mean(rewards):.2f}")
        return Q

    def update_q(self, Q, state, action, next_state, reward):
        x, y = state
        nx, ny = next_state
        best_next = np.max(Q[nx, ny])
        Q[x, y, action] += ALPHA * (reward + GAMMA * best_next - Q[x, y, action])

    def choose_action(self, state, Q):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(ACTIONS.keys()))
        else:
            x, y = state
            return np.argmax(Q[x, y])

    def get_next(self, state, action):
        x, y = state
        dx, dy = ACTIONS[action]
        next_pos = (x + dx, y + dy)
        if (0 <= next_pos[0] < GRID_SIZE and 0 <= next_pos[1] < GRID_SIZE
                and next_pos not in self.obstacles):
            return next_pos
        return state

    def get_reward(self, state):
        if state == self.goal:
            return 100
        if state in self.obstacles:
            return -100
        return -1

    def animate(self, Q):
        self.ani = FuncAnimation(
        self.fig,
        lambda i: self.draw_frame(i, Q),
        frames=len(self.frames),
        interval=200,
        blit=False,
        repeat=False  # ✅ Don't repeat after the last frame
    )

        self.canvas.draw()

    def draw_frame(self, i, Q):
        self.update_visualization()

        if i < len(self.frames):
            path, pos = self.frames[i]

            # Draw the explored path
            if len(path) > 1:
                y, x = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in path])
                self.ax.plot(x, y, color='#1f77b4', linewidth=2, alpha=0.5)  # Blue path

            # Draw the robot position
            self.ax.add_patch(plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.3, color='#FF00FF'))  # Magenta robot

        # Final frame: draw the optimal path and stop animation
        if i == len(self.frames) - 1:
            optimal_path = self.find_optimal_path(Q)
            if len(optimal_path) > 1:
                y_opt, x_opt = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in optimal_path])
                self.ax.plot(x_opt, y_opt, color='#2ca02c', linewidth=3, linestyle='-')  # Green optimal path

            self.ani.event_source.stop()



    def plot_rewards(self, rewards):
        self.reward_ax.clear()
        self.reward_ax.plot(rewards, label="Total Reward")
        self.reward_ax.set_title("Rewards Over Episodes")
        self.reward_ax.set_xlabel("Episode")
        self.reward_ax.set_ylabel("Reward")
        self.reward_ax.grid(True)
        self.reward_ax.legend()
        self.reward_canvas.draw()

    def find_optimal_path(self, Q):
        path = [self.start]
        state = self.start
        visited = set()

        for _ in range(100):
            x, y = state
            action = np.argmax(Q[x, y])
            next_state = self.get_next(state, action)

            if not (0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE):
                break
            if next_state in visited or next_state in self.obstacles:
                break

            path.append(next_state)
            visited.add(next_state)

            if next_state == self.goal:
                break

            state = next_state

        return path


# --- Run App ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RLGui()
    window.show()
    sys.exit(app.exec())
