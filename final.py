import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QMessageBox, QTextEdit, QFrame,
    QGroupBox
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont, QPalette, QColor
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
        self.setWindowTitle("Reinforcement Learning Pathfinding Simulator")
        self.setGeometry(100, 100, 1200, 850)
        
        # Set application style
        self.setStyle()
        self.grid_size = GRID_SIZE
        self.start = START
        self.goal = GOAL
        self.obstacles = DEFAULT_OBSTACLES.copy()
        self.episodes = EPISODES
        self.epsilon = 1.0

        self.frames = []
        self.robot_positions = []
        self.explored_paths = []

        self.initUI()

    def setStyle(self):
        """Set consistent styling for the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D30;
            }
            QGroupBox {
                font: bold 11px;
                border: 1px solid #3F3F46;
                border-radius: 6px;
                margin-top: 1ex;
                padding: 10px;
                background-color: #252526;
                color: #D4D4D4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
            }
            QLabel {
                color: #D4D4D4;
            }
            QPushButton {
                background-color: #007ACC;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #1C97EA;
            }
            QPushButton:pressed {
                background-color: #005A9E;
            }
            QPushButton:disabled {
                background-color: #3F3F46;
                color: #6D6D6D;
            }
            QLineEdit {
                background-color: #3C3C3C;
                color: #D4D4D4;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                padding: 5px;
            }
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3F3F46;
                border-radius: 4px;
                font-family: Consolas, monospace;
            }
            QFrame {
                background-color: #252526;
                border-radius: 4px;
            }
        """)

    def log(self, message):
        self.log_box.append(message)

    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)

        # --- Top-level layout ---
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- Left: Visualization Panel ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        
        # Visualization Group
        vis_group = QGroupBox("Grid World Visualization")
        vis_layout = QVBoxLayout(vis_group)
        vis_layout.setContentsMargins(10, 20, 10, 10)
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.fig.set_facecolor('#1E1E1E')
        self.ax.set_facecolor('#1E1E1E')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumSize(500, 500)
        vis_layout.addWidget(self.canvas)
        
        # Legend
        legend_layout = QHBoxLayout()
        legend_layout.addWidget(QLabel("Legend:"))
        
        legend_items = [
            ("🟩", "Start"), ("🟥", "Goal"), ("⬛", "Obstacle"),
            ("🔵", "Path"), ("🟣", "Robot"), ("🟢", "Optimal Path")
        ]
        
        for icon, text in legend_items:
            item_layout = QHBoxLayout()
            item_layout.setSpacing(5)
            item_layout.addWidget(QLabel(icon))
            item_layout.addWidget(QLabel(text))
            legend_layout.addLayout(item_layout)
            
        vis_layout.addLayout(legend_layout)
        left_panel.addWidget(vis_group)
        
        # --- Right: Control Panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        # Parameters Group
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QGridLayout(params_group)
        params_layout.setVerticalSpacing(10)
        params_layout.setHorizontalSpacing(15)
        
        # Input fields
        params_layout.addWidget(QLabel("Start (x,y):"), 0, 0)
        self.start_input = QLineEdit("0,0")
        self.start_input.setFixedWidth(100)
        params_layout.addWidget(self.start_input, 0, 1)
        
        params_layout.addWidget(QLabel("Goal (x,y):"), 0, 2)
        self.goal_input = QLineEdit("9,9")
        self.goal_input.setFixedWidth(100)
        params_layout.addWidget(self.goal_input, 0, 3)
        
        params_layout.addWidget(QLabel("Episodes:"), 1, 0)
        self.episodes_input = QLineEdit("2000")
        self.episodes_input.setFixedWidth(100)
        params_layout.addWidget(self.episodes_input, 1, 1)
        
        params_layout.addWidget(QLabel("Obstacle Count:"), 1, 2)
        self.obstacle_count_input = QLineEdit("10")
        self.obstacle_count_input.setFixedWidth(100)
        params_layout.addWidget(self.obstacle_count_input, 1, 3)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.clicked.connect(self.run_simulation)
        button_layout.addWidget(self.run_btn)
        
        self.gen_obs_btn = QPushButton("Generate Obstacles")
        self.gen_obs_btn.clicked.connect(self.generate_random_obstacles)
        button_layout.addWidget(self.gen_obs_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset)
        button_layout.addWidget(self.reset_btn)
        
        params_layout.addLayout(button_layout, 2, 0, 1, 4)
        right_panel.addWidget(params_group)

        # Reward Plot Group
        reward_group = QGroupBox("Training Performance")
        reward_layout = QVBoxLayout(reward_group)
        
        self.reward_fig, self.reward_ax = plt.subplots()
        self.reward_fig.set_facecolor('#1E1E1E')
        self.reward_ax.set_facecolor('#1E1E1E')
        self.reward_canvas = FigureCanvas(self.reward_fig)
        reward_layout.addWidget(self.reward_canvas)
        right_panel.addWidget(reward_group)

        # Log Group
        log_group = QGroupBox("Simulation Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(120)
        log_layout.addWidget(self.log_box)
        right_panel.addWidget(log_group)

        # Add panels to main layout
        main_layout.addLayout(left_panel, 65)
        main_layout.addLayout(right_panel, 35)

        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready to start simulation")
        
        self.update_visualization()

    def reset(self):
        self.start_input.setText("0,0")
        self.goal_input.setText("9,9")
        self.episodes_input.setText("2000")
        self.obstacle_count_input.setText("10")
        self.obstacles = DEFAULT_OBSTACLES.copy()
        self.update_visualization()
        self.log("✅ Configuration reset to defaults")

    def update_visualization(self):
        self.ax.clear()
        
        # Set plot aesthetics
        self.ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
        self.ax.set_xlim(0, GRID_SIZE)
        self.ax.set_ylim(0, GRID_SIZE)
        self.ax.set_aspect('equal')
        self.ax.grid(True, color='#404040', linestyle='-')
        self.ax.tick_params(axis='both', colors='#D4D4D4')
        self.ax.set_facecolor('#1E1E1E')
        
        # Set border color
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#404040')

        # Draw obstacles
        for obs in self.obstacles:
            rect = plt.Rectangle((obs[1], obs[0]), 1, 1, facecolor='#555555', edgecolor='#707070')
            self.ax.add_patch(rect)

        # Draw start and goal
        self.ax.add_patch(plt.Circle((self.start[1]+0.5, self.start[0]+0.5), 0.4, color='#4ECB71'))
        self.ax.add_patch(plt.Circle((self.goal[1]+0.5, self.goal[0]+0.5), 0.4, color='#F15F6F'))
        
        # Set title
        self.ax.set_title('Pathfinding Simulation', color='#D4D4D4', fontsize=12)
        
        self.canvas.draw()

    def generate_random_obstacles(self):
        try:
            count = int(self.obstacle_count_input.text())
        except:
            count = 10
            
        self.log(f"🔧 Generating {count} random obstacles...")
        all_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                         if (x, y) != self.start and (x, y) != self.goal]
        
        for _ in range(100):
            sample = random.sample(all_positions, min(count, len(all_positions)))
            self.obstacles = sample
            if self.path_exists():
                self.log("✅ Obstacles generated - valid path exists")
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
            QMessageBox.warning(self, "Input Error", "Invalid input values. Please use format 'x,y' for coordinates.")
            return

        self.update_visualization()
        self.log("🔁 Starting training...")
        self.status_bar.showMessage("Training in progress...")
        QTimer.singleShot(100, self.start_training)  # Allow UI to update before training

    def start_training(self):
        Q = self.train_robot()
        self.log("✅ Training completed successfully!")
        self.status_bar.showMessage("Training complete - animating results...")
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

            # Update progress every 10% of episodes
            if (episode + 1) % max(1, self.episodes // 10) == 0:
                progress = (episode + 1) / self.episodes * 100
                self.log(f"⏳ Training progress: {progress:.0f}% complete")
                self.status_bar.showMessage(f"Training: {progress:.0f}% complete")

        # Save frames from a few episodes
        selected = random.sample(success_episodes, min(10, len(success_episodes)))
        for ep in selected:
            for f in ep:
                self.frames.append(f)
                self.robot_positions.append(f[1])
                self.explored_paths.append(f[0])

        self.plot_rewards(rewards)
        self.log(f"📈 Training complete. Average reward: {np.mean(rewards):.2f}")
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
            repeat=False
        )

        self.canvas.draw()
        self.status_bar.showMessage("Animation playing...")

    def draw_frame(self, i, Q):
        self.update_visualization()

        if i < len(self.frames):
            path, pos = self.frames[i]

            # Draw the explored path
            if len(path) > 1:
                y, x = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in path])
                self.ax.plot(x, y, color='#1f77b4', linewidth=2, alpha=0.5)

            # Draw the robot position
            self.ax.add_patch(plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.3, color='#FF00FF'))

        # Final frame: draw the optimal path
        if i == len(self.frames) - 1:
            optimal_path = self.find_optimal_path(Q)
            if len(optimal_path) > 1:
                y_opt, x_opt = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in optimal_path])
                self.ax.plot(x_opt, y_opt, color='#2ca02c', linewidth=3, linestyle='-')
                
            self.status_bar.showMessage("Animation complete")
            self.ani.event_source.stop()

    def plot_rewards(self, rewards):
        self.reward_ax.clear()
        
        # Set plot aesthetics
        self.reward_ax.set_facecolor('#1E1E1E')
        self.reward_ax.tick_params(axis='both', colors='#D4D4D4')
        self.reward_ax.spines['bottom'].set_color('#404040')
        self.reward_ax.spines['top'].set_color('#404040')
        self.reward_ax.spines['left'].set_color('#404040')
        self.reward_ax.spines['right'].set_color('#404040')
        
        # Plot data
        self.reward_ax.plot(rewards, label="Total Reward", color='#1f77b4')
        self.reward_ax.set_title("Reward Progression", color='#D4D4D4', fontsize=12)
        self.reward_ax.set_xlabel("Episode", color='#D4D4D4')
        self.reward_ax.set_ylabel("Reward", color='#D4D4D4')
        self.reward_ax.grid(True, color='#404040', linestyle='--')
        self.reward_ax.legend(facecolor='#252526', labelcolor='#D4D4D4')
        
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