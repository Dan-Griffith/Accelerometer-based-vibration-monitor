import sys
import random
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFrame
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt

class LiveGraph(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(2)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: white;")  # Set background to white
        self.data_points = []

    def update_data(self, x, y):
        if self.parent().mode not in [1, 2]:
            self.data_points.clear()
            return  # Only update if mode is 1 or 2
        
        self.data_points.append((x, y))
        if len(self.data_points) > 50:
            self.data_points.pop(0)  # Keep only the last 50 points
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        margin = 40  # Increase margin for axis labels

        if self.parent().mode not in [1, 2]:
            self.data_points.clear()
            return  # Only draw if mode is 1 or 2
        
        if not self.data_points:
            return
        
        x_vals = [p[0] for p in self.data_points]
        y_vals = [p[1] for p in self.data_points]
        
        min_x, max_x = min(x_vals), max(x_vals)
        min_y, max_y = min(y_vals), max(y_vals)
        
        def transform(val, min_val, max_val, new_min, new_max):
            return new_min + (val - min_val) / (max_val - min_val + 1e-5) * (new_max - new_min)
        
        painter.setPen(QPen(Qt.blue, 3))
        prev_x, prev_y = None, None
        for x, y in self.data_points:
            px = int(transform(x, min_x, max_x, margin, width - margin))
            py = int(transform(y, min_y, max_y, height - margin, margin))
            
            if prev_x is not None and prev_y is not None:
                painter.drawLine(int(prev_x), int(prev_y), int(px), int(py))
            prev_x, prev_y = px, py

        # Draw axes
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(margin, height - margin, width - margin, height - margin)  # X-axis
        painter.drawLine(margin, margin, margin, height - margin)  # Y-axis
        
        # Draw labels
        painter.drawText(width // 2, height - 10, "mm/s")  # X-axis label
        painter.drawText(10, height // 2, "Hz")  # Y-axis label
        
        # Draw numerical axis values
        for i in range(5):
            x_pos = int(transform(min_x + i * (max_x - min_x) / 4, min_x, max_x, margin, width - margin))
            y_pos = int(transform(min_y + i * (max_y - min_y) / 4, min_y, max_y, height - margin, margin))
            painter.drawText(x_pos, height - 25, f"{min_x + i * (max_x - min_x) / 4:.1f}")
            painter.drawText(5, y_pos, f"{min_y + i * (max_y - min_y) / 4:.1f}")

class ModeSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.mode = 0  # Initialize mode to 0
        self.initUI()
    
    def initUI(self):
        main_layout = QHBoxLayout()
        button_layout = QVBoxLayout()
        
        # Create a live graph display
        self.live_graph = LiveGraph(self)
        main_layout.addWidget(self.live_graph)
        
        self.btn_idle = QPushButton("Idle Mode", self)
        self.btn_idle.clicked.connect(lambda: self.mode_selected(0))
        button_layout.addWidget(self.btn_idle)
        
        self.btn_calibration = QPushButton("Calibration Mode", self)
        self.btn_calibration.clicked.connect(lambda: self.mode_selected(1))
        button_layout.addWidget(self.btn_calibration)
        
        self.btn_monitoring = QPushButton("Monitoring Mode", self)
        self.btn_monitoring.clicked.connect(lambda: self.mode_selected(2))
        button_layout.addWidget(self.btn_monitoring)
        
        self.btn_stop = QPushButton("Stop Monitoring", self)
        self.btn_stop.clicked.connect(lambda: self.mode_selected(0))
        button_layout.addWidget(self.btn_stop)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        self.setWindowTitle("Mode Selector")
        
        # Timer for simulating live data stream
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.generate_live_data)
        self.timer.start(500)  # Update every 500ms
        
        self.show()
    
    def mode_selected(self, mode):
        self.mode = mode
        print(f"Selected Mode: {mode}")
        self.perform_action(mode)
        self.live_graph.data_points.clear()  # Clear graph when mode changes
        self.live_graph.update()
    
    def perform_action(self, mode):
        if mode == 0:
            print("Idle mode activated.")
        elif mode == 1:
            print("Calibration mode activated.")
        elif mode == 2:
            print("Monitoring mode activated.")
    
    def generate_live_data(self):
        if self.mode in [1, 2]:  # Only generate data in Calibration or Monitoring mode
            mm_per_s = random.uniform(0, 100)
            frequency = random.uniform(0, 500)
            self.live_graph.update_data(mm_per_s, frequency)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModeSelector()
    sys.exit(app.exec_())