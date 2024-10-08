from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import (QApplication, QAction, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton, QTextEdit, QToolBar, QVBoxLayout, QWidget,QPushButton, QToolButton, QComboBox, QMenu)
import sys
import os
import time
import csv
from math import radians, sin, cos, sqrt, atan2
from memory_profiler import memory_usage
import heapq

class Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Program 1: Route Finder")
        root_dir = os.getcwd()
        icon_path = os.path.join(root_dir + r"\CS461-Routes", 'lia.ico')
        self.setWindowIcon(QIcon(icon_path))

        # Set up central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.resize(550, 165)

        self.toolbar_light_style = """
            QMenuBar {
                background-color: #353535;
            }
            QMenuBar::item {
                background-color: #353535;
                color: white;
            }
            QMenuBar::item:selected {
                background-color: #292929;
            }
            QToolBar {
                border: none;
            }
            QToolButton {
                background-color: #232939;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                margin-right: 10px;
                margin-bottom: 10px;
                width: 150px;
                height: 25px;
                font-size: 14px;
            }
            QToolButton:hover {
                background-color: #4b516b;
            }
            QToolButton:pressed {
                background-color: #999;
            }
        """
        self.createToolBar()
        self.create_widgets()

        self.selected_algorithm = None
        self.graph = self.load_graph_from_file(r'CS461-Routes\Adjacencies.txt')
        self.coordinates = self.load_coordinates_from_file(r'CS461-Routes\coordinates.csv')

    def load_cities_from_file(self, filename):
        cities = set()
        with open(filename, 'r') as file:
            for line in file:
                city1, city2 = line.strip().split()
                cities.add(city1)
                cities.add(city2)
        return sorted(list(cities))  # Convert the set to a sorted list
    
    def load_graph_from_file(self, filename):
        graph = {}
        with open(filename, 'r') as file:
            for line in file:
                city1, city2 = line.strip().split()
                if city1 not in graph:
                    graph[city1] = []
                if city2 not in graph:
                    graph[city2] = []
                graph[city1].append(city2)
                graph[city2].append(city1)  # Comment this line if your graph is directed
        return graph
    
    def load_coordinates_from_file(self, filename):
        coordinates = {}
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                city, lat, lon = row
                coordinates[city] = (float(lat), float(lon))
        return coordinates
    
    def calculate_distance(self, city1, city2):
        lat1, lon1 = self.coordinates[city1]
        lat2, lon2 = self.coordinates[city2]

        # Convert coordinates to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        # Radius of earth in kilometers
        R = 6371.0

        # Calculate the distance
        distance = R * c

        return distance
    
    def calculate_total_distance(self, route):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += self.calculate_distance(route[i], route[i+1])
        return round(total_distance, 3)
    
    def createStyledMenu(self):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #232939;
                color: white;
                margin: 0px;
            }
            QMenu::item {
                padding: 10px 100px 10px 20px;
            }
            QMenu::item:selected {
                background-color: #4b516b;
            }
        """)
        return menu

    def createToolBar(self):
        toolBar = QToolBar(self)
        self.addToolBar(Qt.LeftToolBarArea, toolBar)
        toolBar.setStyleSheet(self.toolbar_light_style)

        brute = QToolButton(self)
        brute.setText('Brute-force Approaches')
        brute.setPopupMode(QToolButton.InstantPopup)
        toolBar.addWidget(brute)

        brute_menu = self.createStyledMenu() # use this line instead

        bfs = QAction('Breadth-first Search', self)
        bfs.triggered.connect(self.selectBreadthFirstSearch)
        brute_menu.addAction(bfs)

        dfs = QAction('Depth-first Search', self)
        dfs.triggered.connect(self.selectdepthFirstSearch)
        brute_menu.addAction(dfs)

        id_dfs = QAction('ID-DFS Search', self)
        id_dfs.triggered.connect(self.selectidDfsSearch)
        brute_menu.addAction(id_dfs)

        heuristic_menu = self.createStyledMenu()

        heuristic = QToolButton(self)
        heuristic.setText('Heuristic Approaches')
        heuristic.setPopupMode(QToolButton.InstantPopup)
        toolBar.addWidget(heuristic)

        best_first = QAction('Best-first Search', self)
        best_first.triggered.connect(self.selectbestFirstSearch)
        heuristic_menu.addAction(best_first)

        astar = QAction('A* Search', self)
        astar.triggered.connect(self.selectaStarSearch)
        heuristic_menu.addAction(astar)

        # add the menu to the button
        brute.setMenu(brute_menu)
        heuristic.setMenu(heuristic_menu)

    def create_widgets(self):
        # Font Size
        font = QFont()
        font.setPointSize(14)

        # Create a QHBoxLayout
        input_layout = QHBoxLayout()

        # Create labels for the "Start" and "End" dropdown menus
        start_label = QLabel("Start:", self)
        end_label = QLabel("End:", self)

        # Create the "Start" and "End" dropdown menus
        self.start_dropdown = QComboBox(self)
        self.end_dropdown = QComboBox(self)

        # Add cities to the dropdown menus
        cities = self.load_cities_from_file(r'CS461-Routes\Adjacencies.txt')
        self.start_dropdown.addItems(cities)
        self.end_dropdown.addItems(cities)

        self.status_label = QLabel("Selected Algorithm", self)
        self.status_label.setFixedHeight(30)
        self.status_label.setFixedWidth(115)
        self.status_label.setStyleSheet('background-color: white; color: black')
        self.status_label.setAlignment(Qt.AlignCenter)

        self.connect_button = QPushButton("Run", self)
        self.connect_button.clicked.connect(self.runSelectedAlgorithm)
        self.connect_button.setFixedHeight(30)
        self.connect_button.setFixedWidth(100)

        # Add the QLineEdit and QPushButton to the input layout
        input_layout.addWidget(start_label)
        input_layout.addWidget(self.start_dropdown)
        input_layout.addWidget(end_label)
        input_layout.addWidget(self.end_dropdown)
        input_layout.addWidget(self.status_label)
        input_layout.addWidget(self.connect_button)

        # Add the input layout to the main layout
        self.main_layout.addLayout(input_layout)

        self.outputbox = QTextEdit(self)
        self.outputbox.setAcceptRichText(True)

        # Add the QTextEdit to the main layout
        self.main_layout.addWidget(self.outputbox)

    def runSelectedAlgorithm(self):
        if self.selected_algorithm is not None:
            self.selected_algorithm()
        else:
            QMessageBox.warning(self, "No Algorithm Selected", "Please select an algorithm before running.")

    # Undirected (blind) brute-force approaches 
    def selectBreadthFirstSearch(self):
        '''Select the breadth-first search algorithm'''
        self.status_label.setText('Breadth-first Search')
        self.selected_algorithm = self.breadthFirstSearch

    def breadthFirstSearch(self):
        '''breadth-first search: Uses a queue'''
        self.outputbox.clear()
        # Memory usage before the operation in KiB
        start_mem = memory_usage()[0] * 1024

        start_time = time.perf_counter()

        start_city = self.start_dropdown.currentText()
        end_city = self.end_dropdown.currentText()

        graph = self.graph

        visited = []
        queue = [[start_city]]
        while queue:
            path = queue.pop(0)
            city = path[-1]  # The last city in the path is the current city
            if city == end_city:
                # The path is the route from the start city to the end city
                self.outputbox.setText("<b>Path:</b> " + ' -> '.join(path) + '\n')

                # Total time to run search algorithm
                end_time = time.perf_counter()
                total_time_seconds = end_time - start_time
                total_time_microseconds = round(total_time_seconds * 1000000, 3)
                self.outputbox.append(f"<b>Total time:</b> {total_time_microseconds} microseconds.")
                
                # Total distance traveled in KM
                route = path
                total_distance = self.calculate_total_distance(route)
                self.outputbox.append(f"<b>Total distance:</b> {total_distance} KM.")

                # Memory usage after the operation
                end_mem = memory_usage()[0] * 1024
                total_mem = end_mem - start_mem
                self.outputbox.append(f"<b>Total memory used:</b> {round(total_mem, 3)} KiB.")
                return path

            if city not in visited:
                visited.append(city)
                for neighbor in graph[city]:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

    def selectdepthFirstSearch(self):
        '''Select the depth-first search algorithm'''
        self.status_label.setText('Depth-first Search')
        self.selected_algorithm = self.depthFirstSearch

    def depthFirstSearch(self):
        '''depth-first search: Uses a stack'''
        self.outputbox.clear()
        # Memory usage before the operation in KiB
        start_mem = memory_usage()[0] * 1024

        start_time = time.perf_counter()
        
        start_city = self.start_dropdown.currentText()
        end_city = self.end_dropdown.currentText()

        graph = self.graph

        visited = []
        stack = [[start_city]]
        while stack:
            path = stack.pop()
            city = path[-1]  # The last city in the path is the current city
            if city == end_city:
                # The path is the route from the start city to the end city
                self.outputbox.setText("<b>Path:</b> " + ' -> '.join(path) + '\n')

                # Total time to run search algorithm
                end_time = time.perf_counter()
                total_time_seconds = end_time - start_time
                total_time_microseconds = round(total_time_seconds * 1000000, 3)
                self.outputbox.append(f"<b>Total time:</b> {total_time_microseconds} microseconds.")
                
                # Total distance traveled in KM
                route = path
                total_distance = self.calculate_total_distance(route)
                self.outputbox.append(f"<b>Total distance:</b> {total_distance} KM.")

                # Memory usage after the operation
                end_mem = memory_usage()[0] * 1024
                total_mem = end_mem - start_mem
                self.outputbox.append(f"<b>Total memory used:</b> {round(total_mem, 3)} KiB.")
                return path

            if city not in visited:
                visited.append(city)
                for neighbor in graph[city]:
                    new_path = list(path)
                    new_path.append(neighbor)
                    stack.append(new_path)

    def selectidDfsSearch(self):
        '''Select the depth-first search algorithm'''
        self.status_label.setText('ID-DFS search')
        self.selected_algorithm = self.depthFirstSearch

    def idDfsSearch(self):
        '''ID-DFS search: Uses a depth counter'''
        self.outputbox.clear()
        # Memory usage before the operation in KiB
        start_mem = memory_usage()[0] * 1024

        start_time = time.perf_counter()
        
        start_city = self.start_dropdown.currentText()
        end_city = self.end_dropdown.currentText()

        graph = self.graph

        depth = 0
        while True:
            visited = []
            stack = [(start_city, depth)]
            while stack:
                city, depth = stack.pop()
                if city == end_city:
                    # The path is the route from the start city to the end city
                    self.outputbox.setText("<b>Path:</b> " + ' -> '.join(visited) + '\n')

                    # Total time to run search algorithm
                    end_time = time.perf_counter()
                    total_time_seconds = end_time - start_time
                    total_time_microseconds = round(total_time_seconds * 1000000, 3)
                    self.outputbox.append(f"<b>Total time:</b> {total_time_microseconds} microseconds.")
                    
                    # Total distance traveled in KM
                    route = visited
                    total_distance = self.calculate_total_distance(route)
                    self.outputbox.append(f"<b>Total distance:</b> {total_distance} KM.")

                    # Memory usage after the operation
                    end_mem = memory_usage()[0] * 1024
                    total_mem = end_mem - start_mem
                    self.outputbox.append(f"<b>Total memory used:</b> {round(total_mem, 3)} KiB.")
                    return visited

                if city not in visited:
                    visited.append(city)
                    if depth < len(graph):  # Limit the depth to the number of nodes
                        for neighbour in graph[city]:
                            stack.append((neighbour, depth + 1))
            depth += 1

    # Heuristic Approaches
    def selectbestFirstSearch(self):
        '''Select the depth-first search algorithm'''
        self.status_label.setText('Best-first Search')
        self.selected_algorithm = self.bestFirstSearch

    def bestFirstSearch(self):
        '''best-first search: heuristic is shortest straight line distance to each city (priority)'''
        self.outputbox.clear()
        # Memory usage before the operation in KiB
        start_mem = memory_usage()[0] * 1024

        start_time = time.perf_counter()
        
        start_city = self.start_dropdown.currentText()
        end_city = self.end_dropdown.currentText()

        graph = self.graph

        visited = []
        queue = [(self.calculate_distance(start_city, end_city), start_city, [])]
        while queue:
            (priority, city, path) = heapq.heappop(queue)
            if city == end_city:
                # The path is the route from the start city to the end city
                self.outputbox.setText("<b>Path:</b> " + ' -> '.join(path + [end_city]) + '\n')

                # Total time to run search algorithm
                end_time = time.perf_counter()
                total_time_seconds = end_time - start_time
                total_time_microseconds = round(total_time_seconds * 1000000, 3)
                self.outputbox.append(f"<b>Total time:</b> {total_time_microseconds} microseconds.")
                
                # Total distance traveled in KM
                route = path + [end_city]
                total_distance = self.calculate_total_distance(route)
                self.outputbox.append(f"<b>Total distance:</b> {total_distance} KM.")

                # Memory usage after the operation
                end_mem = memory_usage()[0] * 1024
                total_mem = end_mem - start_mem
                self.outputbox.append(f"<b>Total memory used:</b> {round(total_mem, 3)} KiB.")
                return visited

            if city not in visited:
                visited.append(city)
                path = path + [city]
                for neighbour in graph[city]:
                    if neighbour not in visited:
                        priority = self.calculate_distance(neighbour, end_city)
                        heapq.heappush(queue, (priority, neighbour, path))

    def selectaStarSearch(self):
        '''Select the depth-first search algorithm'''
        self.status_label.setText('A* Search')
        self.selected_algorithm = self.aStarSearch

    def aStarSearch(self):
        '''A* search: heuristic is shortest straight line distance to each city (priority).
           Sums the cost to reach the current node (g) and the heuristic cost from the current node to the goal (h)'''
        self.outputbox.clear()
        # Memory usage before the operation in KiB
        start_mem = memory_usage()[0] * 1024

        start_time = time.perf_counter()
        
        start_city = self.start_dropdown.currentText()
        end_city = self.end_dropdown.currentText()

        graph = self.graph

        visited = []
        queue = [(0, start_city, [])]
        while queue:
            (priority, city, path) = heapq.heappop(queue)
            if city == end_city:
                # The path is the route from the start city to the end city
                self.outputbox.setText("<b>Path:</b> " + ' -> '.join(path + [end_city]) + '\n')

                # Total time to run search algorithm
                end_time = time.perf_counter()
                total_time_seconds = end_time - start_time
                total_time_microseconds = round(total_time_seconds * 1000000, 3)
                self.outputbox.append(f"<b>Total time:</b> {total_time_microseconds} microseconds.")
                
                # Total distance traveled in KM
                route = path + [end_city]
                total_distance = self.calculate_total_distance(route)
                self.outputbox.append(f"<b>Total distance:</b> {total_distance} KM.")

                # Memory usage after the operation
                end_mem = memory_usage()[0] * 1024
                total_mem = end_mem - start_mem
                self.outputbox.append(f"<b>Total memory used:</b> {round(total_mem, 3)} KiB.")
                return visited

            if city not in visited:
                visited.append(city)
                path = path + [city]
                for neighbour in graph[city]:
                    if neighbour not in visited:
                        g = len(path)  # cost to reach the current node
                        h = self.calculate_distance(neighbour, end_city)  # heuristic cost
                        f = g + h  # priority
                        heapq.heappush(queue, (f, neighbour, path))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Application()
    window.show()
    sys.exit(app.exec_())