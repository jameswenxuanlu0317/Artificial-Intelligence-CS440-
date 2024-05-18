# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)
import queue
import heapq
def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    path = []
    parent = {}
    visited = set()
    q = queue.Queue()

    # Enqueue the start position and mark it as visited.
    q.put(maze.start)
    visited.add(maze.start)
    parent[maze.start] = None

    # Continue searching until the queue is empty or the first waypoint is found.
    while not q.empty():
        current = q.get()

        if current == maze.waypoints[0]:
            break

        # Explore the neighbors of the current position.
        for neighbor in maze.neighbors(*current):
            if neighbor not in visited:
                parent[neighbor] = current
                q.put(neighbor)
                visited.add(neighbor)

    # Reconstruct the path by following the parent pointers.
    x = maze.waypoints[0]
    while x is not None:
        path.append(x)
        x = parent[x]

    # Reverse the path to get it from start to end.
    path.reverse()

    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    start = None
    waypoint = None
    for x in range(maze.size.y):
        for y in range(maze.size.x):
            if maze[x, y] == maze.legend.start:
                start = (x, y)
            if maze[x, y] == maze.legend.waypoint:
                waypoint = (x, y)

    def heuristic(pos1, pos2) -> int:
        """
        Manhattan distance heuristic.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # Initialize the priority queue with the start position and the corresponding cost and heuristic value.
    priority_queue = [(heuristic(start, waypoint), 0, [start])]
    visited = set()

    # Continue searching until the priority queue is empty or the waypoint is found.
    while priority_queue:
        _, cost, path = heapq.heappop(priority_queue)
        node = path[-1]

        if node == waypoint:
            return path

        if node not in visited:
            visited.add(node)

            # Explore the neighbors of the current position.
            for neighbor in maze.neighbors(*node):
                if neighbor not in visited:
                    new_cost = cost + 1
                    new_path = list(path)
                    new_path.append(neighbor)
                    new_priority = new_cost + heuristic(neighbor, waypoint)
                    heapq.heappush(priority_queue, (new_priority, new_cost, new_path))

    # No path found.
    return []

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
