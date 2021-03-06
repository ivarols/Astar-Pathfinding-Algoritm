import pygame
import math
from queue import PriorityQueue

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Pathfinding Algoritm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 156, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


class NODE:

    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == RED

    def is_open(self):
        return self.color == ORANGE

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == GREEN

    def is_end(self):
        return self.color == TURQUOISE

    def is_node(self):
        return self.color == WHITE

    def reset(self):
        self.color = WHITE

    def make_closed(self):
        self.color = RED

    def make_open(self):
        self.color = ORANGE

    def make_barrier(self):
        self.color = BLACK

    def make_start(self):
        self.color = GREEN

    def make_end(self):
        self.color = TURQUOISE

    def make_path(self):
        self.color = PURPLE

    def make_neighbors(self, grid):

        self.neighbors = []

        if self.col > 0 and (grid[self.row][self.col - 1].is_node() or grid[self.row][self.col - 1].is_end()):
            self.neighbors.append(grid[self.row][self.col - 1])

        if self.col < self.total_rows - 1 and (grid[self.row][self.col + 1].is_node() or grid[self.row][self.col + 1].is_end()):
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.row > 0 and (grid[self.row - 1][self.col].is_node() or grid[self.row - 1][self.col].is_end()):
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.row < self.total_rows - 1 and (grid[self.row + 1][self.col].is_node() or grid[self.row + 1][self.col].is_end()):
            self.neighbors.append(grid[self.row + 1][self.col])

        if grid[0][0].is_node() and (grid[0][1] == grid[self.row][self.col] or grid[1][0] == grid[self.row][self.col]):
            self.neighbors.append(grid[0][0])


    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

def heuristic(curr_p,end_p):
    x1, y1 = curr_p
    x2, y2 = end_p
    return abs(x2 - x1) + abs(y2 - y1)

def construct_path(passed_nodes, current, draw):
    while current in passed_nodes:
        current = passed_nodes[current]
        current.make_path()
        draw()


def algoritm(draw, start, end, grid):
    counter = 0
    open_set = PriorityQueue()
    passed_nodes = {}
    open_set.put((0, counter, start))
    g_score = {}
    f_score = {}

    for row in grid:
        for node in row:
            g_score.update({node: float("inf")})
    g_score[start] = 0

    for row in grid:
        for node in row:
            f_score.update({node: float("inf")})
    f_score[start] = heuristic(start.get_pos(), end.get_pos())

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        if current == end:
            construct_path(passed_nodes, current, draw)
            break

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                passed_nodes[neighbor] = current
                counter += 1
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor.get_pos(), end.get_pos())
                open_set.put((f_score[neighbor], counter, neighbor))
                neighbor.make_open()
                current.make_closed()

                start.make_start()
                end.make_end()

        draw()

def make_grid(rows, width):
    grid = []
    node_gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            Node = NODE(i, j, node_gap, rows)
            grid[i].append(Node)
    return grid


def draw_grid(win, rows, width):
    node_size = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * node_size), (width, i * node_size))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * node_size, 0), (j * node_size, width))


def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for Node in row:
            Node.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos, rows, width):
    node_gap = width // rows
    y, x = pos

    row = y // node_gap
    col = x // node_gap

    return row, col


def main(win, width):
    ROWS = 50

    grid = make_grid(ROWS, width)

    start = None
    end = None

    run = True
    started = False

    while run:
        draw(win, grid, ROWS, WIDTH)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started:
                continue

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                NODE = grid[row][col]

                if not start:
                    start = NODE
                    start.make_start()
                elif not end and NODE != start:
                    end = NODE
                    end.make_end()

                elif NODE != start and NODE != end:
                    NODE.make_barrier()

            if pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                NODE = grid[row][col]
                NODE.reset()

                if NODE == start:
                    start = None
                if NODE == end:
                    end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not started:
                    for row in grid:
                        for NODE in row:
                            NODE.make_neighbors(grid)

                algoritm(lambda: draw(WIN, grid, ROWS, width), start, end, grid)

    pygame.quit()


main(WIN, WIDTH)
