# 2020.08.31
# 시각화용 환경

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

ROW, COL = 2, 3
LANE = 4
LEN = 30

layer = {"loc": 0, "next": 1, "check": 2, "row": 3, "col": 4, "lane": 5, "light": 6, "goal": 7}
board = np.zeros((LEN * (ROW + 1) + ROW * LANE * 2,
                  LEN * (COL + 1) + COL * LANE * 2, 8), dtype=np.int32)
# print("shape", board.shape)
for i in range(ROW):
    r = LEN * (i + 1) + LANE * 2 * i
    board[r: r + LANE * 2, :, layer["row"]] = i
    board[r: r + LANE, 0: LEN, layer["col"]] = 0
    board[r + LANE: r + LANE * 2, board.shape[1] - LEN: board.shape[1], layer["col"]] = COL - 1
    board[r: r + LANE, :, layer["light"]] = 1
    board[r + LANE: r + LANE * 2, :, layer["light"]] = 3
for i in range(COL):
    c = LEN * (i + 1) + LANE * 2 * i
    board[:, c: c + LANE * 2, layer["col"]] = i
    board[0: LEN, c + LANE: c + LANE * 2, layer["row"]] = 0
    board[board.shape[0] - LEN: board.shape[0], c: c + LANE, layer["row"]] = ROW - 1
    board[:, c: c + LANE, layer["light"]] = 0
    board[:, c + LANE: c + LANE * 2, layer["light"]] = 2
for i in range(LANE):
    for j in range(ROW):
        board[LEN * (j + 1) + LANE * 2 * j + i, :, layer["lane"]] = LANE - i
        board[(LEN + LANE * 2) * (j + 1) - 1 - i, :, layer["lane"]] = LANE - i
    for j in range(COL):
        board[:, LEN * (j + 1) + LANE * 2 * j + i, layer["lane"]] = LANE - i
        board[:, (LEN + LANE * 2) * (j + 1) - 1 - i, layer["lane"]] = LANE - i
for i in range(ROW):
    for j in range(COL):
        r = LEN * (i + 1) + LANE * 2 * i
        c = LEN * (j + 1) + LANE * 2 * j

        # 신호 확인 좌표 (layer 2)
        board[r - 1, c: c + LANE, layer["check"]] = 1
        board[r: r + LANE, c + LANE * 2, layer["check"]] = 1
        board[r + LANE * 2, c + LANE: c + LANE * 2, layer["check"]] = 1
        board[r + LANE: r + LANE * 2, c - 1, layer["check"]] = 1

        # 행 (layer 3)
        board[r - LEN: r, c: c + LANE, layer["row"]] = i
        board[r + LANE * 2: r + LANE * 2 + LEN, c + LANE: c + LANE * 2, layer["row"]] = i

        # 열 (layer 4)
        board[r: r + LANE, c + LANE * 2: c + LANE * 2 + LEN, layer["col"]] = j
        board[r + LANE: r + LANE * 2, c - LEN: c, layer["col"]] = j

        # 신호등 (layer 6)
        board[r: r + LANE * 2, c: c + LANE * 2, layer["light"]] = -1

# start, goal list 생성
start = []
goal = []
for i in range(ROW):
    r = LEN * (i + 1) + LANE * 2 * i
    for j in range(LANE):
        start.append([r + j, board.shape[1] - 1])
        goal.append([r + j, 0])
        board[r + j, 0, layer["goal"]] = 1
    for j in range(LANE, LANE * 2):
        start.append([r + j, 0])
        goal.append([r + j, board.shape[1] - 1])
        board[r + j, board.shape[1] - 1, layer["goal"]] = 1
for i in range(COL):
    c = LEN * (i + 1) + LANE * 2 * i
    for j in range(LANE):
        start.append([0, c + j])
        goal.append([board.shape[0] - 1, c + j])
        board[board.shape[0] - 1, c + j, layer["goal"]] = 1
    for j in range(LANE, LANE * 2):
        start.append([board.shape[0] - 1, c + j])
        goal.append([0, c + j])
        board[0, c + j, layer["goal"]] = 1

direct = {0: (-1, 0),   # 북
          1: (-1, 1),   # 북동
          2: (0, 1),    # 동
          3: (1, 1),    # 남동
          4: (1, 0),    # 남
          5: (1, -1),   # 남서
          6: (0, -1),   # 서
          7: (-1, -1)}  # 북서
way = {"str": 0, "lft": 1, "rgt": 2, "U": 3}
light = {"lr": 0, "ud": 1, "tlr": 2, "tud": 3, "stp": 4}
intersection = np.zeros((ROW, COL))

cars = []                   # 자동차 담는 list
out_indices = []            # 도시 탈출한 자동차 인덱스 담는 list
indices = []

# 도시 유입 확률 - 지금은 랜덤이지만 사용자 지정으로 만들 수 있음
enter_prob = np.random.rand(2 * (ROW + COL)) / 6

# 생성지 - 도착지 확률
to_where_prob = np.random.rand(2 * (ROW + COL), 2 * (ROW + COL))
to_where_prob = to_where_prob / to_where_prob.sum(axis=1)[:, None]


class Car:
    def __init__(self, lc, gl):
        self.loc = lc       # 현 좌표
        self.goal = gl      # 목적지 좌표
        self.lane = None    # 목표 차선
        self.direct = None  # direct{} key 저장
        self.nav = []       # 이상적 way{} value list
        self.time = 0       # 정차 시간


# (현 방향, 나중 방향) 받아 way 반환
def get_way(vector1, vector2):
    dtm = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    if dtm == 1:
        return way["lft"]
    if dtm == -1:
        return way["rgt"]
    if vector1[0] == vector2[0] and vector1[1] == vector2[1]:
        return way["str"]
    return way["U"]


# nav 설정
def set_nav(car):
    dr = board[car.goal[0], car.goal[1], layer["row"]] - board[car.loc[0], car.loc[1], layer["row"]]
    dc = board[car.goal[0], car.goal[1], layer["col"]] - board[car.loc[0], car.loc[1], layer["col"]]
    gl = board[car.goal[0], car.goal[1], layer["light"]]

    arr = []

    for i in range(int(abs(dr))):
        arr.append((dr / abs(dr), 0))
    for i in range(int(abs(dc))):
        arr.append((0, dc / abs(dc)))

    random.shuffle(arr)

    if len(arr) > 0:
        car.nav = []
        car.nav.append(get_way(direct[car.direct], arr[0]))
        for i in range(len(arr) - 1):
            car.nav.append(get_way(arr[i], arr[i + 1]))
        car.nav.append(get_way(arr[-1], direct[(gl * 2 + 4) % 8]))
    else:
        car.nav = [get_way(direct[car.direct], direct[(gl * 2 + 4) % 8])]


# nav 새로고침
def refresh_nav(car, wy):
    car.nav = [wy]

    if wy == way["str"]:
        d = car.direct
    elif wy == way["lft"]:
        d = (car.direct + 6) % 8
    elif wy == way["rgt"]:
        d = (car.direct + 2) % 8
    else:
        d = (car.direct + 4) % 8

    r = board[car.loc[0], car.loc[1], layer["row"]] + direct[d][0]
    c = board[car.loc[0], car.loc[1], layer["col"]] + direct[d][1]

    if 0 <= r < ROW and 0 <= c < COL:
        dr = board[car.goal[0], car.goal[1], layer["row"]] - r
        dc = board[car.goal[0], car.goal[1], layer["col"]] - c
        gl = board[car.goal[0], car.goal[1], layer["light"]]

        arr = []

        for i in range(int(abs(dr))):
            arr.append((dr / abs(dr), 0))
        for i in range(int(abs(dc))):
            arr.append((0, dc / abs(dc)))

        random.shuffle(arr)

        if len(arr) > 0:
            car.nav.append(get_way(direct[d], arr[0]))
            for i in range(len(arr) - 1):
                car.nav.append(get_way(arr[i], arr[i + 1]))
            car.nav.append(get_way(arr[-1], direct[(gl * 2 + 4) % 8]))
        else:
            car.nav.append(get_way(direct[d], direct[(gl * 2 + 4) % 8]))


# lane 새로고침
def refresh_lane(car):
    if len(car.nav) == 0:
        car.lane = board[car.goal[0], car.goal[1], layer["lane"]]
    else:
        w = car.nav[0]
        if w == way["str"]:
            ln = board[car.loc[0], car.loc[1], layer["lane"]]
            if ln == 1 and LANE > 1:
                car.lane = 2
            else:
                car.lane = ln
        elif w == way["rgt"]:
            car.lane = LANE
        else:
            car.lane = 1


# direct 새로고침
def refresh_direct(car):
    lgt = board[car.loc[0], car.loc[1], layer["light"]]
    if lgt == 0:
        car.direct = 4  # 남
    elif lgt == 1:
        car.direct = 6  # 서
    elif lgt == 2:
        car.direct = 0  # 북
    else:
        car.direct = 2  # 동


# 자동차 생성
def generate_car(srt, gl):
    c = Car(srt, gl)

    refresh_direct(c)
    set_nav(c)
    refresh_lane(c)

    board[srt[0], srt[1], layer["loc"]] = 1

    cars.append(c)


def step1():
    board[:, :, layer["next"]] = 0

    global out_indices
    global indices

    out_indices = []
    indices = list(range(len(cars)))

    for idx in range(len(cars) - 1, -1, -1):
        c = cars[idx]
        # print(idx, "번째 자동차 step1 계산 시작하겠음.")

        # 신호 확인 좌표?
        if board[c.loc[0], c.loc[1], layer["check"]] == 1:
            # 원하던 차선?
            if board[c.loc[0], c.loc[1], layer["lane"]] == c.lane:
                # 우회전 예정?
                if c.nav[0] == way["rgt"]:
                    # 우회전 방향에 다른 차 존재?
                    if board[c.loc[0] + direct[(c.direct + 1) % 8][0],
                             c.loc[1] + direct[(c.direct + 1) % 8][1],
                             layer["loc"]] == 1:
                        c.time += 1
                        board[c.loc[0], c.loc[1], layer["next"]] = 1
                        del indices[idx]
                # 직진 예정?
                elif c.nav[0] == way["str"]:
                    row = board[c.loc[0], c.loc[1], layer["row"]]
                    col = board[c.loc[0], c.loc[1], layer["col"]]
                    # 직진 신호 켜짐?
                    b1 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 0 and intersection[row, col] == light["ud"]
                    b2 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 1 and intersection[row, col] == light["lr"]
                    if b1 or b2:
                        # 직진 방향에 다른 차 존재?
                        if board[c.loc[0] + direct[c.direct][0],
                                 c.loc[1] + direct[c.direct][1],
                                 layer["loc"]] == 1:
                            c.time += 1
                            board[c.loc[0], c.loc[1], layer["next"]] = 1
                            del indices[idx]
                    else:
                        c.time += 1
                        board[c.loc[0], c.loc[1], layer["next"]] = 1
                        del indices[idx]
                else:
                    row = board[c.loc[0], c.loc[1], layer["row"]]
                    col = board[c.loc[0], c.loc[1], layer["col"]]
                    # 좌회전 신호 켜짐?
                    b1 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 0 and intersection[row, col] == light["tud"]
                    b2 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 1 and intersection[row, col] == light["tlr"]
                    if b1 or b2:
                        # 좌회전 예정?
                        if c.nav[0] == way["lft"]:
                            # 좌회전 방향에 다른 차 존재?
                            if board[c.loc[0] + direct[(c.direct + 7) % 8][0],
                                     c.loc[1] + direct[(c.direct + 7) % 8][1],
                                     layer["loc"]] == 1:
                                c.time += 1
                                board[c.loc[0], c.loc[1], layer["next"]] = 1
                                del indices[idx]
                        else:
                            # U턴 방향에 다른 차 존재?
                            if board[c.loc[0] + direct[(c.direct + 6) % 8][0],
                                     c.loc[1] + direct[(c.direct + 6) % 8][1],
                                     layer["loc"]] == 1:
                                c.time += 1
                                board[c.loc[0], c.loc[1], layer["next"]] = 1
                                del indices[idx]
                    else:
                        c.time += 1
                        board[c.loc[0], c.loc[1], layer["next"]] = 1
                        del indices[idx]
            else:  # 신호 확인 위치이지만 원하던 차선이 아니어서 네비게이션을 재설정하는 부분
                # 현재 1차선?
                if board[c.loc[0], c.loc[1], layer["lane"]] == 1:
                    # 좌회전 신호 켜짐?
                    row = board[c.loc[0], c.loc[1], layer["row"]]
                    col = board[c.loc[0], c.loc[1], layer["col"]]
                    b1 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 0 and intersection[row, col] == light["tud"]
                    b2 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 1 and intersection[row, col] == light["tlr"]
                    if b1 or b2:
                        # 50% 확률로 좌회전
                        if random.random() < 0.5:
                            refresh_nav(c, way["lft"])
                            refresh_lane(c)
                            # 좌회전 방향에 다른 차 존재?
                            if board[c.loc[0] + direct[(c.direct + 7) % 8][0],
                                     c.loc[1] + direct[(c.direct + 7) % 8][1],
                                     layer["loc"]] == 1:
                                c.time += 1
                                board[c.loc[0], c.loc[1], layer["next"]] = 1
                                del indices[idx]
                        else:
                            refresh_nav(c, way["U"])
                            refresh_lane(c)
                            # U턴 방향에 다른 차 존재?
                            if board[c.loc[0] + direct[(c.direct + 6) % 8][0],
                                     c.loc[1] + direct[(c.direct + 6) % 8][1],
                                     layer["loc"]] == 1:
                                c.time += 1
                                board[c.loc[0], c.loc[1], layer["next"]] = 1
                                del indices[idx]
                    else:
                        c.time += 1
                        board[c.loc[0], c.loc[1], layer["next"]] = 1
                        del indices[idx]
                else:
                    # 50% 확률로 직진
                    if random.random() < 0.5:
                        refresh_nav(c, way["str"])
                        refresh_lane(c)
                        # 직진 신호 켜짐?
                        row = board[c.loc[0], c.loc[1], layer["row"]]
                        col = board[c.loc[0], c.loc[1], layer["col"]]
                        b1 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 0 and intersection[row, col] == light["ud"]
                        b2 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 1 and intersection[row, col] == light["lr"]
                        if b1 or b2:
                            # 직진 방향에 다른 차 존재?
                            if board[c.loc[0] + direct[c.direct][0],
                                     c.loc[1] + direct[c.direct][1],
                                     layer["loc"]] == 1:
                                c.time += 1
                                board[c.loc[0], c.loc[1], layer["next"]] = 1
                                del indices[idx]
                        else:
                            c.time += 1
                            board[c.loc[0], c.loc[1], layer["next"]] = 1
                            del indices[idx]
                    else:
                        refresh_nav(c, way["rgt"])
                        refresh_lane(c)
                        # 우회전 방향에 다른 차 존재?
                        if board[c.loc[0] + direct[(c.direct + 1) % 8][0],
                                 c.loc[1] + direct[(c.direct + 1) % 8][1],
                                 layer["loc"]] == 1:
                            c.time += 1
                            board[c.loc[0], c.loc[1], layer["next"]] = 1
                            del indices[idx]
        else:
            # 교차로 한복판?
            if board[c.loc[0], c.loc[1], layer["light"]] == -1:
                # 이동 방향(직진, 좌회전)에 다른 차 존재?
                if board[c.loc[0] + direct[c.direct][0],
                         c.loc[1] + direct[c.direct][1],
                         layer["loc"]] == 1:
                    c.time += 1
                    board[c.loc[0], c.loc[1], layer["next"]] = 1
                    del indices[idx]
            else:
                # 원하던 차선?
                if board[c.loc[0], c.loc[1], layer["lane"]] == c.lane:
                    # 직진 방향에 다른 차 존재?
                    if board[c.loc[0] + direct[c.direct][0],
                             c.loc[1] + direct[c.direct][1],
                             layer["loc"]] == 1:
                        c.time += 1
                        board[c.loc[0], c.loc[1], layer["next"]] = 1
                        del indices[idx]


def step2():
    random.shuffle(indices)
    for idx in range(len(indices) - 1, -1, -1):
        c = cars[indices[idx]]
        # print(indices[idx], "번째 자동차 step2 계산 시작하겠음.")
        # 교차로 한복판이 아니고 신호 확인 좌표도 아니다?
        if board[c.loc[0], c.loc[1], layer["light"]] != -1 and board[c.loc[0], c.loc[1], layer["check"]] != 1:
            d = c.lane - board[c.loc[0], c.loc[1], layer["lane"]]
            # 원하던 차선이 아니다?
            if d != 0:
                d = int(abs(d) / d)
                row = c.loc[0] + direct[(c.direct + 8 + d) % 8][0]
                col = c.loc[1] + direct[(c.direct + 8 + d) % 8][1]
                # 끼어들 좌표에 위치할 예정인 차가 자신 뿐이다?
                if board[row, col, layer["next"]] != 1:
                    # 끼어들 좌표에 이미 차가 존재한다?
                    if board[row, col, layer["loc"]] == 1:
                        # 50% 확률로 정차
                        if random.random() < 0.5:
                            c.time += 1
                        else:
                            sr = c.loc[0] + direct[c.direct][0]
                            sc = c.loc[1] + direct[c.direct][1]
                            # 직진 방향에 차 존재하거나 존재할 예정이 아님?
                            if board[sr, sc, layer["next"]] != 1 and board[sr, sc, layer["loc"]] != 1:
                                c.loc[0] = sr
                                c.loc[1] = sc
                                # 도시 탈출?
                                if board[sr, sc, layer["goal"]] == 1:
                                    out_indices.append(indices[idx])
                                else:
                                    c.time = 0
                            else:
                                c.time += 1
                    else:
                        c.loc[0] = row
                        c.loc[1] = col
                        # 도시 탈출?
                        if board[row, col, layer["goal"]] == 1:
                            out_indices.append(indices[idx])
                        else:
                            c.time = 0
                    board[c.loc[0], c.loc[1], layer["next"]] = 1
                    del indices[idx]


def step3():
    for idx in range(len(indices) - 1, -1, -1):
        c = cars[indices[idx]]

        # 신호 확인 좌표?
        if board[c.loc[0], c.loc[1], layer["check"]] == 1:
            # 원하던 차선?
            if board[c.loc[0], c.loc[1], layer["lane"]] == c.lane:
                # 우회전 예정?
                if c.nav[0] == way["rgt"]:
                    row = c.loc[0] + direct[(c.direct + 1) % 8][0]
                    col = c.loc[1] + direct[(c.direct + 1) % 8][1]
                    # 우회전 방향에 차 존재하거나 존재할 예정?
                    if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                        c.time += 1
                    else:
                        c.loc[0] = row
                        c.loc[1] = col
                        c.direct = (c.direct + 2) % 8
                        del c.nav[0]
                        refresh_lane(c)
                        # 도시 탈출?
                        if board[row, col, layer["goal"]] == 1:
                            out_indices.append(indices[idx])
                        else:
                            c.time = 0
                # 직진 예정?
                elif c.nav[0] == way["str"]:
                    row = c.loc[0] + direct[c.direct][0]
                    col = c.loc[1] + direct[c.direct][1]
                    # 직진 방향에 차 존재하거나 존재할 예정?
                    if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                        c.time += 1
                    else:
                        c.loc[0] = row
                        c.loc[1] = col
                        if board[row, col, layer["goal"]] == 1:
                            out_indices.append(indices[idx])
                        else:
                            c.time = 0
                # 좌회전 예정?
                elif c.nav[0] == way["lft"]:
                    row = c.loc[0] + direct[(c.direct + 7) % 8][0]
                    col = c.loc[1] + direct[(c.direct + 7) % 8][1]
                    # 좌회전 방향에 차 존재하거나 존재할 예정?
                    if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                        c.time += 1
                    else:
                        c.loc[0] = row
                        c.loc[1] = col
                        c.direct = (c.direct + 7) % 8
                        if board[row, col, layer["goal"]] == 1:
                            out_indices.append(indices[idx])
                        else:
                            c.time = 0
                else:
                    row = c.loc[0] + direct[(c.direct + 6) % 8][0]
                    col = c.loc[1] + direct[(c.direct + 6) % 8][1]
                    if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                        c.time += 1
                    else:
                        c.loc[0] = row
                        c.loc[1] = col
                        c.direct = (c.direct + 4) % 8
                        del c.nav[0]
                        refresh_lane(c)
                        if board[row, col, layer["goal"]] == 1:
                            out_indices.append(indices[idx])
                        else:
                            c.time = 0
            else:
                # 신호 확인 위치이지만 원하던 차선이 아니어서 네비게이션을 재설정하는 부분
                pass
        else:
            # 교차로 한복판?
            if board[c.loc[0], c.loc[1], layer["light"]] == -1:
                # 좌회전 중?
                if c.direct % 2 == 1:
                    row = c.loc[0] + direct[c.direct][0]
                    col = c.loc[1] + direct[c.direct][1]
                    # 좌회전 방향에 차 존재하거나 존재할 예정?
                    if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                        c.time += 1
                    else:
                        c.loc[0] = row
                        c.loc[1] = col
                        # 교차로 벗어남?
                        if board[row, col, layer["light"]] != -1:
                            c.direct = (c.direct + 7) % 8
                            del c.nav[0]
                            refresh_lane(c)
                        # 도시 탈출?
                        if board[row, col, layer["goal"]] == 1:
                            out_indices.append(indices[idx])
                        else:
                            c.time = 0
                else:
                    row = c.loc[0] + direct[c.direct][0]
                    col = c.loc[1] + direct[c.direct][1]
                    # 직진 방향에 차 존재하거나 존재할 예정?
                    if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                        c.time += 1
                    else:
                        c.loc[0] = row
                        c.loc[1] = col
                        # 교차로 벗어남?
                        if board[row, col, layer["light"]] != -1:
                            del c.nav[0]
                            refresh_lane(c)
                        # 도시 탈출?
                        if board[row, col, layer["goal"]] == 1:
                            out_indices.append(indices[idx])
                        else:
                            c.time = 0
            else:
                row = c.loc[0] + direct[c.direct][0]
                col = c.loc[1] + direct[c.direct][1]
                # 직진 방향에 차 존재하거나 존재할 예정?
                if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                    c.time += 1
                else:
                    c.loc[0] = row
                    c.loc[1] = col
                    # 도시 탈출?
                    if board[row, col, layer["goal"]] == 1:
                        out_indices.append(indices[idx])
                    else:
                        c.time = 0

        board[c.loc[0], c.loc[1], layer["next"]] = 1

    board[:, :, layer["loc"]] = board[:, :, layer["next"]]


def step4():
    out_indices.sort()

    for idx in range(len(out_indices) - 1, -1, -1):
        del cars[out_indices[idx]]


# 자동차 도시 유입
def cars_enter():
    rand_prob = np.random.rand(2 * (ROW + COL))
    arr, = np.where(rand_prob < enter_prob)

    for s in arr:
        g = np.random.choice(list(range(2 * (ROW + COL))), p=to_where_prob[s, :])
        i = s * LANE + random.randint(0, LANE - 1)
        j = g * LANE + random.randint(0, LANE - 1)
        if board[start[i][0], start[i][1], layer["loc"]] != 1:
            generate_car(list(start[i]), list(goal[j]))


# 모든 교차로의 신호 변경
def change_signal(new_signal):
    global intersection
    # new_signal.shape = (ROW, COL)
    intersection = new_signal


# 보상 반환
def get_reward():
    # {0: 정차 차량, 1: 대기 시간}
    rwd = np.zeros((ROW, COL, 4, 2))
    for car in cars:
        if board[car.loc[0], car.loc[1], layer["light"]] == -1:
            continue
        if car.time > 0:
            r = board[car.loc[0], car.loc[1], layer["row"]]
            c = board[car.loc[0], car.loc[1], layer["col"]]
            lgt = board[car.loc[0], car.loc[1], layer["light"]]
            rwd[r, c, lgt, 0] += 1
            if board[car.loc[0], car.loc[1], layer["check"]] == 1:
                rwd[r, c, lgt, 1] += car.time
    return rwd


# 자동차 정보 print
def show_situation(n, m):
    print("\n============ {}th frame after step {} ============\n".format(n, m))
    print("indices", indices)
    print("out_indices", out_indices)
    print("len(cars)", len(cars), "\n")

    for i in range(len(cars)):
        print("c[{}].loc ".format(i), cars[i].loc)
        print("c[{}].goal".format(i), cars[i].goal)
        print("c[{}].lane".format(i), cars[i].lane, "from", board[cars[i].loc[0], cars[i].loc[1], layer["lane"]])
        print("c[{}].dir ".format(i), cars[i].direct)
        print("c[{}].nav ".format(i), cars[i].nav)
        print("c[{}].time".format(i), cars[i].time, "\n")

    print("\n==================================================\n")


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# 그림 범위 및 눈금 제거
ax.set_xlim(0, board.shape[1])
ax.set_ylim(0, board.shape[0])
plt.tick_params(axis="both", which="both", bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False)


palette = {"purple": "#BB8EFD", "blue": "#81A1FB", "black": "#000000", "dark_gray": "#707070", "red": "#FF2D2D", "light_gray": "#E0E0E0"}

# 교차로 그리기
for i in range(1, board.shape[0]):
    ax.plot([0, board.shape[1]], [i, i], color=palette["light_gray"])
for i in range(1, board.shape[1]):
    ax.plot([i, i], [0, board.shape[0]], color=palette["light_gray"])
for i in range(COL + 1):
    for j in range(ROW + 1):
        x = i * (LEN + LANE * 2)
        y = j * (LEN + LANE * 2)
        ax.add_patch(Rectangle((x, y), LEN, LEN, color=palette["dark_gray"], zorder=10))


# 자동차 크기, 색
mark_size = 6
mark_color = palette["black"]


scat = list()
scat.append(ax.plot([], [], marker="^", color=mark_color, markersize=mark_size, linestyle="None")[0])
scat.append(ax.plot([], [], marker=">", color=mark_color, markersize=mark_size, linestyle="None")[0])
scat.append(ax.plot([], [], marker="v", color=mark_color, markersize=mark_size, linestyle="None")[0])
scat.append(ax.plot([], [], marker="<", color=mark_color, markersize=mark_size, linestyle="None")[0])
scat.append(ax.plot([], [], marker="s", color=mark_color, markersize=mark_size, linestyle="None")[0])


def animate(frame_num):
    cars_enter()
    # show_situation(frame_num, 0)
    step1()
    # show_situation(frame_num, 1)
    step2()
    # show_situation(frame_num, 2)
    step3()
    # show_situation(frame_num, 3)
    step4()
    # show_situation(frame_num, 4)

    # 1. 신호 매번 랜덤하게
    # change_signal(np.random.randint(4, size=(ROW, COL)))

    # 2. 신호 변경 전에는 모든 신호등이 적색등을 보이도록
    # mod = 5
    # if frame_num % mod == 0:
    #     intersection[:, :] = (frame_num / mod) % mod
    # elif frame_num % mod == frame_num % mod - 1:
    #     intersection[:, :] = 4

    mod = LANE * 4
    if frame_num % mod == 0:
        intersection[:, :] = 4
    elif frame_num % mod == LANE * 2:
        change_signal(np.random.randint(4, size=(ROW, COL)))

    print("{}th signal\n".format(frame_num + 1), intersection, "\n")

    ax.set_title("{}th frame".format(frame_num + 1))

    # 자동차 위치 표시
    x = [[], [], [], [], []]
    y = [[], [], [], [], []]

    for c in cars:
        if c.direct == 0:
            x[0].append(c.loc[1] + 0.5)
            y[0].append(board.shape[0] - c.loc[0] - 0.5)
        elif c.direct == 2:
            x[1].append(c.loc[1] + 0.5)
            y[1].append(board.shape[0] - c.loc[0] - 0.5)
        elif c.direct == 4:
            x[2].append(c.loc[1] + 0.5)
            y[2].append(board.shape[0] - c.loc[0] - 0.5)
        elif c.direct == 6:
            x[3].append(c.loc[1] + 0.5)
            y[3].append(board.shape[0] - c.loc[0] - 0.5)
        else:
            x[4].append(c.loc[1] + 0.5)
            y[4].append(board.shape[0] - c.loc[0] - 0.5)

    for i in range(len(scat)):
        scat[i].set_data(x[i], y[i])

    return scat


animation = FuncAnimation(fig, animate, frames=10000, interval=10, repeat=False)
plt.get_current_fig_manager().full_screen_toggle()
plt.show()
