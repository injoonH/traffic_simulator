# 2020.09.01
# 에이전트 학습용 환경

import numpy as np
import random


class Car:
    def __init__(self, lc, gl):
        self.loc = lc
        self.goal = gl
        self.lane = None
        self.direct = None
        self.nav = []
        self.time = 0


class Environment:
    def __init__(self, row=2, col=3, lane=4, len=30, max_step=1_000_000):
        self.ROW = row
        self.COL = col
        self.LANE = lane
        self.LEN = len

        self.board = np.zeros((len * (row + 1) + row * lane * 2, len * (col + 1) + col * lane * 2, 8), dtype=np.int32)
        self.intersection = np.zeros((row, col))

        self.layer = {"loc": 0, "next": 1, "check": 2, "row": 3, "col": 4, "lane": 5, "light": 6, "goal": 7}
        self.direct = {0: (-1, 0), 1: (-1, 1), 2: (0, 1), 3: (1, 1), 4: (1, 0), 5: (1, -1), 6: (0, -1), 7: (-1, -1)}
        self.way = {"str": 0, "lft": 1, "rgt": 2, "U": 3}
        self.light = {"lr": 0, "ud": 1, "tlr": 2, "tud": 3, "stp": 4}

        # 매개변수로 받아도 되나, 중요한 것 아님.
        self.enter_prob = np.random.rand(2 * (row + col)) / 6
        self.to_where_prob = np.random.rand(2 * (row + col), 2 * (row + col))
        self.to_where_prob = self.to_where_prob / self.to_where_prob.sum(axis=1)[:, None]

        self.start = []
        self.goal = []

        self.cars = []
        self.indices = []
        self.out_indices = []

        self.max_step = max_step
        self.cur_step = 0

        for i in range(row):
            r = len * (i + 1) + lane * 2 * i
            self.board[r: r + lane * 2, :, self.layer["row"]] = i
            self.board[r: r + lane, 0: len, self.layer["col"]] = 0
            self.board[r + lane: r + lane * 2, self.board.shape[1] - len: self.board.shape[1], self.layer["col"]] = col - 1
            self.board[r: r + lane, :, self.layer["light"]] = 1
            self.board[r + lane: r + lane * 2, :, self.layer["light"]] = 3
            for j in range(lane):
                self.start.append([r + j, self.board.shape[1] - 1])
                self.goal.append([r + j, 0])
                self.board[r + j, 0, self.layer["goal"]] = 1
            for j in range(lane, lane * 2):
                self.start.append([r + j, 0])
                self.goal.append([r + j, self.board.shape[1] - 1])
                self.board[r + j, self.board.shape[1] - 1, self.layer["goal"]] = 1
        for i in range(col):
            c = len * (i + 1) + lane * 2 * i
            self.board[:, c: c + lane * 2, self.layer["col"]] = i
            self.board[0: len, c + lane: c + lane * 2, self.layer["row"]] = 0
            self.board[self.board.shape[0] - len: self.board.shape[0], c: c + lane, self.layer["row"]] = row - 1
            self.board[:, c: c + lane, self.layer["light"]] = 0
            self.board[:, c + lane: c + lane * 2, self.layer["light"]] = 2
            for j in range(lane):
                self.start.append([0, c + j])
                self.goal.append([self.board.shape[0] - 1, c + j])
                self.board[self.board.shape[0] - 1, c + j, self.layer["goal"]] = 1
            for j in range(lane, lane * 2):
                self.start.append([self.board.shape[0] - 1, c + j])
                self.goal.append([0, c + j])
                self.board[0, c + j, self.layer["goal"]] = 1
        for i in range(lane):
            for j in range(row):
                self.board[len * (j + 1) + lane * 2 * j + i, :, self.layer["lane"]] = lane - i
                self.board[(len + lane * 2) * (j + 1) - 1 - i, :, self.layer["lane"]] = lane - i
            for j in range(col):
                self.board[:, len * (j + 1) + lane * 2 * j + i, self.layer["lane"]] = lane - i
                self.board[:, (len + lane * 2) * (j + 1) - 1 - i, self.layer["lane"]] = lane - i
        for i in range(row):
            for j in range(col):
                r = len * (i + 1) + lane * 2 * i
                c = len * (j + 1) + lane * 2 * j
                self.board[r - 1, c: c + lane, self.layer["check"]] = 1
                self.board[r: r + lane, c + lane * 2, self.layer["check"]] = 1
                self.board[r + lane * 2, c + lane: c + lane * 2, self.layer["check"]] = 1
                self.board[r + lane: r + lane * 2, c - 1, self.layer["check"]] = 1
                self.board[r - len: r, c: c + lane, self.layer["row"]] = i
                self.board[r + lane * 2: r + lane * 2 + len, c + lane: c + lane * 2, self.layer["row"]] = i
                self.board[r: r + lane, c + lane * 2: c + lane * 2 + len, self.layer["col"]] = j
                self.board[r + lane: r + lane * 2, c - len: c, self.layer["col"]] = j
                self.board[r: r + lane * 2, c: c + lane * 2, self.layer["light"]] = -1

    def get_way(self, vector1, vector2):
        dtm = vector1[0] * vector2[1] - vector1[1] * vector2[0]
        if dtm == 1:
            return self.way["lft"]
        if dtm == -1:
            return self.way["rgt"]
        if vector1[0] == vector2[0] and vector1[1] == vector2[1]:
            return self.way["str"]
        return self.way["U"]

    def set_nav(self, car):
        dr = self.board[car.goal[0], car.goal[1], self.layer["row"]] - self.board[car.loc[0], car.loc[1], self.layer["row"]]
        dc = self.board[car.goal[0], car.goal[1], self.layer["col"]] - self.board[car.loc[0], car.loc[1], self.layer["col"]]
        gl = self.board[car.goal[0], car.goal[1], self.layer["light"]]
        arr = []
        for i in range(int(abs(dr))):
            arr.append((dr / abs(dr), 0))
        for i in range(int(abs(dc))):
            arr.append((0, dc / abs(dc)))
        random.shuffle(arr)
        if len(arr) > 0:
            car.nav = []
            car.nav.append(self.get_way(self.direct[car.direct], arr[0]))
            for i in range(len(arr) - 1):
                car.nav.append(self.get_way(arr[i], arr[i + 1]))
            car.nav.append(self.get_way(arr[-1], self.direct[(gl * 2 + 4) % 8]))
        else:
            car.nav = [self.get_way(self.direct[car.direct], self.direct[(gl * 2 + 4) % 8])]

    def refresh_nav(self, car, wy):
        car.nav = [wy]
        if wy == self.way["str"]:
            d = car.direct
        elif wy == self.way["lft"]:
            d = (car.direct + 6) % 8
        elif wy == self.way["rgt"]:
            d = (car.direct + 2) % 8
        else:
            d = (car.direct + 4) % 8
        r = self.board[car.loc[0], car.loc[1], self.layer["row"]] + self.direct[d][0]
        c = self.board[car.loc[0], car.loc[1], self.layer["col"]] + self.direct[d][1]
        if 0 <= r < self.ROW and 0 <= c < self.COL:
            dr = self.board[car.goal[0], car.goal[1], self.layer["row"]] - r
            dc = self.board[car.goal[0], car.goal[1], self.layer["col"]] - c
            gl = self.board[car.goal[0], car.goal[1], self.layer["light"]]
            arr = []
            for i in range(int(abs(dr))):
                arr.append((dr / abs(dr), 0))
            for i in range(int(abs(dc))):
                arr.append((0, dc / abs(dc)))
            random.shuffle(arr)
            if len(arr) > 0:
                car.nav.append(self.get_way(self.direct[d], arr[0]))
                for i in range(len(arr) - 1):
                    car.nav.append(self.get_way(arr[i], arr[i + 1]))
                car.nav.append(self.get_way(arr[-1], self.direct[(gl * 2 + 4) % 8]))
            else:
                car.nav.append(self.get_way(self.direct[d], self.direct[(gl * 2 + 4) % 8]))

    def refresh_lane(self, car):
        if len(car.nav) == 0:
            car.lane = self.board[car.goal[0], car.goal[1], self.layer["lane"]]
        else:
            w = car.nav[0]
            if w == self.way["str"]:
                ln = self.board[car.loc[0], car.loc[1], self.layer["lane"]]
                if ln == 1 and self.LANE > 1:
                    car.lane = 2
                else:
                    car.lane = ln
            elif w == self.way["rgt"]:
                car.lane = self.LANE
            else:
                car.lane = 1

    def refresh_direct(self, car):
        lgt = self.board[car.loc[0], car.loc[1], self.layer["light"]]
        if lgt == 0:
            car.direct = 4
        elif lgt == 1:
            car.direct = 6
        elif lgt == 2:
            car.direct = 0
        else:
            car.direct = 2

    def reset(self):
        self.board[:, :, self.layer["loc"]] = 0
        self.board[:, :, self.layer["next"]] = 0
        self.intersection[:, :] = 0

        self.cars = []
        self.indices = []
        self.out_indices = []

        self.cur_step = 0

        return np.zeros((self.ROW, self.COL, 4, 2), dtype=int)

    def frame(self):
        # 자동차 유입
        rand_prob = np.random.rand(2 * (self.ROW + self.COL))
        arr, = np.where(rand_prob < self.enter_prob)
        for s in arr:
            g = np.random.choice(list(range(2 * (self.ROW + self.COL))), p=self.to_where_prob[s, :])
            i = s * self.LANE + random.randint(0, self.LANE - 1)
            j = g * self.LANE + random.randint(0, self.LANE - 1)
            if self.board[self.start[i][0], self.start[i][1], self.layer["loc"]] != 1:
                c = Car(list(self.start[i]), list(self.goal[j]))
                self.refresh_direct(c)
                self.set_nav(c)
                self.refresh_lane(c)
                self.board[self.start[i][0], self.start[i][1], self.layer["loc"]] = 1
                self.cars.append(c)

        # step1
        self.board[:, :, self.layer["next"]] = 0
        self.indices = list(range(len(self.cars)))
        self.out_indices = []
        for idx in range(len(self.cars) - 1, -1, -1):
            c = self.cars[idx]
            if self.board[c.loc[0], c.loc[1], self.layer["check"]] == 1:
                if self.board[c.loc[0], c.loc[1], self.layer["lane"]] == c.lane:
                    if c.nav[0] == self.way["rgt"]:
                        if self.board[
                            c.loc[0] + self.direct[(c.direct + 1) % 8][0], c.loc[1] + self.direct[(c.direct + 1) % 8][
                                1], self.layer["loc"]] == 1:
                            c.time += 1
                            self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                            del self.indices[idx]
                    elif c.nav[0] == self.way["str"]:
                        row = self.board[c.loc[0], c.loc[1], self.layer["row"]]
                        col = self.board[c.loc[0], c.loc[1], self.layer["col"]]
                        b1 = self.board[c.loc[0], c.loc[1], self.layer["light"]] % 2 == 0 and self.intersection[
                            row, col] == self.light["ud"]
                        b2 = self.board[c.loc[0], c.loc[1], self.layer["light"]] % 2 == 1 and self.intersection[
                            row, col] == self.light["lr"]
                        if b1 or b2:
                            if self.board[
                                c.loc[0] + self.direct[c.direct][0], c.loc[1] + self.direct[c.direct][1], self.layer[
                                    "loc"]] == 1:
                                c.time += 1
                                self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                                del self.indices[idx]
                        else:
                            c.time += 1
                            self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                            del self.indices[idx]
                    else:
                        row = self.board[c.loc[0], c.loc[1], self.layer["row"]]
                        col = self.board[c.loc[0], c.loc[1], self.layer["col"]]
                        b1 = self.board[c.loc[0], c.loc[1], self.layer["light"]] % 2 == 0 and self.intersection[
                            row, col] == self.light["tud"]
                        b2 = self.board[c.loc[0], c.loc[1], self.layer["light"]] % 2 == 1 and self.intersection[
                            row, col] == self.light["tlr"]
                        if b1 or b2:
                            if c.nav[0] == self.way["lft"]:
                                if self.board[c.loc[0] + self.direct[(c.direct + 7) % 8][0], c.loc[1] + self.direct[
                                    (c.direct + 7) % 8][1], self.layer["loc"]] == 1:
                                    c.time += 1
                                    self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                                    del self.indices[idx]
                            else:
                                if self.board[c.loc[0] + self.direct[(c.direct + 6) % 8][0], c.loc[1] + self.direct[
                                    (c.direct + 6) % 8][1], self.layer["loc"]] == 1:
                                    c.time += 1
                                    self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                                    del self.indices[idx]
                        else:
                            c.time += 1
                            self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                            del self.indices[idx]
                else:
                    if self.board[c.loc[0], c.loc[1], self.layer["lane"]] == 1:
                        row = self.board[c.loc[0], c.loc[1], self.layer["row"]]
                        col = self.board[c.loc[0], c.loc[1], self.layer["col"]]
                        b1 = self.board[c.loc[0], c.loc[1], self.layer["light"]] % 2 == 0 and self.intersection[
                            row, col] == self.light["tud"]
                        b2 = self.board[c.loc[0], c.loc[1], self.layer["light"]] % 2 == 1 and self.intersection[
                            row, col] == self.light["tlr"]
                        if b1 or b2:
                            if random.random() < 0.5:
                                self.refresh_nav(c, self.way["lft"])
                                self.refresh_lane(c)
                                if self.board[c.loc[0] + self.direct[(c.direct + 7) % 8][0], c.loc[1] + self.direct[
                                    (c.direct + 7) % 8][1], self.layer["loc"]] == 1:
                                    c.time += 1
                                    self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                                    del self.indices[idx]
                            else:
                                self.refresh_nav(c, self.way["U"])
                                self.refresh_lane(c)
                                if self.board[c.loc[0] + self.direct[(c.direct + 6) % 8][0], c.loc[1] + self.direct[
                                    (c.direct + 6) % 8][1], self.layer["loc"]] == 1:
                                    c.time += 1
                                    self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                                    del self.indices[idx]
                        else:
                            c.time += 1
                            self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                            del self.indices[idx]
                    else:
                        if random.random() < 0.5:
                            self.refresh_nav(c, self.way["str"])
                            self.refresh_lane(c)
                            row = self.board[c.loc[0], c.loc[1], self.layer["row"]]
                            col = self.board[c.loc[0], c.loc[1], self.layer["col"]]
                            b1 = self.board[c.loc[0], c.loc[1], self.layer["light"]] % 2 == 0 and self.intersection[
                                row, col] == self.light["ud"]
                            b2 = self.board[c.loc[0], c.loc[1], self.layer["light"]] % 2 == 1 and self.intersection[
                                row, col] == self.light["lr"]
                            if b1 or b2:
                                if self.board[c.loc[0] + self.direct[c.direct][0], c.loc[1] + self.direct[c.direct][1],
                                              self.layer["loc"]] == 1:
                                    c.time += 1
                                    self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                                    del self.indices[idx]
                            else:
                                c.time += 1
                                self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                                del self.indices[idx]
                        else:
                            self.refresh_nav(c, self.way["rgt"])
                            self.refresh_lane(c)
                            if self.board[c.loc[0] + self.direct[(c.direct + 1) % 8][0], c.loc[1] + self.direct[
                                (c.direct + 1) % 8][1], self.layer["loc"]] == 1:
                                c.time += 1
                                self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                                del self.indices[idx]
            else:
                if self.board[c.loc[0], c.loc[1], self.layer["light"]] == -1:
                    if self.board[c.loc[0] + self.direct[c.direct][0], c.loc[1] + self.direct[c.direct][1], self.layer[
                        "loc"]] == 1:
                        c.time += 1
                        self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                        del self.indices[idx]
                else:
                    if self.board[c.loc[0], c.loc[1], self.layer["lane"]] == c.lane:
                        if self.board[
                            c.loc[0] + self.direct[c.direct][0], c.loc[1] + self.direct[c.direct][1], self.layer[
                                "loc"]] == 1:
                            c.time += 1
                            self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                            del self.indices[idx]

        # step2
        random.shuffle(self.indices)
        for idx in range(len(self.indices) - 1, -1, -1):
            c = self.cars[self.indices[idx]]
            if self.board[c.loc[0], c.loc[1], self.layer["light"]] != -1 and self.board[
                c.loc[0], c.loc[1], self.layer["check"]] != 1:
                d = c.lane - self.board[c.loc[0], c.loc[1], self.layer["lane"]]
                if d != 0:
                    d = int(abs(d) / d)
                    row = c.loc[0] + self.direct[(c.direct + 8 + d) % 8][0]
                    col = c.loc[1] + self.direct[(c.direct + 8 + d) % 8][1]
                    if self.board[row, col, self.layer["next"]] != 1:
                        if self.board[row, col, self.layer["loc"]] == 1:
                            if random.random() < 0.5:
                                c.time += 1
                            else:
                                sr = c.loc[0] + self.direct[c.direct][0]
                                sc = c.loc[1] + self.direct[c.direct][1]
                                if self.board[sr, sc, self.layer["next"]] != 1 and self.board[
                                    sr, sc, self.layer["loc"]] != 1:
                                    c.loc[0] = sr
                                    c.loc[1] = sc
                                    if self.board[sr, sc, self.layer["goal"]] == 1:
                                        self.out_indices.append(self.indices[idx])
                                    else:
                                        c.time = 0
                                else:
                                    c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            if self.board[row, col, self.layer["goal"]] == 1:
                                self.out_indices.append(self.indices[idx])
                            else:
                                c.time = 0
                        self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
                        del self.indices[idx]

        # step3
        for idx in range(len(self.indices) - 1, -1, -1):
            c = self.cars[self.indices[idx]]
            if self.board[c.loc[0], c.loc[1], self.layer["check"]] == 1:
                if self.board[c.loc[0], c.loc[1], self.layer["lane"]] == c.lane:
                    if c.nav[0] == self.way["rgt"]:
                        row = c.loc[0] + self.direct[(c.direct + 1) % 8][0]
                        col = c.loc[1] + self.direct[(c.direct + 1) % 8][1]
                        if self.board[row, col, self.layer["next"]] == 1 or self.board[
                            row, col, self.layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            c.direct = (c.direct + 2) % 8
                            del c.nav[0]
                            self.refresh_lane(c)
                            if self.board[row, col, self.layer["goal"]] == 1:
                                self.out_indices.append(self.indices[idx])
                            else:
                                c.time = 0
                    elif c.nav[0] == self.way["str"]:
                        row = c.loc[0] + self.direct[c.direct][0]
                        col = c.loc[1] + self.direct[c.direct][1]
                        if self.board[row, col, self.layer["next"]] == 1 or self.board[
                            row, col, self.layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            if self.board[row, col, self.layer["goal"]] == 1:
                                self.out_indices.append(self.indices[idx])
                            else:
                                c.time = 0
                    elif c.nav[0] == self.way["lft"]:
                        row = c.loc[0] + self.direct[(c.direct + 7) % 8][0]
                        col = c.loc[1] + self.direct[(c.direct + 7) % 8][1]
                        if self.board[row, col, self.layer["next"]] == 1 or self.board[
                            row, col, self.layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            c.direct = (c.direct + 7) % 8
                            if self.board[row, col, self.layer["goal"]] == 1:
                                self.out_indices.append(self.indices[idx])
                            else:
                                c.time = 0
                    else:
                        row = c.loc[0] + self.direct[(c.direct + 6) % 8][0]
                        col = c.loc[1] + self.direct[(c.direct + 6) % 8][1]
                        if self.board[row, col, self.layer["next"]] == 1 or self.board[
                            row, col, self.layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            c.direct = (c.direct + 4) % 8
                            del c.nav[0]
                            self.refresh_lane(c)
                            if self.board[row, col, self.layer["goal"]] == 1:
                                self.out_indices.append(self.indices[idx])
                            else:
                                c.time = 0
            else:
                if self.board[c.loc[0], c.loc[1], self.layer["light"]] == -1:
                    if c.direct % 2 == 1:
                        row = c.loc[0] + self.direct[c.direct][0]
                        col = c.loc[1] + self.direct[c.direct][1]
                        if self.board[row, col, self.layer["next"]] == 1 or self.board[
                            row, col, self.layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            if self.board[row, col, self.layer["light"]] != -1:
                                c.direct = (c.direct + 7) % 8
                                del c.nav[0]
                                self.refresh_lane(c)
                            if self.board[row, col, self.layer["goal"]] == 1:
                                self.out_indices.append(self.indices[idx])
                            else:
                                c.time = 0
                    else:
                        row = c.loc[0] + self.direct[c.direct][0]
                        col = c.loc[1] + self.direct[c.direct][1]
                        if self.board[row, col, self.layer["next"]] == 1 or self.board[
                            row, col, self.layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            if self.board[row, col, self.layer["light"]] != -1:
                                del c.nav[0]
                                self.refresh_lane(c)
                            if self.board[row, col, self.layer["goal"]] == 1:
                                self.out_indices.append(self.indices[idx])
                            else:
                                c.time = 0
                else:
                    row = c.loc[0] + self.direct[c.direct][0]
                    col = c.loc[1] + self.direct[c.direct][1]
                    if self.board[row, col, self.layer["next"]] == 1 or self.board[row, col, self.layer["loc"]] == 1:
                        c.time += 1
                    else:
                        c.loc[0] = row
                        c.loc[1] = col
                        if self.board[row, col, self.layer["goal"]] == 1:
                            self.out_indices.append(self.indices[idx])
                        else:
                            c.time = 0
            self.board[c.loc[0], c.loc[1], self.layer["next"]] = 1
        self.board[:, :, self.layer["loc"]] = self.board[:, :, self.layer["next"]]

        # step4
        self.out_indices.sort()
        for idx in range(len(self.out_indices) - 1, -1, -1):
            del self.cars[self.out_indices[idx]]

    def step(self, action, alpha=1, beta=40):
        self.intersection = np.copy(action)
        for i in range(self.LANE * 2):
            self.frame()
        self.intersection[:, :] = self.light["stp"]
        for i in range(self.LANE * 2):
            self.frame()

        state = np.zeros((self.ROW, self.COL, 4, 2))
        done = False
        for car in self.cars:
            if self.board[car.loc[0], car.loc[1], self.layer["light"]] == -1:
                continue
            if car.time > 0:
                r = self.board[car.loc[0], car.loc[1], self.layer["row"]]
                c = self.board[car.loc[0], car.loc[1], self.layer["col"]]
                lgt = self.board[car.loc[0], car.loc[1], self.layer["light"]]
                state[r, c, lgt, 0] += 1
                if self.board[car.loc[0], car.loc[1], self.layer["check"]] == 1:
                    state[r, c, lgt, 1] += car.time
                    if car.time >= beta * self.LANE * 4:
                        done = True

        self.cur_step += 1
        return state, state[:, :, :, 0].sum() + alpha * state[:, :, :, 1].sum(), self.cur_step >= self.max_step or done


env = Environment(row=3, col=4, lane=40, max_step=100)
stp = 0

while True:
    stp += 1
    print("step", stp, "= frame", stp * env.LANE * 4)
    stt, rwd, dn = env.step(np.random.randint(4, size=(env.ROW, env.COL)))
    # print("state", stt)
    print("reward", rwd, "\n")
    if dn:
        break
