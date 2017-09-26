'''
Nils Napp
Sliding Probelm for AI-Class
'''

from slideproblem import *
import time
import heapq
from itertools import permutations, chain, product, combinations
from pprint import pprint
import operator
import sqlite3
print(sqlite3.sqlite_version)
class Searches:

    database3x3 = {}
    dSolver = {}
    vSolver = {}
    hSolver = {}
    d1Solver = {}
    dSwap = {}
    hSwap = {}
    vSwap = {}
    d1Swap = {}
    database = {}

    def createDSwap(self,s: State):
        size = s.boardSize
        for i in range(size * size):
            old = s.board[int(i/size)][i%size]
            new = s.board[i%size][int(i/size)]
            self.dSwap[old] = new

    def createVSwap(self, s: State):
        size = s.boardSize
        for i in range(size):
            for j in range(size):
                if i == 0 and j == 0:
                    self.vSwap[0] = 0
                    continue
                elif j == 0:
                    self.vSwap[(i*size)+j] = ((size-i)*(size))
                else:
                    self.vSwap[(i*size)+j] = ((size-i)*(size))-(size -j)

    def createHSwap(self, s: State):
        size = s.boardSize
        for i in range(1,size+1):
            for j in range(size):
                if i == 1 and j == 0:
                    self.hSwap[0] = 0
                    continue
                if i == 1:
                    self.hSwap[(i-1)*size +j] = (size*i -j)
                else:
                    self.hSwap[(i-1)*size +j] = (size*i -j -1)

    def createD1Swap(self, s: State):
        size = s.boardSize
        for i in range(size):
            for j in range(size):
                if i == 0 and j == 0:
                    self.d1Swap[0] = 0
                elif j == 0:
                    self.d1Swap[size*i] = (size*size) - i
                elif i == 3:
                    self.d1Swap[(size*i) + j] = size*(size-j)
                else:
                    self.d1Swap[(size*i) + j] = (size*(size-j))-(i+1)

    def vMirror(self, s0: State):
        s = State(s0)
        size = s.boardSize
        for i in range(int(size/2)):
            for j in range(size):
                x = s.board[i][j]
                s.board[i][j] = s.board[size-i-1][j]
                s.board[size-i-1][j] = x
        for i in range(size*size):
            s.board[int(i/size)][i%size] = self.vSwap[s.board[int(i/size)][i%size]]
        return s

    def hMirror(self, s0: State):
        s = State(s0)
        size = s.boardSize
        for i in range(size):
            for j in range(int(size/2)):
                x = s.board[i][j]
                s.board[i][j] = s.board[i][size-j-1]
                s.board[i][size-j-1] = x
        for i in range(size*size):
            s.board[int(i/size)][i%size] = self.hSwap[s.board[int(i/size)][i%size]]
        return s

    def dMirror(self, s0: State):
        s = State(s0)
        size = s.boardSize
        for i in range(size):
            for j in range(i,size):
                x = s.board[i][j]
                s.board[i][j] = s.board[j][i]
                s.board[j][i] = x
        for i in range (size * size):
            s.board[int(i/size)][i%size] = self.dSwap[s.board[int(i/size)][i%size]]
        return s

    def d1Mirror(self, s0: State):
        s = State(s0)
        size = s.boardSize
        for i in range(size):
            for j in range(size-i):
                x = s.board[i][j]
                s.board[i][j] = s.board[size-1-j][size-1-i]
                s.board[size-1-j][size-1-i] = x
        for i in range(size*size):
            s.board[int(i/size)][i%size] = self.d1Swap[s.board[int(i/size)][i%size]]
        return s

    def addAll(self, s, s0: State):

        d = self.dMirror(s0)
        v = self.vMirror(s0)
        h = self.hMirror(s0)
        d1 = self.d1Mirror(s0)

        v1 = self.vMirror(d)
        h1 = self.hMirror(d)
        d2 = self.d1Mirror(d)

        s.add(self.convert(s0).toTuple())
        s.add((self.convert(d).toTuple()))
        s.add((self.convert(v).toTuple()))
        s.add((self.convert(h).toTuple()))
        s.add((self.convert(d1).toTuple()))
        s.add((self.convert(v1).toTuple()))
        s.add((self.convert(h1).toTuple()))
        s.add((self.convert(d2).toTuple()))

    def addDB(self,s0: State,d,n0: Node):

        s = State(s0)
        m = self.dMirror(s)
        v = self.vMirror(s)
        h = self.hMirror(s)
        d1 = self.d1Mirror(s)

        v1 = self.vMirror(m)
        h1 = self.hMirror(m)
        d2 = self.d1Mirror(m)

        if (self.convert(s).toTuple() not in d and
            self.convert(m).toTuple() not in d and
            self.convert(v).toTuple() not in d and
            self.convert(h).toTuple() not in d and
            self.convert(d1).toTuple() not in d and
            self.convert(v1).toTuple() not in d and
            self.convert(h1).toTuple() not in d and
            self.convert(d2).toTuple() not in d):
            d[self.convert(s).toTuple()] = n0.cost

    def convert(self, s):
        s0 = State(s)
        l = [0,3,7,11,12,13,14,15]
        for i in range(s.boardSize):
            for j in range(s.boardSize):
                if (s0.board[i][j] not in l):
                    s0.board[i][j] = -1
        return s0

    def createDB(self):
        conn = sqlite3.connect('puzzle.sqlite3')
        cur = conn.cursor()
        cur.execute('DROP TABLE IF EXISTS PuzzleDB')
        cur.execute('CREATE TABLE PuzzleDB (config TEXT, cost INTEGER)')
        cur.execute('CREATE UNIQUE INDEX configidx ON PuzzleDB (config)')
        Node.nodeCount = 0
        p = Problem()
        s = State()
        p = Problem()
        s = State()
        p.initialState = s
        n=Node(None,None,0,p.initialState)
        frontier=[n]
        cur.execute('INSERT INTO PuzzleDB (config, cost) VALUES ( ?, ? )',( str(self.convert(n.state)), n.cost ))
        count = 1
        while len(frontier) > 0:
            n = frontier.pop(0)
            for a in p.applicable(n.state):
                s1 = p.apply(a,State(n.state))
                n0 = Node(None,None,n.cost+1,s1)
                x = cur.lastrowid
                cur.execute('INSERT OR IGNORE INTO PuzzleDB (config, cost) VALUES ( ?, ? )',( str(self.convert(n0.state)), n0.cost ))
                if cur.lastrowid != x:
                    count += 1
                    s1 = self.dMirror(n0.state)
                    cur.execute('INSERT OR IGNORE INTO PuzzleDB (config, cost) VALUES ( ?, ? )',( str(self.convert(s1)), n0.cost ))
                    count += 1
                    frontier.append(n0)
                    if count >= 100000:
                        count = 0
                        conn.commit()
        conn.commit()
        conn.close()

    def create3x3DB(self):
        conn = sqlite3.connect('puzzle3x3.sqlite3')
        cur = conn.cursor()
        cur.execute('DROP TABLE IF EXISTS PuzzleDB3x3')
        cur.execute('CREATE TABLE PuzzleDB3x3 (config TEXT, cost INTEGER)')
        cur.execute('CREATE INDEX configidx ON PuzzleDB3x3 (config)')
        s = State()
        t = [[1,2,4,5],[6,8,9,10],[2,5,8,10],[1,4,6,9]]
        for x in t:
            x.append(0)
            p = Problem()
            s = State()
            b = [[0,1,2],[3,4,5],[6,7,8]]
            for i in range(3):
                for j in range(3):
                    if b[i][j] not in x:
                        b[i][j] = -1
            s.board = b
            p.initialState = s
            n=Node(None,None,0,p.initialState)
            frontier=[n]
            explored=set()
            self.database3x3[n.state.toTuple()] = n.cost
            cur.execute('INSERT INTO PuzzleDB3x3 (config, cost) VALUES ( ?, ? )',( str(n.state), n.cost ))
            explored.add(n.state.toTuple())
            while len(frontier) > 0:
                n = frontier.pop(0)
                for a in p.applicable(n.state):
                    nc=child_node(n,a,p)
                    childState = nc.state.toTuple()
                    if not(childState in explored):
                        frontier.append(nc)
                        explored.add(childState)
                        self.database3x3[childState] = nc.cost
                        cur.execute('INSERT INTO PuzzleDB3x3 (config, cost) VALUES ( ?, ? )',( str(nc.state), nc.cost ))
            conn.commit()
        conn.close()

    def subSolver3x3(self):
        t = [[1,2,4,5],[6,8,9,10],[2,5,8,10],[1,4,6,9]]
        count = 0
        for x in t:
            s = State()
            b = [[0,1,2],[4,5,6],[8,9,10]]
            p = Problem()
            x.append(0)
            for i in range(3):
                for j in range(3):
                    if b[i][j] not in x:
                        b[i][j] = -1
            s.board = b
            p.initialState = s
            n=Node(None,None,0,p.initialState)
            frontier=[n]
            explored=set()
            self.database3x3[n.state.toTuple()] = n.cost
            explored.add(n.state.toTuple())
            while len(frontier) > 0:
                n = frontier.pop(0)
                for a in p.applicable(n.state):
                    nc=child_node(n,a,p)
                    childState = nc.state.toTuple()
                    if not(childState in explored):
                        frontier.append(nc)
                        explored.add(childState)
                        self.database3x3[childState] = nc.cost
            print(len(self.database3x3))

    def tree_bfs(self, problem):
        #reset the node counter for profiling
        Node.nodeCount=0
        n=Node(None,None,0,problem.initialState)
        print(n)
        frontier=[n]
        while len(frontier) > 0:
            n = frontier.pop(0)
            for a in p.applicable(n.state):
                nc=child_node(n,a,p)
                if nc.state == p.goalState:
                    return solution(nc)
                else:
                    frontier.append(nc)

    def graph_bfs(self, problem):
        Node.nodeCount=0
        n=Node(None,None,0,problem.initialState)
        frontier=[n]
        explored=set()
        while len(frontier) > 0:
            n = frontier.pop(0)
            for a in p.applicable(n.state):
                nc=child_node(n,a,p)
                if nc.state == p.goalState:
                    return solution(nc)
                else:
                    childState=nc.state.toTuple()
                    if not(childState in explored):
                        frontier.append(nc)
                        explored.add(childState)


    def recursiveDL_DFS(self, lim,problem):
        n=Node(None,None,0,problem.initialState)
        return self.depthLimitedDFS(n,lim,problem)

    def depthLimitedDFS(self, n, lim, problem):
        #print lim
        #print n

        #reasons to cut off brnaches
        if n.state == problem.goalState:
            return solution(n)
        elif lim == 0:
            return None

        cutoff=False
        for a in p.applicable(n.state):
            nc=child_node(n,a,problem)
            result = self.depthLimitedDFS(nc,lim-1,problem)

            if not result==None:
                return result

        return None

    def id_dfs(self,problem):

        Node.nodeCount=0

        maxLim=32
        for d in range(1,maxLim):
            result = self.recursiveDL_DFS(d,problem)
            if not result == None:
                return result
        print('Hit max limit of ' + str(maxLim))
        return None


    def h_1(self,s0: State,sf: State ) -> numbers.Real:
        count = 0
        for x in range(0, len(s0.board)):
            for j in range(0,len(s0.board)):
                if (s0.board[x][j] != 0 and s0.board[x][j] != sf.board[x][j]):
                    count += 1
        return count

    '''def h_2(self,s0: State,sf: State ) -> numbers.Real:
        count = 0
        for i in range(len(s0.board)):
            for j in range(len(s0.board[0])):
                x = s0.board[i][j]
                if x == 0:
                    continue
                for k in range(len(s0.board)):
                    for l in range(len(s0.board[0])):
                        if (sf.board[k][l] == x):
                            count += (abs(i-k)+abs(j-l))
        return count'''

    def h_2(self,s0: State,s1: State,cur) -> numbers.Real:
        t = [[1,2,4,5],[6,8,9,10],[2,5,8,10],[1,4,6,9]]
        rets = []
        for x in t:
            s = State(s0)
            x.append(0)
            for i in range(3):
                for j in range(3):
                    if (s.board[i][j] not in x):
                        s.board[i][j] = -1
            cur.execute("SELECT * FROM PuzzleDB3x3 WHERE config=?", (str(s),))
            x = cur.fetchone()[1]
            rets.append(x)
        return max(rets)

    def a_star_tree(self,problem : Problem) -> tuple:
        conn = sqlite3.connect('puzzle3x3.sqlite3')
        cur = conn.cursor()
        Node.nodeCount=0
        n=Node(None,None,0,problem.initialState)
        frontier=[]
        heapq.heappush(frontier,(n.cost,n))
        while len(frontier) > 0:
            n = heapq.heappop(frontier)
            if n[1].state == p.goalState:
                return solution(n[1])
            for a in p.applicable(n[1].state):
                nc=child_node(n[1],a,p)
                #f = nc.cost + self.h_2(nc.state,problem.goalState)
                f = nc.cost + self.h_2(nc.state,nc.state,cur)
                childState=nc.state.toTuple()
                for x in frontier:
                    if x[1].state == childState:
                        if x[0] > f:
                            x[0] = f
                            heapq.heapify(frontier)
                heapq.heappush(frontier,(f,nc))

    def a_star_graph(self,p : Problem) -> tuple:
        conn = sqlite3.connect('puzzle3x3.sqlite3')
        cur = conn.cursor()
        Node.nodeCount=0
        n=Node(None,None,0,p.initialState)
        frontier=[]
        heapq.heappush(frontier,(n.cost,n))
        explored=set()
        explored.add(n.state.toTuple())
        while len(frontier) > 0:
            n = heapq.heappop(frontier)
            if n[1].state == p.goalState:
                return solution(n[1])
            for a in p.applicable(n[1].state):
                nc=child_node(n[1],a,p)
                f = nc.cost + self.h_2(nc.state,p.goalState,cur)
                childState=nc.state.toTuple()
                for x in frontier:
                    if x[1].state == childState:
                        if x[0] > f:
                            x[0] = f
                            heapq.heapify(frontier)
                if not(childState in explored):
                    heapq.heappush(frontier,(f,nc))
                    explored.add(childState)

import time

p=Problem()
p2 = Problem()
s=State()
n=Node(None,None, 0, s)
n2=Node(n,None, 0, s)
p.goalState = s
searches = Searches()
print(str(n.state))
searches.create3x3DB()
#searches.createDSwap(s)
#searches.createDB()

'''print(searches.dMirror(s))
print(searches.vMirror(s))
print(searches.hMirror(s))
print(searches.d1Mirror(s))
for i in range (10):
    si = State(s)
    apply_rnd_moves(6,si,p)
    p.initialState=si
    res=searches.a_star_graph(p)
    print("True cost: ",res)
    print("D Mirror:\n",searches.dMirror(si))
    print("V Mirror:\n", searches.vMirror(si))

    print("H Mirror:\n",searches.hMirror(si))

    print("D1 Mirror:\n",searches.d1Mirror(si))




p.initialState=State(s)
print('\n\n=== A*-Graph ===\n')
startTime=time.clock()
res=searches.a_star_graph(p)
print(time.clock()-startTime)
print(Node.nodeCount)
print(res)'''
for i in range(50):
    si=State(s)
    # change the number of random moves appropriately
    # If you are curious see if you get a solution >30 moves. The
    apply_rnd_moves(100,si,p)
    #si.board=((7,2,4),(5,0,6),(8,3,1))
    p.initialState=si
    print(p.initialState)
    startTime=time.clock()
    print('\n\n=== A*-Graph ===\n')
    startTime=time.clock()
    res=searches.a_star_graph(p)
    print(time.clock()-startTime)
    print(Node.nodeCount)
    print(res)

'''print('=== Bfs*  ===')
startTime=time.clock()
res=searches.graph_bfs(p)
print(res)
print(time.clock()-startTime)
print(Node.nodeCount)

print('=== id DFS*  ===')
startTime=time.clock()
res=searches.id_dfs(p)
print(res)
print(time.clock()-startTime)
print(Node.nodeCount)

print('\n\n=== A*-Tree ===\n')
startTime=time.clock()
res=searches.a_star_tree(p)
print(time.clock()-startTime)
print(Node.nodeCount)
print(res)'''



#
#print('\n\n=== A*-G-SL  ===\n')
#startTime=time.clock()
#res=AstarGraph2(p)
#print(time.clock()-startTime)
#print(node.nodeCount)
#print(res)
#
#print('\n\n=== A*-G-HQ  ===\n')
#startTime=time.clock()
#res=AstarGraph3(p)
#print(time.clock()-startTime)
#print(node.nodeCount)
#print(res)
#
#print('=== Bfs*  ===')
#startTime=time.clock()
#res=bfsGraph(p)
#print(res)
#print(time.clock()-startTime)
#print(node.nodeCount)
#

'''
print('\n\n=== A* - Tree  ===\n')
startTime=time.clock()
res=Astar(p)
print(time.clock()-startTime)
print(node.nodeCount)
print(res)

print('\n\n=== A*-Tree-SL ===\n')
startTime=time.clock()
res=Astar2(p)
print(time.clock()-startTime)
print(node.nodeCount)
print(res)

'''

'''
print('=== iDFS*  ===')
startTime=time.clock()
res=iDL_DFS(p)
print(res)
print(time.clock()-startTime)
print(node.nodeCount)
'''

startTime=time.clock()
