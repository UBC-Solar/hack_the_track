
#Terminology:
# A state is a list of positions alongside their times

# Class to help defining mocked state to be easier, might be removed
class StatePosition(): 
    def __init__(self, x: float, y: float, time: float):
        self.x = x
        self.y = y
        self.time = time

gates = [
    [[1,0], [1,2]],
    [[-1,0], [-1,-2]],
]

originalState = [StatePosition((i-5)/10, (i-5)/10, i/10) for i in range(50)]
modifiedState1 = [StatePosition((i-5)/8, (i-5)/8, i/10) for i in range(50)]
modifiedState2 = [StatePosition((i-5)/12, (i-5)/12, i/10) for i in range(50)]
modifiedState3 = [StatePosition(-(i-5)/12, -(i-5)/12, i/10) for i in range(50)]

# Efficient Intersection Algorithm, taken from https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

def ccw(A: list, B: list, C: list):
    '''
    Returns true if 3 points are in clockwise order
    '''
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0] - A[0])

def intersect(segment1: list, segment2: list):
    '''
    Returns true if both segments intersect
    '''
    return ccw(segment1[0],segment2[0],segment2[1]) != ccw(segment1[1],segment2[0],segment2[1]) and ccw(segment1[0],segment1[1],segment2[0]) != ccw(segment1[0],segment1[1],segment2[1])

# Gate intersections functions

def findIntersection(state: list[StatePosition], gates: list[list]):
    '''
    Finds the gate and time to cross gate if the state crosses a gate in a list of gates
    '''
    initialTime = state[0].time

    for index in range(len(state)):
        if index != len(state):
            point1 = [state[index].x, state[index].x]
            point2 = [state[index + 1].x, state[index + 1].x]

            segment = [point1, point2]

            for gate in gates:
                if intersect(gate, segment):
                    intersectedGate = gate
                    time = state[index].time - initialTime

                    return intersectedGate, time

def testIntersection(state: list[StatePosition], gate: list):
    '''
    Finds the time for a state to cross a certain gate
    '''
    initialTime = state[0].time

    for index in range(len(state)):
        if index != len(state):
            point1 = [state[index].x, state[index].x]
            point2 = [state[index + 1].x, state[index + 1].x]

            segment = [point1, point2]

            if intersect(gate, segment):
                time = state[index].time - initialTime
                return time

def scoreState(originalState: list[StatePosition], modifiedState: list[StatePosition], gates: list[list]): #This might be modified to calculate for a list of modified states because it feels wasteful to recalculated base time every time
    '''
    Returns the time difference to cross a gate between two states
    '''
    intersectedGate, baseTime = findIntersection(originalState, gates)
    modifiedTime = testIntersection(modifiedState, intersectedGate)

    return modifiedTime - baseTime

print(scoreState(originalState, modifiedState1, gates))
print(scoreState(originalState, modifiedState3, gates))