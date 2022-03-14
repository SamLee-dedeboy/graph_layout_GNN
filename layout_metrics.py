from sklearn import neighbors
from torch_geometric.data import Data
import numpy as np
import math
class Point:
    def __init__(self, pos):
        self.x = pos[0]
        self.y = pos[1]
def edge_length_variation(graphData):
    nodes = graphData.x.numpy()
    edges = np.transpose(graphData.pos_edge_index.numpy())
    E_n = edges.size
    edge_length_list = [edge_length(edge, nodes) for edge in edges]
    l_mu = np.mean(edge_length_list)

    l_a = np.sqrt(np.sum([(l_e - l_mu)**2/(E_n*(l_mu**2)) for l_e in edge_length_list]))
    M_l = l_a/(np.sqrt(E_n - 1))
    return M_l
def edge_crossings(graphData):
    nodes = graphData.x.numpy()
    edges = np.transpose(graphData.pos_edge_index.numpy())
    cross_num = 0
    for i in range(len(edges-1)):
        for j in range(i+1, len(edges)):
            edge1 = edges[i]
            edge2 = edges[j]
            cross_num += check_cross(edge1, edge2, nodes)
    return cross_num
def minimum_angle(graphData):
    nodes = graphData.x.numpy()
    edges = np.transpose(graphData.pos_edge_index.numpy())
    num_nodes = len(nodes)
    # Dict[nodeId, edges of node]
    edges_dict = gen_edges_dict(edges, num_nodes)
    incident_angle_list = [incident_min_angle(nodes[i], [nodes[neighbor] for neighbor in edges_dict[i]]) for i in range(num_nodes)]
    minimum_angle_list = [360/len(edges_dict[i]) if len(edges_dict[i])!=0 else 0 for i in range(num_nodes)]
    M_a = 1 - np.sum([(incident_angle_list[v] - minimum_angle_list[v])/minimum_angle_list[v] if minimum_angle_list[v] !=0 else 0 for v in range(num_nodes)])/num_nodes
    return M_a
def gen_edges_dict(edges, num_nodes):
    edges_dict = [[] for _ in range(num_nodes)]
    for edge in edges:
        node1 = edge[0]
        node2 = edge[1]
        edges_dict[node1].append(node2)
    return edges_dict
def incident_min_angle(o_pos, neighbors_pos):
    n_neighbors = len(neighbors_pos)
    if n_neighbors <= 1:
        return 360
    incident_angle_list = []
    for i in range(n_neighbors-1):
        for j in range(i+1, n_neighbors):
            incident_angle_list.append(incident_angle(o_pos, neighbors_pos[i], neighbors_pos[j]))
    min_angle = np.min(incident_angle_list)
    return min_angle

def incident_angle(C, B, A):
    # C as origin 
    a = math.dist(C, B)
    b = math.dist(C, A)
    c = math.dist(A, B)
    cos_C = (a**2 + b**2 - c**2)/(2*a*b)
    return math.degrees(math.acos(cos_C))
def check_cross(e1, e2, nodes):
    p1 = Point(nodes[e1][0])
    p2 = Point(nodes[e1][1])
    q1 = Point(nodes[e2][0])
    q2 = Point(nodes[e2][1])
    return int(doIntersect(p1, q1, p2, q2))
def onSegment(p, q, r):
    if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
           (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
     
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
    # for details of below formula.
     
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if (val > 0):
         
        # Clockwise orientation
        return 1
    elif (val < 0):
         
        # Counterclockwise orientation
        return 2
    else:
         
        # Collinear orientation
        return 0
def doIntersect(p1,q1,p2,q2):
    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
 
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
 
    # Special Cases
 
    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
 
    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
 
    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
 
    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
 
    # If none of the cases
    return False

def edge_length(edge, nodes):
    n1 = edge[0]
    n2 = edge[1]
    pos1 = nodes[n1]
    pos2 = nodes[n2]
    return math.dist(pos1, pos2)