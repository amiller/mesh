from OpenGL.GL import *
import numpy as np
from numpy import inf


def from_faces(verts, faces):
    bvh = BVH()
    assert verts.shape[1] == 4
    assert faces.shape[1] == 4
    assert faces.dtype == 'i'
    bvh.verts = np.array(verts,'f',copy=True)

    Ttri = 4
    Taabb = 1

    def Centroids():
        """
        return the centroid for each face
        """
        v = verts[faces[:,:3],:3]
        return v.mean(1)

    def Box(S):
        """S is a list of indices in faces
        Return the mins and maxs of the whole set
        """
        v = verts[faces[S,:3].flatten(),:3]
        return np.min(v,0), np.max(v,0)

    def Sorted(S, axis):
        inds = np.argsort(centroids[S,axis])
        return S[inds,:]

    def AreaMM(min,max):
        X,Y,Z = max - min
        return 2*(X*Y + Y*Z + Z*X)

    def Area(S):
        if len(S) == 0: return inf
        min, max = Box(S)
        return AreaMM(min, max)

    # Use the algorithm as described in Wald 2006
    # http://www.cs.utah.edu/~boulos/papers/dynbvh.pdf
    # It's better than quadratic!
    def partitionSweep(S):
        BestCost = Ttri * len(S)
        BestAxis = -1
        BestEvent = -1
        OverallBox = Box(S)
        A = Area(S)

        for axis in range(3):
            S = Sorted(S, axis)
            minL = np.array([inf,inf,inf])
            maxL = np.array([-inf,-inf,-inf])
            LeftArea = np.empty(len(S))

            for i in range(len(S)):
                LeftArea[i] = AreaMM(minL,maxL)
                min, max = Box([S[i]])
                minL = np.minimum(minL, min)
                maxL = np.maximum(maxL, max)

            minR = np.array([inf,inf,inf])
            maxR = np.array([-inf,-inf,-inf])
            for i in range(len(S))[::-1]:
                min, max = Box([S[i]])
                minR = np.minimum(minR, min)
                maxR = np.maximum(maxR, max)

                A1 = LeftArea[i]
                A2 = AreaMM(minR,maxR)
                ThisCost = 2*Taabb + A1/A*i*Ttri + A2/A*(len(S)-i)*Ttri
                if ThisCost < BestCost:
                    BestCost = ThisCost
                    BestEvent = i
                    BestAxis = axis

        if BestAxis == -1:
            # Make Leaf node, return its ID
            triID = len(bvh.tris)
            node = OverallBox + (triID, len(S), -1, -1)
            nodeID = len(bvh.nodes)
            bvh.tris += faces[S,:].tolist()
            bvh.nodes += [node]
            return nodeID

        else:
            S = Sorted(S, BestAxis)
            S1 = S[:BestEvent]
            S2 = S[BestEvent:]
            # Make Inner node
            N1,N2 = partitionSweep(S1),partitionSweep(S2)
            nodeID = len(bvh.nodes)
            node = OverallBox + ((N1,N2), 0, -1, -1)
            bvh.nodes += [node]
            return nodeID

    centroids = Centroids()
    bvh.tris = []
    bvh.nodes = []
    partitionSweep(np.arange(len(faces)))
    bvh.buffers = BVH.box_vertices(bvh.nodes)
    return bvh


def intersect_triangle(origin, direction, v0, v1, v2):
    EPSILON = 1.e-5
    # Fast Minimum Storage Ray/Triangle Intersection
    # by Moller and Trumbore
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(direction,e2)
    det = np.dot(e1,pvec)

    if (det > -EPSILON and det < EPSILON): return inf
    invdet = 1. / det

    tvec = origin - v0
    u = np.dot(tvec,pvec) * invdet
    if (u < 0.0 or u > 1.0): return inf

    qvec = np.cross(tvec, e1)
    v = np.dot(direction,qvec) * invdet
    if (v < 0.0 or u + v > 1.0): return inf

    t = np.dot(e2, qvec) * invdet
    if (t > EPSILON): return t
    else: return inf


def intersect_box(origin, inv_d, sign, bounds, t0=0., t1=1.e5):
    # Fast AABB intersection due to Williams et al 2004
    # http://people.csail.mit.edu/amy/papers/box-jgt.pdf
    tmin = (bounds[sign[0]][0] - origin[0]) * inv_d[0]
    tmax = (bounds[1-sign[0]][0] - origin[0]) * inv_d[0]
    tymin = (bounds[sign[1]][1] - origin[1]) * inv_d[1]
    tymax = (bounds[1-sign[1]][1] - origin[1]) * inv_d[1]

    if tmin > tymax or tymin > tmax: return False
    if tymin > tmin: tmin = tymin
    if tymax < tmax: tmax = tymax

    tzmin = (bounds[sign[2]][2] - origin[2]) * inv_d[2]
    tzmax = (bounds[1-sign[2]][2] - origin[2]) * inv_d[2]

    if tmin > tzmax or tzmin > tmax: return False
    if tzmin > tmin: tmin = tzmin
    if tzmax < tmax: tmax = tzmax

    return tmin < t1 and tmax > t0


class BVH(object):

    def intersect_all(self, origin, direction):
        best_tri = None
        min_T = inf

        for tri in self.tris:
            hit_T = intersect_triangle(origin, direction,
                                       *self.verts[tri[:3],:3])
            if hit_T < min_T:
                min_T = hit_T
                best_tri = tri

        return best_tri, min_T

    def intersect(self, origin, direction):
        assert origin.flatten().shape == (3,)
        assert direction.flatten().shape == (3,)
        origin = origin.astype('f')
        direction = direction.astype('f')

        # Precompute some inverse factors for the ray
        inv_d = 1. / direction
        sign = (inv_d < 0).astype('i')

        # Start with the root node
        stack = []
        stack.append(self.nodes[-1])

        # Store all the intersected nodes, just for fun
        hit_boxes = []

        # Find the nearest intersecting triangle
        best_tri = None
        min_T = inf

        # Try one box at a time
        while len(stack) > 0:
            node = stack.pop()
            min, max, ID, n_tri, _, _ = node
            if n_tri > 0:  # Leaf node
                # Intersect against each of the triangles
                for tri in self.tris[ID:ID+n_tri]:
                    hit_T = intersect_triangle(origin, direction,
                                               *self.verts[tri[:3],:3])
                    if hit_T < min_T:
                        best_tri = tri
                        min_T = hit_T

            else:  # Inner node
                if intersect_box(origin, inv_d, sign, (min,max), 0, min_T):
                    hit_boxes.append(node)
                    N1,N2 = ID
                    # Push the children on the stack
                    stack.append(self.nodes[N1])
                    stack.append(self.nodes[N2])

        return hit_boxes, best_tri, min_T

    def draw_boxes(self, max_depth=2):
        #verts, line_inds, _ = BVH.box_vertices(self.nodes)
        verts, line_inds, _ = self.buffers
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointerf(verts)
        glDrawElementsui(GL_LINES, line_inds)
        glDisableClientState(GL_VERTEX_ARRAY)

    @classmethod
    def box_vertices(cls, nodes):
        """Given some BVH nodes, produce a list of vertices and indices
        suitable for loading into an opengl vertex array
        for drawing quads or line strips around the boxes
        """
        q = [[[1,1,0],[0,1,0],[0,1,1],[1,1,1]], \
             [[1,0,1],[0,0,1],[0,0,0],[1,0,0]], \
             [[1,1,1],[0,1,1],[0,0,1],[1,0,1]], \
             [[1,0,0],[0,0,0],[0,1,0],[1,1,0]], \
             [[0,1,1],[0,1,0],[0,0,0],[0,0,1]], \
             [[1,1,0],[1,1,1],[1,0,1],[1,0,0]]]
        q = np.array(q).reshape(1,-1,3)

        min = np.array([min for min,_,_,_,_,_ in nodes]).reshape(-1,1,3)
        max = np.array([max for _,max,_,_,_,_ in nodes]).reshape(-1,1,3)
        vertices = (q*min + (1-q)*max).reshape(-1,3)
        line_inds = np.arange(0,len(min)*6).reshape(-1,1)*4 + [0,1,1,2,2,3,3,0]
        quad_inds = np.arange(0,len(max)*6).reshape(-1,1)*4 + [0,1,2,3]
        return vertices, line_inds, quad_inds

    def __init__(self):
        """Nodes are
        min(x,y,z), max(x,y,z), {triID|(nID1,nID2}, n_tri, axis, sign

        If the node is a leaf:
           the third element will be the first triangle id
           n_tri is the number of triangles
        If the node is an inner node:
           the third element is a tuple of the two children node indices
           n_tri = 0
        """
        self.nodes = []
        self.tris = []
        self.verts = np.empty((0,4),'f')
