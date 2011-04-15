cimport numpy as np
import numpy as np
cdef float inf = np.inf

cdef struct V3:
    float x
    float y
    float z


cdef inline float dot(V3 a, V3 b):
    return a.x*b.x + a.y*b.y + a.z*b.z


cdef inline V3 cross(V3 a, V3 b):
    cdef V3 c
    c.x = a.y*b.z - b.y*a.z
    c.y = a.z*b.x - b.z*a.x
    c.z = a.x*b.y - b.x*a.y
    return c


cdef inline V3 sub(V3 a, V3 b):
    cdef V3 c
    c.x = a.x - b.x
    c.y = a.y - b.y
    c.z = a.z - b.z
    return c


cdef intersect_box(float *origin,
                   float *inv_d,
                   int *sign,
                   float *bounds,
                   float t0=0,
                   float t1=1.e5):
    pass
    


cdef float intersect_triangle_p(V3 origin,
                                V3 direction,
                                V3 v0,
                                V3 v1,
                                V3 v2):
    cdef float EPSILON = 1.e-5
    # Fast Minimum Storage Ray/Triangle Intersection
    # by Moller and Trumbore
    cdef V3 e1 = sub(v1, v0)
    cdef V3 e2 = sub(v2, v0)
    cdef V3 pvec = cross(direction,e2)

    cdef float det = dot(e1,pvec)

    if (det > -EPSILON and det < EPSILON): return inf
    cdef float invdet = 1. / det

    cdef V3 tvec = sub(origin, v0)
    cdef float u = dot(tvec,pvec) * invdet
    if (u < 0.0 or u > 1.0): return inf

    cdef V3 qvec = cross(tvec, e1)
    cdef float v = dot(direction,qvec) * invdet
    if (v < 0.0 or u + v > 1.0): return inf

    cdef float t = dot(e2, qvec) * invdet
    if (t > EPSILON): return t
    else: return inf



def intersect_triangle(np.ndarray[np.float32_t] origin,
                       np.ndarray[np.float32_t] direction,
                       np.ndarray[np.float32_t] v0,
                       np.ndarray[np.float32_t] v1,
                       np.ndarray[np.float32_t] v2):
    cdef float t = intersect_triangle_p((<V3*>origin.data)[0],
                                        (<V3*>direction.data)[0],
                                        (<V3*>v0.data)[0],
                                        (<V3*>v1.data)[0],
                                        (<V3*>v2.data)[0])
    return t


cdef struct Node:
    float t

cdef struct Tri:
    int v0
    int v1
    int v2

cdef class CyBVH(object):
    cdef np.ndarray verts
    cdef np.ndarray tris
    cdef Node *nodes

    def __init__(self, np.ndarray verts, np.ndarray tris, nodes):
        assert verts.dtype == np.float32
        assert tris.dtype == np.int
        assert tris.shape[1] == 3
        assert verts.shape[1] == 3
        self.verts = verts
        self.tris = tris

    cpdef intersect_all(self,
                        np.ndarray[np.float32_t] origin,
                        np.ndarray[np.float32_t] direction):

        cdef V3 _origin = (<V3*>origin.data)[0]
        cdef V3 _direction = (<V3*>direction.data)[0]
        
        cdef Tri best_tri
        cdef float min_T = inf

        cdef V3 *verts = <V3*> self.verts.data
        cdef Tri *tris = <Tri*> self.tris.data
        cdef Tri tri
        cdef V3 v0, v1, v2
        cdef float hit_T
        
        for i in xrange(len(self.tris)):
            tri = tris[i]
            v0 = verts[tri.v0]
            v1 = verts[tri.v1]
            v2 = verts[tri.v2]
            hit_T = intersect_triangle_p(_origin, _direction, v0, v1, v2)
        
            if hit_T < min_T:
                min_T = hit_T
                best_tri = tri
    
        return best_tri, min_T


        
