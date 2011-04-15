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
    return V3(a.x - b.x, a.y - b.y, a.z - b.z)

cdef inline V3 add(V3 a, V3 b):
    return V3(a.x + b.x, a.y + b.y, a.z + b.z)

cdef inline V3 scale(V3 a, float f):
    return V3(a.x*f, a.y*f, a.z*f)


cdef extern from "stdlib.h":
    int rand()
    int INT_MAX
cdef float iM = 1./INT_MAX

cdef V3 sample_unitsphere_p():
    cdef float sq = inf
    cdef V3 ray

    while sq > 0.25:
        ray = V3(rand()*iM-0.5,
                 rand()*iM-0.5,
                 rand()*iM-0.5)
        sq = dot(ray,ray)
    sq = 1./sqrt(sq)
    ray = V3(ray.x*sq, ray.y*sq, ray.z*sq)
    return ray

def sample_unitsphere():
    cdef V3 v = sample_unitsphere_p()
    return np.array((v.x, v.y, v.z), np.float32)





cdef extern from "math.h":
    float sqrt(float s)
    
cdef float intersect_sphere_p(float r,
                              V3 c,
                              V3 p,
                              V3 d):
    cdef V3 oc = sub(c, p)
    cdef float loc2 = dot(oc,oc)
    if (loc2 < r*r):
        return inf  # Starting inside the sphere!
    cdef float tca = dot(d, oc)
    if (tca < 0):
        return inf  # Sphere center is behind us
    cdef float lhc2 = r*r - loc2 + tca*tca
    if (lhc2 < 0):
        return inf  # Missed!
    cdef float t = tca - sqrt(lhc2)
    return t



def intersect_sphere(float radius,
                     np.ndarray[np.float32_t, ndim=1] center,
                     np.ndarray[np.float32_t, ndim=1] camera,
                     np.ndarray[np.float32_t, ndim=1] direction):
    return intersect_sphere_p(radius,
                              (<V3*>center.data)[0],
                              (<V3*>camera.data)[0],
                              (<V3*>direction.data)[0])





cdef int intersect_box_p(V3 origin,
                         V3 inv_d,
                         int *sign,
                         V3 *bounds,
                         float t0,
                         float t1):
    # Fast AABB intersection due to Williams et al 2004
    # http://people.csail.mit.edu/amy/papers/box-jgt.pdf
    cdef float tmin = (bounds[sign[0]].x - origin.x) * inv_d.x
    cdef float tmax = (bounds[1-sign[0]].x - origin.x) * inv_d.x
    cdef float tymin = (bounds[sign[1]].y - origin.y) * inv_d.y
    cdef float tymax = (bounds[1-sign[1]].y - origin.y) * inv_d.y

    if tmin > tymax or tymin > tmax: return False
    if tymin > tmin: tmin = tymin
    if tymax < tmax: tmax = tymax

    cdef float tzmin = (bounds[sign[2]].z - origin.z) * inv_d.z
    cdef float tzmax = (bounds[1-sign[2]].z - origin.z) * inv_d.z

    if tmin > tzmax or tzmin > tmax: return False
    if tzmin > tmin: tmin = tzmin
    if tzmax < tmax: tmax = tzmax

    return tmin < t1 and tmax > t0
    

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
    V3 min
    V3 max
    int ID
    int n_tri
    int N2
    

cdef struct Tri:
    int v0
    int v1
    int v2

cdef struct NodeArray:
    int n
    Node p[1024]

cdef inline push(NodeArray *nodes, Node node):
    assert nodes[0].n < 1024
    nodes[0].p[nodes[0].n] = node
    nodes[0].n += 1

cdef inline Node pop(NodeArray *nodes):
    assert nodes[0].n > 0
    nodes[0].n -= 1
    return nodes[0].p[nodes[0].n]

cdef struct IntersectResult:
    Tri best_tri
    float t


cdef enum PathType:
    SOURCE, SINK, DIVERGE, SCAFLECT

cdef struct Path:
    V3 origin
    V3 direction
    PathType ntype
    float cumdist


cdef class CyBVH(object):
    cdef np.ndarray verts
    cdef np.ndarray tris
    cdef NodeArray nodes
    

    def __init__(self,
                 np.ndarray[np.float32_t, ndim=2] verts,
                 np.ndarray[np.int_t, ndim=2] tris,
                 nodes):
        assert tris.shape[1] == 3
        assert verts.shape[1] == 3
        self.verts = verts
        self.tris = tris
        self.nodes = NodeArray(0)
        cdef V3 _min, _max
        for min, max, ID, n_tri, _, _ in nodes:
            assert min.dtype == np.float32
            assert max.dtype == np.float32
            assert min.flags.contiguous
            assert max.flags.contiguous
            _min = V3(min[0], min[1], min[2])
            _max = V3(max[0], max[1], max[2])
            if n_tri == 0:
                push(&self.nodes, Node(_min, _max,
                                       ID[0], n_tri, ID[1]))
            else:
                push(&self.nodes, Node(_min, _max,
                                       ID, n_tri, -1))

    cpdef intersect(self,
                    np.ndarray[np.float32_t] origin,
                    np.ndarray[np.float32_t] direction):
        cdef V3 _origin = (<V3*>origin.data)[0]
        cdef V3 _direction = (<V3*>direction.data)[0]
        return self.intersect_(_origin, _direction)

    cdef IntersectResult intersect_(self, V3 origin, V3 direction):
        # Precompute some inverse factors for the ray
        cdef V3 inv_d = V3(1./direction.x, 1./direction.y, 1./direction.z)
        cdef int *sign = [inv_d.x<0, inv_d.y<0, inv_d.z<0]

        # Start with the root node
        cdef NodeArray stack = NodeArray(0)
        push(&stack, self.nodes.p[self.nodes.n-1])

        # Find the nearest intersecting triangle
        cdef Tri best_tri
        cdef float min_T = inf

        cdef Node node
        cdef V3 min, max
        cdef float hit_T
        cdef int ID, n_tri
        cdef Tri tri
        cdef V3 bounds[2]

        cdef V3 *verts = <V3*> self.verts.data
        cdef Tri *tris = <Tri*> self.tris.data

        # Try one box at a time
        while stack.n > 0:
            node = pop(&stack)
            min = node.min
            max = node.max
            n_tri = node.n_tri
            
            if n_tri > 0:  # Leaf node
                ID = node.ID
                # Intersect against each of the triangles
                for i in range(ID,ID+n_tri):
                    tri = tris[i]

                    hit_T = intersect_triangle_p(origin, direction,
                                                 verts[tri.v0],
                                                 verts[tri.v1],
                                                 verts[tri.v2])
                    if hit_T < min_T:
                        best_tri = tri
                        min_T = hit_T

            else:  # Inner node
                bounds[0] = min
                bounds[1] = max
                if intersect_box_p(origin, inv_d, sign, bounds, 0, min_T):
                    # Push the children on the stack
                    push(&stack, self.nodes.p[node.ID])
                    push(&stack, self.nodes.p[node.N2])

        return IntersectResult(best_tri, min_T)


    cpdef intersect_all(self, 
                        np.ndarray[np.float32_t] origin,
                        np.ndarray[np.float32_t] direction):
        cdef V3 _origin = (<V3*>origin.data)[0]
        cdef V3 _direction = (<V3*>direction.data)[0]
        return self.intersect_all_(_origin, _direction)


    cdef intersect_all_(self, V3 origin, V3 direction):
        
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
            hit_T = intersect_triangle_p(origin, direction, v0, v1, v2)
        
            if hit_T < min_T:
                min_T = hit_T
                best_tri = tri
    
        return best_tri, min_T

    cpdef sample_rays(self,
                      np.ndarray[np.float32_t] source_,
                      np.ndarray[np.float32_t] sink_,
                      float sinkrad,
                      int n_rays=10000,
                      float roulette=0.2):

        cdef V3 source = (<V3*>source_.data)[0]
        cdef V3 sink= (<V3*>sink_.data)[0]

        paths = []
        path = []
        cdef Path p
        
        for _ in range(n_rays):
            # path a list of vertices from the origin to the final intersection
            path = []
            p = Path(source, sample_unitsphere_p(), SOURCE, 0)
            
            # Generate a fixed number of rays
            # TODO implement russian roulette
            while True:
                path.append(p)
                if p.ntype in (DIVERGE, SINK) or rand()*iM < roulette:
                    break
                p = self.step(p, sinkrad, sink)

            if p.ntype == SINK:
                paths.append(path)

        return paths



    cdef Path step(self, Path path, float sinkrad, V3 sink):

        cdef float cumdist = path.cumdist
        cdef V3 origin = path.origin
        cdef V3 direction = path.direction

        cdef V3 *verts = <V3*> self.verts.data
        cdef Tri *tris = <Tri*> self.tris.data
    
        # Check if it intserects the listener sphere
        # sphere_t = sphere_intersect(sinkrad, sink, origin, direction)
        cdef float sphere_t = intersect_sphere_p(sinkrad, sink, origin, direction)

        # See if it intersects the scene
        cdef IntersectResult _ir = self.intersect_(origin, direction)
        cdef Tri tri = _ir.best_tri
        cdef float t = _ir.t

        cdef V3 normal, v0, v1, v2
        cdef V3 specular, diffuse
        cdef float alpha

        if sphere_t < inf and sphere_t < t:
            # We made it to the listener before crashing
            return Path(add(origin, scale(direction,sphere_t)),V3(0,0,0),SINK,cumdist+sphere_t)

        elif t == inf:
            # We diverged
            return Path(V3(0,0,0),V3(0,0,0),DIVERGE,inf)

        else:
            # We intersected the scene. Choose a weighted combination of
            # diffuse and specular reflection as per (Rezk 2007)
            origin = add(origin, scale(direction, t))

            # Get the normal from the vertices and triangle list in the bvh
            v0 = verts[tri.v0]
            v1 = verts[tri.v1]
            v2 = verts[tri.v2]

            normal = cross(sub(v1,v0),sub(v2,v1))
            normal = scale(normal, 1./sqrt(dot(normal,normal)))

            # Choose a weighted sum of the specular reflection and
            # a hemisphere sampling
            specular = sub(direction, scale(normal, 2*dot(normal,direction)))
            diffuse = sample_unitsphere_p()
            diffuse = scale(diffuse, -1 if dot(diffuse,normal) < 0 else 1)
            alpha = 0.3
            direction = add(scale(diffuse,alpha), scale(specular, 1 - alpha))
            direction = scale(direction, 1./sqrt(dot(direction,direction)))

            return Path(origin,direction,SCAFLECT,cumdist+t)

