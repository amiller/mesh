import cybvh
import bvh
import numpy as np
import mesh
import time

N_RAYS = 10000
origins = np.random.rand(N_RAYS,3).astype('f')
directions = np.random.rand(N_RAYS,3).astype('f')-0.5
directions /= np.sqrt(np.sum(directions*directions,1))\
              .reshape(-1,1).astype('f')

mesh.load('blueghost')
global mycybvh
mycybvh = cybvh.CyBVH(mesh.mybvh.verts[:,:3].copy(),
                      np.array(mesh.mybvh.tris)[:,:3].copy(),
                      mesh.mybvh.nodes)


def test_intersect_triangle():
    N_TRIS = 100

    v0,v1,v2 = np.random.rand(3, N_TRIS, 3).astype('f')

    bvht = [bvh.intersect_triangle(origins[i], directions[i],
                                   v0[j], v1[j], v2[j])
            for i in range(N_RAYS) for j in range(N_TRIS)]
    cybvht = [cybvh.intersect_triangle(origins[i], directions[i],
                                       v0[j], v1[j], v2[j])
              for i in range(N_RAYS) for j in range(N_TRIS)]
    bvht = np.array(bvht)
    cybvht = np.array(cybvht)

    err = np.abs(bvht-cybvht)
    assert np.all((err<1e-5) | (bvht==cybvht))


def test_load_bvh():

    global tri_T1
    t0_0 = time.time()
    tri_T1 = [mycybvh.intersect_all(origins[i], directions[i])
              for i in range(N_RAYS)]
    t0_1 = time.time()
    print "[cython] %d tests in %.3f seconds" % (N_RAYS*len(mesh.mybvh.tris),
                                                 t0_1-t0_0)

    global tri_T2
    t1_0 = time.time()
    tri_T2 = [mesh.mybvh.intersect_all(origins[i], directions[i])
              for i in range(N_RAYS/1000)]
    t1_1 = time.time()
    print "[numpy]  %d tests in %.3f seconds" % \
          (len(mesh.mybvh.tris)*N_RAYS/1000, t1_1-t1_0)
