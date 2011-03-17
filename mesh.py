from objloader import OBJ,MTL
import bvh
import cPickle as pickle
import os
import glob
import numpy as np

meshes = [os.path.split(os.path.splitext(_)[0])[1]
          for _ in glob.glob('data/*.obj')]


def load_random():
    import random
    load(random.choice(meshes))


def load(meshname, do_scale=False):
    if meshname in ('teapot', 'bunny69k', 'blueghost'):
        do_scale = True

    # Load a mesh from an obj file
    global obj

    def build(src_file):
        savedpath = os.getcwd()
        try:
            os.chdir('data')
            return OBJ('%s.obj' % meshname)
        finally:
            os.chdir(savedpath)
    obj = cache_or_build('data/%s.obj' % meshname,
                         'data/%s_cache.pkl' % meshname,
                         build)
    obj.compile()
    global points
    points = np.array(obj.vertices,'f')

    # Scale the model down and center it
    global scale
    scale = 1.0
    if do_scale:
        scale = points.std()*2
        points /= scale

    points = np.hstack((points,np.zeros((points.shape[0],1),'f')))

    # Just the vertices, useful for putting in a vertexbuffer
    global vertices
    vertices = np.ascontiguousarray(points)

    # The faces, specifically triangles, as vertex indices
    global faces
    faces = np.array([(v[0],v[1],v[2],0) for v,_,_,_ in obj.faces])-1

    global mybvh
    mybvh = cache_or_build('data/%s.obj' % meshname,
                           'data/%s_bvh.pkl' % meshname,
                           lambda _: bvh.from_faces(vertices, faces))


def cache_or_build(src_file, cache_file, build_func):
    try:
        t1 = os.stat(src_file).st_mtime
        t2 = os.stat(cache_file).st_mtime
        assert t1 < t2
        print 'Loading from cache %s' % cache_file
        with open(cache_file, 'rb') as f:
            obj = pickle.load(f)
    except (OSError, IOError, AssertionError):
        print 'Building %s -> %s' % (src_file, cache_file)
        obj = build_func(src_file)
        print 'Saving to cache %s' % cache_file
        with open(cache_file, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return obj
