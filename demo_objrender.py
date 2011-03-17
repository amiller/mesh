from visuals.pointwindow import PointWindow
from objloader import OBJ,MTL
from OpenGL.GL import *
import opencl
import cv
import bvh
import cPickle as pickle
import os

#meshname = 'teapot'
#meshname = 'blueghost'
#meshname = 'bunny69k'
meshname = 'hall1'

if not 'window' in globals():
    window = PointWindow(size=(640,480))


def load_mesh():
    opencl.load_mesh(vertices, faces)
    window.Refresh()


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


def load_obj():
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

    points = np.array(obj.vertices,'f')

    # Scale the model down and center it
    global scale
    scale = points.std()*2
    points /= scale
    points = np.hstack((points,np.zeros((points.shape[0],1),'f')))
    window.update_points(points[:,:3])
    window.lookat = points[:,:3].mean(0)
    window.Refresh()

    # Just the vertices, useful for putting in a vertexbuffer
    global vertices
    vertices = np.ascontiguousarray(points)

    # The faces, specifically triangles, as vertex indices
    global faces
    faces = np.array([(v[0],v[1],v[2],0) for v,_,_,_ in obj.faces])-1

    load_mesh()

    global mybvh
    mybvh = cache_or_build('data/%s.obj' % meshname,
                           'data/%s_bvh.pkl' % meshname,
                           lambda _: bvh.from_faces(vertices, faces))

if not 'obj' in globals(): load_obj()
load_mesh()


def make_random_ray():
    # Create two random points, one on the r=2 sphere...
    x1 = np.random.rand(3)-0.5
    x1 /= np.sqrt(np.dot(x1, x1))
    x1 *= 2

    # and another on the r=0.5 sphere
    x2 = np.random.rand(3)-0.5
    x2 /= np.sqrt(np.dot(x2, x2))
    x2 *= 0.3

    dir = x2 - x1
    dir /= np.sqrt(np.dot(dir,dir))
    ray = x1, dir
    return ray


def random_ray():
    global ray
    ray = make_random_ray()


@window.eventx
def EVT_CHAR(evt):
    key = evt.GetKeyCode()
    if key == ord(' '):
        random_ray()
        window.Refresh()


@window.event
def post_draw():
    glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.3, 0.3, 0.3, 0.0))
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    #mat = glGetFloatv(GL_MODELVIEW_MATRIX).transpose()
    glPushMatrix()
    glScale(1.0/scale,1.0/scale,1.0/scale)
    glCallList(obj.gl_list)
    glPopMatrix()
    glDisable(GL_LIGHTING)

    if 1:
        glColor(1,1,0)
        mybvh.draw_boxes()

    if 'ray' in globals():
        glColor(1,1,0)
        nodes, tri, t = mybvh.intersect(*ray)
        x1,d = ray
        if t > 1000: t = 1000

        verts, line_inds, _ = bvh.BVH.box_vertices(nodes)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointerf(verts)
        glDrawElementsui(GL_LINES, line_inds)
        glDisableClientState(GL_VERTEX_ARRAY)

        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glPointSize(10)
        glColor(0,1,0)
        glDisable(GL_DEPTH_TEST)
        glBegin(GL_POINTS)
        glVertex(*(x1+d*t))
        glEnd()
        glEnable(GL_DEPTH_TEST)
        glLineWidth(3)
        glBegin(GL_LINES)
        glVertex(*x1)
        glVertex(*(x1+d*t))
        glEnd()
        glPopAttrib(GL_ALL_ATTRIB_BITS)

    if 0:
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        #opencl.compute_raytrace(mat).wait()

        global img
        img = opencl.get_img()
        cvim = cv.CreateImage((640,480),8,3)
        cv.SetData(cvim, np.array(img[:,:,:3]*255,'u1').data)
        cv.ShowImage('im',cvim)

window.Refresh()
