from visuals.pointwindow import PointWindow
from OpenGL.GL import *
from OpenGL.GLU import *
import cv
import mesh
import wx
import cybvh
import random

if not 'window' in globals():
    window = PointWindow(size=(640,480))


meshname = 'hall1'
mesh.load(meshname)
mycybvh = cybvh.CyBVH(mesh.mybvh.verts[:,:3].copy(),
                      np.array(mesh.mybvh.tris)[:,:3].copy(),
                      mesh.mybvh.nodes)


def reset_sink():
    global sink, sinkrad, source
    sinkrad = 0.8
    source = np.array([-1,0,0.2],'f')
    sink = np.array([0,0,0],'f')

if not 'sink' in globals():
    reset_sink()
    window.Refresh()


def sphere_intersect(radius, center, camera, direction):
    import scipy.weave
    assert center.dtype == camera.dtype == direction.dtype == np.float32
    assert center.shape == (3,)
    assert camera.shape == (3,)
    assert direction.shape == (3,)
    r = radius
    c = center
    d = direction
    p = camera
    oc = c - p
    loc2 = np.dot(oc,oc)
    if (loc2 < r*r):
        return inf  # Starting inside the sphere!
    tca = np.dot(d, oc)
    if (tca < 0):
        return inf  # Sphere center is behind us
    lhc2 = r*r - loc2 + tca*tca
    if (lhc2 < 0):
        return inf  # Missed!
    t = tca - np.sqrt(lhc2)
    return t


def random_ray():
    """Generate some rays and intersect with the world. If it intersects,
    make a specular reflections. Check for intersection with the sphere.
    """
    d = np.random.rand(3).astype('f')-.5
    d /= np.sqrt(np.dot(d,d))
    t = sphere_intersect(sinkrad, sink, source, d)
    global ray
    ray = dict(direction=d,origin=source,t=t)
    #window.Refresh()


def find_direct_ray():
    for i in range(10000):
        random_ray()
        if ray['t'] < inf:
            break


def show_distribution():
    global paths
    dists = np.array([p[-1]['cumdist'] for p in paths])
    return dists


def sample_rays(n_rays=10000):
    global paths
    ps = mycybvh.sample_rays(source, sink, sinkrad, n_rays)
    keys = ['source','sink','diverge','scaflect']
    paths = []
    for path in ps:
        p_ = []
        for p in path:
            o = p['origin']
            d = p['direction']
            n = p['ntype']
            cumdist = p['cumdist']
            p_.append(dict(origin=np.array((o['x'],o['y'],o['z'])),
                           direction=np.array((d['x'],d['y'],d['z'])),
                           ntype=keys[n],
                           cumdist=cumdist))
        paths.append(p_)
    window.Refresh()


def set_camera(self):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    if self.mode == 'perspective':
      gluPerspective(60, 4/3., 0.3, 200)
    else:
      glOrtho(-1.33,1.33,-1,1,0.3,200)

    glMatrixMode(GL_MODELVIEW)
    # flush that stack in case it's broken from earlier
    try:
      while 1: glPopMatrix()
    except:
      pass

    glLoadIdentity()
    R = np.cross(self.upvec, [0,0,1])
    R /= np.sqrt(np.dot(R,R))
    glScale(self.zoomdist,self.zoomdist,1)
    glTranslate(0, 0,-10.5)
    glRotatef(self.rotangles[0], *R)
    glRotatef(self.rotangles[1], *self.upvec)
    glTranslate(*-self.lookat)
window.set_camera = lambda : set_camera(window)


@window.event
def pre_draw():
    mat = glGetFloatv(GL_MODELVIEW_MATRIX).transpose()
    Z = -np.cross(mat[0,:3],window.upvec)
    X = -np.cross(window.upvec, Z)
    Z /= np.sqrt(np.dot(Z,Z))
    X /= np.sqrt(np.dot(X,X))

    dt = 0.03
    keyspeed = 0.3
    global sink
    if keymask[wx.WXK_LEFT] or keymask[ord('j')]:
        sink -= keyspeed*dt*X
    if keymask[wx.WXK_RIGHT] or keymask[ord('l')]:
        sink += keyspeed*dt*X
    if keymask[wx.WXK_DOWN] or keymask[ord('k')]:
        sink -= keyspeed*dt*Z
    if keymask[wx.WXK_UP] or keymask[ord('i')]:
        sink += keyspeed*dt*Z
    window.lookat = (sink+source)/2
    window.set_camera()
    if np.any(keymask.values()):
        window.Refresh()


if not 'keymask' in globals():
    keymask = {}
    for x in (wx.WXK_RIGHT, wx.WXK_DOWN, wx.WXK_UP, wx.WXK_LEFT,
              ord('j'), ord('k'), ord('l'), ord('i')):
        keymask[x] = 0


@window.eventx
def EVT_KEY_DOWN(evt):
    keymask[evt.GetKeyCode()] = 1
    window.Refresh()
    evt.Skip()


@window.eventx
def EVT_CHAR(evt):
    key = evt.GetKeyCode()
    if key == ord(' '):
        find_direct_ray()
        window.Refresh()


@window.eventx
def EVT_KEY_UP(evt):
    keymask[evt.GetKeyCode()] = 0
    window.Refresh()
    evt.Skip()


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
    glColor(1,1,1,1)
    glEnable(GL_CULL_FACE)
    #mat = glGetFloatv(GL_MODELVIEW_MATRIX).transpose()
    glCallList(mesh.obj.gl_list)

    glPushMatrix()
    glTranslate(*sink)
    glColor(1,0,0)
    gluSphere(gluNewQuadric(), sinkrad, 10, 10)
    glPopMatrix()

    glPushMatrix()
    glTranslate(*source)
    glColor(0.2,0.2,0.6)
    gluSphere(gluNewQuadric(), sinkrad, 10, 10)
    glPopMatrix()

    glDisable(GL_LIGHTING)

    #find_direct_ray()

    glPushAttrib(GL_ALL_ATTRIB_BITS)
    if 'paths' in globals():
        glLineWidth(2)
        glBegin(GL_LINES)
        for path in paths:
            x1 = path[0]['origin']
            #orgn = True
            for d in path[1:]:
                ntype = d['ntype']
                origin = d['origin']
                if ntype == 'sink': glColor(1,.7,.7)
                #elif orgn: glColor(.7,.7,1); orgn=False
                else: glColor(1,1,1)
                x2 = origin
                glVertex(*x1)
                glVertex(*x2)
                x1 = x2
        glEnd()

    if 'ray' in globals():
        glColor(1,1,0)
        x1,d = ray['origin'], ray['direction']
        t = ray['t']
        if t > 1000: t = 1000

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


window.Refresh()
