from visuals.pointwindow import PointWindow
from OpenGL.GL import *
from OpenGL.GLU import *
import cv
import mesh
import wx
import cybvh
import random
import quantities as pq
import sounds

if not 'window' in globals():
    window = PointWindow(size=(640,480))


ROULETTE=0.2

meshname = 'ellipse'
mesh.load(meshname)
mycybvh = cybvh.CyBVH(mesh.mybvh.verts[:,:3].copy(),
                      np.array(mesh.mybvh.tris)[:,:3].copy(),
                      mesh.mybvh.nodes)


def reset_sink():
    global sink, sinkrad, source
    sinkrad = 0.8
    source = np.array([0,0,0],'f')
    sink = np.array([0,0,0],'f')

if not 'sink' in globals():
    reset_sink()
    window.Refresh()


def random_ray():
    """Generate some rays and intersect with the world. If it intersects,
    make a specular reflections. Check for intersection with the sphere.
    """
    d = np.random.rand(3).astype('f')-.5
    d /= np.sqrt(np.dot(d,d))
    t = cybvh.intersect_sphere(sinkrad, sink, source, d)
    global ray
    ray = dict(direction=d,origin=source,t=t)
    #window.Refresh()


def find_direct_ray():
    for i in range(10000):
        random_ray()
        if ray['t'] < inf:
            break

from scipy import stats


def pink1d(n, rvs=stats.norm.rvs):
    k = min(int(np.floor(np.log2(n))), 6)
    pink = np.zeros((n,), 'f')
    m = 1
    for i in range(k):
        p = int(np.ceil(float(n) / m))
        pink += np.repeat(rvs(size=p), m,axis=0)[:n]
        m <<= 1
    return pink/k
pinksample = pink1d(sounds.filter_len)


def update_filter(identity=False):
    #import pdb
    #pdb.set_trace()
    times, amp = energy_contributions()
    global H
    if identity:
        H = np.zeros(sounds.filter_len,'f')
        H[0]=1
    else:
        H,_ = np.histogram(times,
                           weights=amp,
                           bins=sounds.filter_len,
                           range=(0,1.*sounds.filter_len/sounds.rate))
    filt = H.astype('f')
    #filt /= np.abs(filt).sum()
    #filt *= (np.random.rand(sounds.filter_len)-0.5)*2
    #filt *= pink1d(sounds.filter_len)
    filt *= pinksample*80
    sounds.set_filter(filt)


def accumulate(reset=True):
    import pylab
    sample_rays(1000, reset=reset);
    try:
        while True:
            sample_rays(1000);
            draw()
    except KeyboardInterrupt:
        pass
    draw()


def energy_contributions():

    global paths, dists, total_rays

    # Find the time for each impulse based on the speed of sound and distance
    dists = np.array([p[-1][-1] for p in paths])
    times = dists*pq.m / (343*pq.m/pq.s)

    # Find the energy using sound propagation
    #dB = 20 * pq.micro * pq.pascal
    # attenuation = np.exp()

    # Divide by roulette compensation factor
    bounces = np.array([len(p)-2 for p in paths])*0
    pressure_contribution = np.sqrt((1/(1-ROULETTE))**bounces)/total_rays

    return times, pressure_contribution


def sample_rays(n_rays=10000, reset=False):
    global paths
    global total_rays
    global line_verts, line_colors

    if reset or not 'line_verts' in globals():
        paths = []
        total_rays = 0
        line_verts = np.empty((0,3),'f')
        line_colors = np.empty((0,3),'f')
    total_rays += n_rays
    line_verts_ = []
    line_colors_ = []

    ps = mycybvh.sample_rays(source, sink, sinkrad, n_rays, ROULETTE)
    keys = ['source','sink','diverge','scaflect']

    for path in ps:
        p_ = []
        x1 = path[0]['origin']
        x1 = x1['x'], x1['y'], x1['z']
        orgn = True
        for p in path[1:]:
            o = p['origin']
            d = p['direction']
            ntype = keys[p['ntype']]
            cumdist = p['cumdist']
            origin = o['x'],o['y'],o['z']
            direction = d['x'],d['y'],d['z']
            x2 = origin
            if ntype == 'sink':
                line_colors_ += 2*((1,.6,.6),)
            elif orgn:
                line_colors_ += 2*((.6,.6,1),)
                orgn=False
            else:
                line_colors_ += 2*((1,1,1),)
            x2 = origin
            line_verts_.append(x1)
            line_verts_.append(x2)
            x1 = x2
            p_.append((origin, direction, ntype, cumdist))
        paths.append(p_)
    if line_colors_:
        line_colors = np.vstack((line_colors, np.array(line_colors_,'f')))
        line_verts = np.vstack((line_verts, np.array(line_verts_,'f')))
    window.Refresh()

    pylab.clf();
    times, pressure = energy_contributions()
    pylab.hist(times,weights=pressure,bins=100, range=(0,0.2))
    pylab.waitforbuttonpress(0.03)
    update_filter()


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
        sample_rays(30000,True)
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

    glPushAttrib(GL_ALL_ATTRIB_BITS)
    if 'line_verts' in globals() and len(line_verts):
        glLineWidth(1)
        glVertexPointerf(line_verts)
        glColorPointerf(line_colors)
        glDisableClientState(GL_NORMAL_ARRAY)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glDrawElementsui(GL_LINES, np.arange(len(line_verts))[-1000:])
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    glPopAttrib(GL_ALL_ATTRIB_BITS)


window.Refresh()
