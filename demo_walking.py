from visuals.pointwindow import PointWindow
from OpenGL.GL import *
from OpenGL.GLU import *
import cv
import mesh
import wx

if not 'window' in globals():
    window = PointWindow(size=(640,480))


meshname = 'hall1'
mesh.load(meshname)


def reset_sink():
    global sink
    sink = np.array([0,0,0],'f')

if not 'sink' in globals():
    reset_sink()
    window.Refresh()

if not 'source' in globals():
    source = np.array([1,0,0],'f')


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
    window.lookat = sink
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


@window.eventx
def EVT_KEY_UP(evt):
    keymask[evt.GetKeyCode()] = 0
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
    glColor(1,1,1,1)
    glEnable(GL_CULL_FACE)
    #mat = glGetFloatv(GL_MODELVIEW_MATRIX).transpose()
    glCallList(mesh.obj.gl_list)

    glPushMatrix()
    glTranslate(*sink)
    glColor(1,0,0)
    gluSphere(gluNewQuadric(), 0.3, 10, 10)
    glPopMatrix()
    
    glPushMatrix()
    glColor(0.2,0.2,0.6)
    gluSphere(gluNewQuadric(), 0.3, 10, 10)
    glPopMatrix()

    glDisable(GL_LIGHTING)

window.Refresh()
