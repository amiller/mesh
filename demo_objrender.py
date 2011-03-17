from visuals.pointwindow import PointWindow
from OpenGL.GL import *
import opencl
import cv
import mesh


if not 'window' in globals():
    window = PointWindow(size=(640,480))


def load_obj():
    mesh.load_random()
    window.update_points(mesh.points[:,:3])
    window.lookat = mesh.points[:,:3].mean(0)
    opencl.load_mesh(mesh.vertices, mesh.faces)
    window.Refresh()


if not 'obj' in mesh.__dict__: load_obj()


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
    glScale(1.0/mesh.scale,1.0/mesh.scale,1.0/mesh.scale)
    glCallList(mesh.obj.gl_list)
    glPopMatrix()
    glDisable(GL_LIGHTING)

    if 1:
        glColor(1,1,0)
        mesh.mybvh.draw_boxes()

    if 'ray' in globals():
        glColor(1,1,0)
        nodes, tri, t = mesh.mybvh.intersect(*ray)
        x1,d = ray
        if t > 1000: t = 1000

        verts, line_inds, _ = mesh.bvh.BVH.box_vertices(nodes)
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
