import pyglet.gl
from OpenGL.GL import *
from OpenGL.GLU import *
from molecule import Molecule

import pyopencl as cl
import numpy as np

def print_info(obj, info_cls):
    for info_name in sorted(dir(info_cls)):
        if not info_name.startswith("_") and info_name != "to_string":
            info = getattr(info_cls, info_name)
            try:
                info_value = obj.get_info(info)
            except:
                info_value = "<error>"

            print "%s: %s" % (info_name, info_value)

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
print_info(context.devices[0], cl.device_info)
queue = cl.CommandQueue(context, 
	properties = cl.command_queue_properties.PROFILING_ENABLE)
mf = cl.mem_flags

N = 512

class CLRender(object):
	
	angles = [0,0,0]
	scale = 1
	mol = None
	env_buf = None
	
	def __init__(self):
		self.dst = np.empty((N,N,4)).astype(np.uint8)
		self.dst_buf = cl.Buffer(context, mf.WRITE_ONLY, self.dst.nbytes)
		self.inv_matrix = cl.Buffer(context, mf.READ_ONLY, 16 * 4)
		self.matrix = cl.Buffer(context, mf.READ_ONLY, 16 * 4)
		
		with open('kernel.cl','r') as f:
			self.program = cl.Program(context, f.read()).build("-cl-mad-enable")
		print self.program.get_build_info(context.devices[0], cl.program_build_info.LOG)
		
		self.dstTex = glGenTextures(1);
		glBindTexture(GL_TEXTURE_2D, self.dstTex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, None);
		glBindTexture(GL_TEXTURE_2D, 0);
		
		print_info(self.program, cl.program_info)
		print_info(self.program.pdbTracer, cl.kernel_info)
		
		grid = np.array(range(256),dtype=np.float32)/256
		x1,x2 = np.meshgrid(grid, grid)
		rad = np.sqrt(x1)
		phi = 2*np.pi * x2
		phimap = np.dstack((np.cos(phi)*rad, np.sin(phi)*rad, np.sqrt(1-rad*rad), 0*rad))
		self.p = phimap
		fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
		self.phimap = cl.Image(context, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, 
			shape=phimap.shape[:2], hostbuf=np.array(phimap, order='C'))

	def applySceneTransforms(self):
		gluLookAt(0, 0, 2*self.mol.radius, 0, 0, 0, 0, 1, 0); # Push molecule away from the origin along -Z direction.
		glScalef(self.scale,self.scale,self.scale);
		def mouse_rotate(xAngle, yAngle, zAngle):
			glRotatef(xAngle, 1.0, 0.0, 0.0);
			glRotatef(yAngle, 0.0, 1.0, 0.0);
			glRotatef(zAngle, 0.0, 0.0, 1.0);
		mouse_rotate(self.angles[0],self.angles[1],self.angles[2]);
		glTranslatef(-self.mol.x, -self.mol.y, -self.mol.z); # Bring molecue center to origin
		
	def render(self):
		glBindTexture(GL_TEXTURE_2D, self.dstTex)
		glEnable(GL_TEXTURE_2D)
		glBegin(GL_QUADS)
		glTexCoord2f( 0.0, 0.0 ); glVertex3f( -1.0, -1.0, -1.0 )
		glTexCoord2f( 0.0, 1.0 );	glVertex3f( -1.0, 1.0, -1.0 )
		glTexCoord2f( 1.0, 1.0 ); glVertex3f( 1.0, 1.0, -1.0 )
		glTexCoord2f( 1.0, 0.0 ); glVertex3f( 1.0, -1.0, -1.0 )
		glEnd()
		glDisable(GL_TEXTURE_2D)
		
	def compute(self):
		glMatrixMode(GL_MODELVIEW)
		glPushMatrix()
		glLoadIdentity()
		self.applySceneTransforms()
		mat = np.array(glGetFloat(GL_MODELVIEW_MATRIX).transpose(), order='C')
		glPopMatrix()
		inv = np.array(np.linalg.inv(mat), order='C')
		
		e1 = cl.enqueue_write_buffer(queue, self.matrix, mat)
		e2 = cl.enqueue_write_buffer(queue, self.inv_matrix, inv)
		
		e3 = self.program.pdbTracer(queue, self.dst.shape[:2], self.dst_buf,
			self.matrix, self.inv_matrix, 
			np.array(len(self.mol.spheres)), self.spheredata,
			self.envmap, self.phimap, self.sampler)
		e4 = cl.enqueue_read_buffer(queue, self.dst_buf, self.dst)
		queue.finish()
		e4.wait()

		for e in [e3]:
			print (e.profile.END - e.profile.START)*1e-9
		
		glBindTexture(GL_TEXTURE_2D, self.dstTex)
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, N, N, GL_RGBA, GL_UNSIGNED_BYTE, self.dst)
		
	def set_envmap(self, envmap):
		fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)
		em = np.zeros(envmap.shape[:2] + (4,), dtype=np.float32)
		em[:,:,:3] = envmap; em[:,:,3] = 1;
		self.envmap = cl.Image(context, mf.READ_ONLY | mf.COPY_HOST_PTR, fmt, 
			shape=em.shape[:2], hostbuf=em)
		self.sampler = cl.Sampler(context, True, cl.addressing_mode.CLAMP, cl.filter_mode.LINEAR)

	def set_molecule(self, mol):
		self.mol = mol
		self.spheredata = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, 
				hostbuf = self.mol.spheredata)

        def load_molecule(self, filename):
            self.set_molecule(Molecule(filename))
						
				
if __name__ == "__main__":

	from pfmloader import load_pfm
	r = CLRender()
	r.set_molecule(Molecule('data/sugars/sucrose.pdb'))
	r.set_envmap(load_pfm('data/probes/stpeters_probe.pfm'))
	r.compute()
