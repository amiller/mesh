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
queue = cl.CommandQueue(context)
mf = cl.mem_flags
sampler = cl.Sampler(context, True, cl.addressing_mode.CLAMP, cl.filter_mode.LINEAR)

with open('kernel.cl') as f: 
    kernel = f.read()

program = cl.Program(context, kernel).build("-cl-mad-enable")
print program.get_build_info(context.devices[0], cl.program_build_info.LOG)

def print_all():
    print_info(context.devices[0], cl.device_info)
    print_info(program, cl.program_info)
    print_info(queue, cl.command_queue_info)
    
    
# I have no explanation for this workaround. Presumably it's fixed in 
# another version of pyopencl. Wtf. Getting the kernel like this
# makes it go much faster when we __call__ it.
def bullshit(self):
    return self
cl.Kernel.bullshit = bullshit
#program.flatrot_compute = program.flatrot_compute.bullshit()


#print_all()
H,W = 480,640
img_buf    = cl.Buffer(context, mf.READ_WRITE, H*W*4*4)

mat_buf = cl.Buffer(context, mf.READ_WRITE, 4*4*4)
inv_mat_buf = cl.Buffer(context, mf.READ_WRITE, 4*4*4) 

reduce_buf    = cl.Buffer(context, mf.READ_WRITE, 8*4*100)
reduce_scratch = cl.LocalMemory(64*8*4)


def load_mesh(vertices, faces):
    assert vertices.dtype == np.float32
    assert vertices.shape[1] == 4
    assert vertices.flags['C_CONTIGUOUS']
    assert faces.dtype == np.int32
    assert faces.shape[1] == 4
    assert faces.flags['C_CONTIGUOUS']


    global vert_buf, face_buf, num_faces
    num_faces = faces.shape[0]
    vert_buf = cl.Buffer(context, mf.READ_WRITE, vertices.shape[0]*4*4)
    face_buf = cl.Buffer(context, mf.READ_WRITE, faces.shape[0]*4*4)

    evt = cl.enqueue_write_buffer(queue, vert_buf, vertices, 
                                  is_blocking=True)
    evt = cl.enqueue_write_buffer(queue, face_buf, faces, 
                                  is_blocking=True)
    return evt

def get_img():
    img = np.empty((H,W,4),'f')
    cl.enqueue_read_buffer(queue, img_buf, img).wait()
    return img

def compute_raytrace(mat):
    assert mat.dtype == np.float32
    assert mat.shape == (4,4)
    mat = np.ascontiguousarray(mat)

    cl.enqueue_write_buffer(queue, mat_buf, mat)
    cl.enqueue_write_buffer(queue, inv_mat_buf, 
                            np.ascontiguousarray(np.linalg.inv(mat))).wait()
    
    evt = program.raytrace(queue, (H,W), None,
                           img_buf,
                           mat_buf, inv_mat_buf,
                           np.int32(num_faces), vert_buf, face_buf)
                           
    evt.wait()
    return evt
