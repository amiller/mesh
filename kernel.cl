float4 matmul(global const float4 *m, const float4 r1) {
	return (float4)(dot(m[0],r1),dot(m[1],r1),dot(m[2],r1),dot(m[3],r1));
}

__constant float EPSILON = 1e-5;
__constant float PI = 3.1415;

float intersect_ray_triangle(float4 origin, float4 dir, 
     			    float4 v0, float4 v1, float4 v2)
{
	// Fast Minimum Storage Ray/Triangle Intersection
	// by Moller and Trumbore

	float4 e1 = v1 - v0;
	float4 e2 = v2 - v0;
	float4 pvec = cross(dir,e2);
	float det = dot(e1,pvec);

	if (det > -EPSILON && det < EPSILON) return MAXFLOAT;
	float invdet = 1.0 / det;

	float4 tvec = origin - v0;
	float u = dot(tvec,pvec) * invdet;
	if (u < 0.0 || u > 1.0) return MAXFLOAT;

	float4 qvec = cross(tvec, e1);
	float v = dot(dir,qvec) * invdet;
	if (v < 0.0 || u + v > 1.0) return MAXFLOAT;

	float t = dot(e2, qvec) * invdet;
	if (t > EPSILON) return t;
	else return MAXFLOAT;
}

// Main Kernel code
kernel void raytrace(
	global float4 *img,
	global const float4 *m,	// inverted camera matrix
	global const float4 *minv,
	
	const int num_faces,
	global const float4 *vertices,
	global const int4 *faces
)
{	
	unsigned int x = get_global_id(1);
	unsigned int y = get_global_id(0);
	unsigned int width  = get_global_size(1);
	unsigned int height = get_global_size(0);
	unsigned int index = (y * width) + x;

	// 0.5 increment is to reach the center of the pixel.
	float u = (((x-0.0) / width)*2.0f-1.0f)/1.2999;
	float v = -(((y-0.0) / height)*2.0f-1.0f)/1.732;

	// The camera center
	float4 p = matmul(minv, (float4)(0,0,0,1));
	p.s3 = 0;

	// Ray vector assuming 90 degree viewing angle 
	// (image plane at z=-1)
	float4 d = matmul(minv, (float4)(u,v,-1,0));
	d.s3 = 0; d = normalize(d);
	
	//img[index] = minv[2]; return;
	
	float4 color = (float4)(0);
	color = fabs(d);
	float mint = MAXFLOAT;
	int mini;
	for (int i = 0; i < num_faces; i++) {
		int4 face = faces[i];
		float4 v0 = vertices[face.s0];
		float4 v1 = vertices[face.s1];
		float4 v2 = vertices[face.s2];
		
		float rI = intersect_ray_triangle(p,d,v0,v1,v2);
		bool hit = (rI < MAXFLOAT);
		if (!hit) continue;

		if (rI <= mint) {
		   mint = rI;	
		   mini = i;
		}
	}
	
	// Find the normal, if we have a hit
	if (mint < MAXFLOAT) {
	   	color = mint/3;
		//color = (float4)(mini)/255;
	} else {
		//color = (float4)(0,0,0,0);*/
	}
	
	img[index] = color;
}
