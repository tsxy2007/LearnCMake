#pragma once 

#include "ray.h"

class camera
{
public:
    __host__ __device__ camera();
    __host__ __device__ ray get_ray(double s, double t);
public:
    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
};

camera::camera()
{
    lower_left_corner = vec3(-2.0, -1.0, -1.0);
    horizontal = vec3(4.0, 0.0, 0.0);
    vertical = vec3(0.0, 2.0, 0.0);
    origin = vec3(0.0, 0.0, 0.0);
}

ray camera::get_ray(double s, double t)
{
    return ray(origin, lower_left_corner + horizontal * s + vertical * t );
}