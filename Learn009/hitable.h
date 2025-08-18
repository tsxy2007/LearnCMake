#pragma once

#include "ray.h"

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
};

class hitable
{
public:
__host__ __device__ hitable(){}
    __host__ __device__ virtual bool hit(const ray& r,float t_min,float t_max,hit_record& rec) const = 0 ;
};