#pragma once
#include "hitable.h"
#include "vec3.h"
#include <cmath>

class sphere : public hitable
{
public:
    __host__ __device__ sphere(): hitable(){}
    __host__ __device__ sphere(vec3 cen, float r) : hitable(), center(cen),radius(r){};

    __host__ __device__ virtual bool hit(const ray& r,float tmin,float tmax,hit_record& rec) const override;
public:
    vec3 center;
    float radius;
};

bool sphere::hit(const ray& r,float tmin,float tmax,hit_record& rec) const
{
    // printf("sphere::tmin = %f  tmax = %f\n",tmin,tmax);
    vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc,r.direction());
    float c = dot(oc,oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;  
    if (discriminant > 0) 
    {
        float temp = (-b - sqrt(b*b - a*c)) / a;
        if (temp < tmax && temp > tmin) 
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p- center)/ radius;
            return true;
        }
        temp = (-b+sqrt(b*b-a*c)) / a ;
        if (temp < tmax && temp>tmin) 
        {
            rec.t = temp;
            rec.p= r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) /radius;
            return true;
        }
    }

    return false;
}