#pragma once 

#include "hitable.h"
#include "sphere.h"
#include "vec3.h"
#include <cstdio>

class hitable_list : public hitable
{
public:
    __host__ __device__ hitable_list(){}
    __host__ __device__ hitable_list(sphere** l,int n)
    { 
        list = l ; 
        list_size = n;
        // printf("initialize hitable_list\n");
    }
    __host__ __device__ virtual bool hit(const ray& r,float t_min,float t_max,hit_record& rec) const override ;

public:
    sphere** list;
    int list_size;
};

bool hitable_list::hit(const ray& r,float t_min,float t_max,hit_record& rec) const
{
    // printf("hitable_list::hit size = %d\n",list_size);
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0 ; i < list_size; i++) 
    {
        sphere* s = list[i];
        if (s->hit(r, t_min, closest_so_far, temp_rec)) 
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}