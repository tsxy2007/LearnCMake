// File: MyClass.h created: 2025-08-12
#ifndef _vec3__H_
#define _vec3__H_

#include <cmath> 
#include <cstdlib>
#include <iostream>


class vec3
{
public:
    __host__ __device__  vec3(){ 
        x = 0; 
        y = 0;
        z = 0;
    };
    __host__ __device__  vec3(float e0,float e1,float e2) 
    { 
        x = e0; 
        y = e1;
        z = e2;
    }
    __host__ __device__  ~vec3(){};

    __host__ __device__  inline float r() const {return x;}
    __host__ __device__  inline float g() const {return y;}
    __host__ __device__  inline float b() const {return z;}

    __host__ __device__  inline const vec3& operator+() const {return *this;}
    __host__ __device__  inline vec3 operator-()const {return vec3(-x,-y,-z);}
    __host__ __device__  inline float operator[](int i) const 
    {
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                return x;
        }
    }
    __host__ __device__  inline float& operator[](int i)
    { 
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                return x;
        }
    }
 
    __host__ __device__  inline vec3& operator+=(const vec3& v2);
    __host__ __device__  inline vec3& operator-=(const vec3& v2);
    __host__ __device__  inline vec3& operator*=(const vec3& v2);
    __host__ __device__  inline vec3& operator/=(const vec3& v2);
    __host__ __device__  inline vec3& operator*=(const float t);
    __host__ __device__  inline vec3& operator/=(const float t);

    __host__ __device__  inline float length()const{
        return sqrt(x*x+y*y+z*z);
    }
    __host__ __device__  inline float squared_length() const{
        return x*x+y*y+z*z;
    }
    __host__ __device__  inline void make_unit_vector();

    __host__ __device__  inline void Log() const
    {
        printf("vec3 = {%f %f %f}\n",x,y,z);
    }
    __host__ __device__  auto normalize() const -> vec3 {
        float len = length();
        return vec3(x/len, y/len, z/len);
    }
public:
    float x , y , z;
};


__host__ __device__  inline std::istream& operator>>(std::istream& is, vec3& t){
    is>>t.x>>t.y>>t.z;
    return is;
}

__host__  inline std::ostream& operator<<(std::ostream& os , const vec3& t){
    os<<t.x<<" "<<t.y<<" "<<t.z;
    return os;
}

__host__ __device__  inline void vec3::make_unit_vector(){
    float k = 1.0 / length();
    x *=k;
    y *=k;
    z *=k;
}

__host__ __device__  inline vec3 operator+(const vec3& v1,const vec3& v2){
    return vec3(v1.x + v2.x,v1.y+v2.y,v1.z+v2.z);
}

__host__ __device__  inline vec3 operator-(const vec3& v1,const vec3& v2){
    return vec3(v1.x - v2.x,v1.y - v2.y,v1.z - v2.z);
}

__host__ __device__  inline vec3 operator*(const vec3& v1,const vec3& v2){
    return vec3(v1.x * v2.x,v1.y * v2.y,v1.z * v2.z);
}

__host__ __device__  inline vec3 operator/(const vec3& v1,const vec3& v2){
    return vec3(v1.x / v2.x,v1.y / v2.y,v1.z / v2.z);
}

__host__ __device__  inline vec3 operator*(const vec3& v1,float t)
{
    return vec3(v1.x * t,v1.y * t,v1.z * t);
}

__host__ __device__  inline vec3 operator*(float t ,const vec3& v1)
{
    return vec3(v1.x * t,v1.y * t,v1.z * t);
}

__host__ __device__  inline vec3 operator/(const vec3& v1,float t)
{
    return vec3(v1.x / t,v1.y / t,v1.z / t);
}

__host__ __device__  inline float dot(const vec3& v1,const vec3& v2){
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__  inline vec3 cross(const vec3& v1,const vec3& v2){
    return {(v1.y* v2.z - v1.z*v2.y), 
                -(v1.x*v2.z - v1.z * v2.x),
                (v1.x * v2.y - v1.y*v2.x)
            };
}

inline vec3& vec3::operator+=(const vec3& v2){
    x += v2.x;
    y += v2.y;
    z += v2.z;
    return *this;
}

inline vec3& vec3::operator-=(const vec3& v2){
    x -= v2.x;
    y -= v2.y;
    z -= v2.z;
    return *this;
}

inline vec3& vec3::operator*=(const vec3& v2){
    x *= v2.x;
    y *= v2.y;
    z *= v2.z;
    return *this;
}

inline vec3& vec3::operator/=(const vec3& v2){
    x /= v2.x;
    y /= v2.y;
    z /= v2.z;
    return *this;
}
    
inline vec3& vec3::operator*=(const float t){
    x *= t;
    y *= t;
    z *= t;
    return *this;
}

inline vec3& vec3::operator/=(const float t){
    x /= t;
    y /= t;
    z /= t;
    return *this;
}

__host__ __device__   inline vec3 unit_vector(vec3  v){
    return v / v.length();
}

#endif
