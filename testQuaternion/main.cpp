#include <iostream>
// #include <Eigen/Dense>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
// #include <Eigen/Geometry>
using namespace std;

int main()
{
    Eigen::Quaterniond q=Eigen::Quaterniond(0,0,1,0);
    // st<<1,2-3,4;
    cout<<q.coeffs()<<endl;
    cout<<q.inverse().coeffs()<<endl;
    cout<<(q.inverse()*q).coeffs()<<endl;
    cout<<"============================\n";
    Eigen::Isometry3d T(q);
    cout<<T.matrix()<<endl;
    T.pretranslate(Eigen::Vector3d(1,0,0));
    cout<<T.rotation()<<endl;
    cout<<T.translation()<<endl;
    return 0;
}