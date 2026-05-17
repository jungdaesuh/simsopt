#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;
Array ReimanB(double& iota0, double& iota1, Array& k_theta, Array& epsilon, int& m0_symmetry, Array& points){
    int num_points = points.shape(0);
    int num_coeffs = k_theta.shape(0);
    Array B        = xt::zeros<double>({points.shape(0), points.shape(1)});
    const double R_axis = 1.0;
    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        const double x      = points(i, 0);
        const double y      = points(i, 1);
        const double ZZ     = points(i, 2);
        const double RR     = sqrt(x*x+y*y);
        const double cosphi = x/RR;
        const double sinphi = y/RR;
        const double varphi = atan2(y,x);
        const double theta  = atan2(ZZ, RR - R_axis);
        const double rmin = sqrt(pow((RR-R_axis), 2.0) + pow(ZZ, 2.0));

        double combo = iota0 + iota1*rmin*rmin;
        double combo1 = 0.0;

        for (int ind=0; ind < num_coeffs; ++ind) {
            const double kth = k_theta[ind];
            const double angle = kth*theta - m0_symmetry*varphi;
            const double scaled_rpow = kth * epsilon[ind] * pow(rmin, kth - 2);
            combo       -= scaled_rpow * cos(angle);
            combo1      += scaled_rpow * sin(angle);
        }

        const double BR   =  ( (RR - R_axis)/RR)*combo1 + (ZZ/RR)*combo ;
        const double BZ   = -( (RR - R_axis)/RR)*combo  + (ZZ/RR)*combo1;
        const double Bphi = -1.0;

        B(i,0) = BR*cosphi-Bphi*sinphi;
        B(i,1) = BR*sinphi+Bphi*cosphi;
        B(i,2) = BZ;
    }
    return B;
}

Array ReimandB(double& iota0, double& iota1, Array& k_theta, Array& epsilon, int& m0_symmetry, Array& points){
    int num_points = points.shape(0);
    int num_coeffs = k_theta.shape(0);
    Array dB       = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    const double R_axis = 1.0;
    #pragma omp parallel for
    for (int i = 0; i < num_points; ++i) {
        const double x      = points(i, 0);
        const double y      = points(i, 1);
        const double ZZ     = points(i, 2);
        const double RR     = sqrt(x*x+y*y);
        const double cosphi = x/RR;
        const double sinphi = y/RR;
        const double varphi = atan2(y,x);
        const double theta  = atan2(ZZ, RR - R_axis);
        const double rmin = sqrt(pow((RR-R_axis), 2.0) + pow(ZZ, 2.0));

        double combo = iota0 + iota1*rmin*rmin;
        double combo1 = 0.0;

        double dcombodR = 2.0*iota1*(RR - R_axis);
        double dcombodZ = 2.0*iota1*ZZ;
        double dcombodphi  = 0.0;
        double dcombo1dR   = 0.0;
        double dcombo1dZ   = 0.0;
        double dcombo1dphi = 0.0;

        for (int ind=0; ind < num_coeffs; ++ind) {
            const double kth = k_theta[ind];
            const double eps = epsilon[ind];
            const double angle = kth*theta - m0_symmetry*varphi;
            const double cos_angle = cos(angle);
            const double sin_angle = sin(angle);
            const double scaled_rpow_m2 = kth * eps * pow(rmin, kth - 2);
            const double scaled_rpow_m4 = kth * eps * pow(rmin, kth - 4);
            combo       -= scaled_rpow_m2 * cos_angle;
            combo1      += scaled_rpow_m2 * sin_angle;
            dcombodR    -= scaled_rpow_m4 * (
                kth * ZZ * sin_angle + (kth - 2) * (RR - R_axis) * cos_angle
            );
            dcombodZ    += scaled_rpow_m4 * (
                kth * sin_angle * (RR - R_axis) - (kth - 2) * ZZ * cos_angle
            );
            dcombodphi  -= scaled_rpow_m2 * sin_angle*m0_symmetry;
            dcombo1dR   += scaled_rpow_m4 * (
                -kth * ZZ * cos_angle + (kth - 2) * sin_angle * (RR - R_axis)
            );
            dcombo1dZ   += scaled_rpow_m4 * (
                kth * cos_angle * (RR - R_axis) + (kth - 2) * sin_angle * ZZ
            );
            dcombo1dphi -= scaled_rpow_m2 * cos_angle*m0_symmetry;
        }

        const double BR   =  ( (RR - R_axis)/RR)*combo1 + (ZZ/RR)*combo ;
        const double BZ   = -( (RR - R_axis)/RR)*combo  + (ZZ/RR)*combo1;
        const double Bphi = -1.0;

        const double dRBR     = (-ZZ / pow(RR, 2.0)) * combo
            + (ZZ / RR) * dcombodR
            + combo1 * R_axis / pow(RR, 2.0)
            + dcombo1dR * (RR - R_axis) / RR;
        const double dZBR     = ( 1.0 / RR ) * combo + (ZZ / RR) * dcombodZ + dcombo1dZ * (RR - R_axis) / RR;
        const double dphiBR   = ( (RR - R_axis)/RR)*dcombo1dphi + (ZZ/RR)*dcombodphi;
        const double dRBZ     = (-R_axis / pow(RR, 2.0)) * combo
            - ((RR - R_axis) / RR) * dcombodR
            - combo1 * ZZ / pow(RR, 2.0)
            + dcombo1dR * ZZ / RR;
        const double dZBZ     = -((RR - R_axis) / RR) * dcombodZ
            + combo1 * (1.0 / RR)
            + dcombo1dZ * ZZ / RR;
        const double dphiBZ   = -( (RR - R_axis)/RR)*dcombodphi + (ZZ/RR)*dcombo1dphi;
        const double dRBphi   = 0.0;
        const double dZBphi   = 0.0;
        const double dphiBphi = 0.0;

        dB(i,0,0) = dRBR*cosphi*cosphi-(dphiBR-Bphi+dRBphi*RR)*cosphi*sinphi/RR+sinphi*sinphi*(dphiBphi+BR)/RR;
        dB(i,0,1) = sinphi*cosphi*(dRBR*RR-dphiBphi-BR)/RR+sinphi*sinphi*(Bphi-dphiBR)/RR+cosphi*cosphi*dRBphi;
        dB(i,0,2) = dRBZ*cosphi-dphiBZ*sinphi/RR;
        dB(i,1,0) = sinphi*cosphi*(dRBR*RR-dphiBphi-BR)/RR+cosphi*cosphi*(dphiBR-Bphi)/RR-sinphi*sinphi*dRBphi;
        dB(i,1,1) = dRBR*sinphi*sinphi+(dphiBR-Bphi+dRBphi*RR)*cosphi*sinphi/RR+cosphi*cosphi*(dphiBphi+BR)/RR;
        dB(i,1,2) = dRBZ*sinphi+dphiBZ*cosphi/RR;
        dB(i,2,0) = dZBR*cosphi-dZBphi*sinphi;
        dB(i,2,1) = dZBR*sinphi+dZBphi*cosphi;
        dB(i,2,2) = dZBZ;
    }
    return dB;
}
