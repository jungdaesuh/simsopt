#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include "magneticfield.h"
#include "boozermagneticfield.h"
#include "regular_grid_interpolant_3d.h"

using std::shared_ptr;
using std::vector;
using std::tuple;

double get_phi(double x, double y, double phi_near);

class StoppingCriterion {
    public:
        // Should return true if the Criterion is satisfied.
        virtual bool operator()(int iter, double t, double x, double y, double z) = 0;
        virtual bool supports_dense_rootfinding() const {
            return false;
        }
        virtual double dense_root_value(double t, double x, double y, double z) const {
            return 1.0;
        }
        virtual int dense_root_refinement_depth(
            double t0,
            double x0,
            double y0,
            double z0,
            double t1,
            double x1,
            double y1,
            double z1
        ) const {
            (void)t0;
            (void)x0;
            (void)y0;
            (void)z0;
            (void)t1;
            (void)x1;
            (void)y1;
            (void)z1;
            return 12;
        }
        virtual ~StoppingCriterion() {}
};

class ToroidalTransitStoppingCriterion : public StoppingCriterion {
    private:
        double max_transits;
        double phi_last;
        double phi_init;
        bool flux;
    public:
        ToroidalTransitStoppingCriterion(double max_transits, bool flux) : max_transits(max_transits), flux(flux) {
        };
        bool operator()(int iter, double t, double x, double y, double z) override {
            if (iter == 1) {
              phi_last = M_PI;
            }
            double phi = z;
            if (!flux) {
              phi = get_phi(x, y, phi_last);
            }
            if (iter == 1) {
              phi_init = phi;
            }
            phi_last = phi;
            double ntransits = std::abs((phi - phi_init) / (2 * M_PI));
            return ntransits >= max_transits;
        };
};

class MaxToroidalFluxStoppingCriterion : public StoppingCriterion{
    private:
        double max_s;
    public:
        MaxToroidalFluxStoppingCriterion(double max_s) : max_s(max_s) {};
        bool operator()(int iter, double t, double s, double theta, double zeta) override {
            return s>=max_s;
        };
};

class MinToroidalFluxStoppingCriterion : public StoppingCriterion{
    private:
        double min_s;
    public:
        MinToroidalFluxStoppingCriterion(double min_s) : min_s(min_s) {};
        bool operator()(int iter, double t, double s, double theta, double zeta) override {
            return s<=min_s;
        };
};

class MinZStoppingCriterion : public StoppingCriterion{
    private:
        double crit_z;
    public:
        MinZStoppingCriterion(double crit_z) : crit_z(crit_z) {};
        bool operator()(int iter, double t, double x, double y, double z) override {
            return z<=crit_z;
        };
};

class MaxZStoppingCriterion : public StoppingCriterion{
    private:
        double crit_z;
    public:
        MaxZStoppingCriterion(double crit_z) : crit_z(crit_z) {};
        bool operator()(int iter, double t, double x, double y, double z) override {
            return z>=crit_z;
        };
};

class MinRStoppingCriterion : public StoppingCriterion{
    private:
        double crit_r;
    public:
        MinRStoppingCriterion(double crit_r) : crit_r(crit_r) {};
        bool operator()(int iter, double t, double x, double y, double z) override {
            return std::sqrt(x*x+y*y)<=crit_r;            
        };
};

class MaxRStoppingCriterion : public StoppingCriterion{
    private:
        double crit_r;
    public:
        MaxRStoppingCriterion(double crit_r) : crit_r(crit_r) {};
        bool operator()(int iter, double t, double x, double y, double z) override {
            return std::sqrt(x*x+y*y)>=crit_r;            
        };
};

class IterationStoppingCriterion : public StoppingCriterion{
    private:
        int max_iter;
    public:
        IterationStoppingCriterion(int max_iter) : max_iter(max_iter) {};
        bool operator()(int iter, double t, double x, double y, double z) override {
            return iter>=max_iter;
        };
};

template<class Array>
class LevelsetStoppingCriterion : public StoppingCriterion{
    private:
        shared_ptr<RegularGridInterpolant3D<Array>> levelset;
    public:
        LevelsetStoppingCriterion(shared_ptr<RegularGridInterpolant3D<Array>> levelset) : levelset(levelset) { };
        bool supports_dense_rootfinding() const override {
            return true;
        }
        double dense_root_value(double t, double x, double y, double z) const override {
            (void)t;
            double r = std::sqrt(x*x + y*y);
            double phi = std::atan2(y, x);
            if(phi < 0)
                phi += 2*M_PI;
            return levelset->evaluate(r, phi, z)[0];
        };
        int dense_root_refinement_depth(
            double t0,
            double x0,
            double y0,
            double z0,
            double t1,
            double x1,
            double y1,
            double z1
        ) const override {
            (void)t0;
            (void)t1;
            double r0 = std::sqrt(x0*x0 + y0*y0);
            double r1 = std::sqrt(x1*x1 + y1*y1);
            double phi0 = std::atan2(y0, x0);
            if(phi0 < 0)
                phi0 += 2*M_PI;
            double phi1 = get_phi(x1, y1, phi0);
            constexpr double refinement_safety_factor = 4.0;
            constexpr int refinement_depth_cap = 20;
            double required_segments = refinement_safety_factor * std::max({
                1.0,
                std::abs(r1-r0) / std::max(levelset->cell_width_x(), 1e-12),
                std::abs(phi1-phi0) / std::max(levelset->cell_width_y(), 1e-12),
                std::abs(z1-z0) / std::max(levelset->cell_width_z(), 1e-12),
            });
            int depth = int(std::ceil(std::log2(required_segments)));
            if (depth < 1)
                depth = 1;
            if (depth > refinement_depth_cap)
                depth = refinement_depth_cap;
            return depth;
        };
        bool operator()(int iter, double t, double x, double y, double z) override {
            double f = dense_root_value(t, x, y, z);
            //fmt::print("Levelset at xyz=({}, {}, {}), rphiz=({}, {}, {}), f={}\n", x, y, z, r, phi, z, f);
            return f<0;
        };
};

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 5>>, vector<array<double, 6>>>
particle_guiding_center_boozer_tracing(
        shared_ptr<BoozerMagneticField<T>> field, array<double, 3> stz_init,
        double m, double q, double vtotal, double vtang, double tmax, double tol,
        bool vacuum, bool noK, vector<double> zetas, vector<shared_ptr<StoppingCriterion>> stopping_criteria);

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 5>>, vector<array<double, 6>>>
particle_guiding_center_tracing(
        shared_ptr<MagneticField<T>> field, array<double, 3> xyz_init,
        double m, double q, double vtotal, double vtang, double tmax, double tol, bool vacuum,
        vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 7>>, vector<array<double, 8>>>
particle_fullorbit_tracing(
        shared_ptr<MagneticField<T>> field, array<double, 3> xyz_init, array<double, 3> v_init,
        double m, double q, double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 4>>, vector<array<double, 5>>>
fieldline_tracing(
        shared_ptr<MagneticField<T>> field, array<double, 3> xyz_init,
        double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);
