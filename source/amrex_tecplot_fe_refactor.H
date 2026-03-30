#ifndef AMREX_TECPLOT_FE_REFACTOR_H_
#define AMREX_TECPLOT_FE_REFACTOR_H_

#include <string>

#include <AMReX_REAL.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

namespace amr_tecplot_fe {

struct TecplotFEOptions
{
    std::string zone_name = "solution";
    bool connect_cc = true;
    int finest_level = -1;
    amrex::Vector<int> comps;
    
    bool write_binary = true;
    bool binary_point = false;   // false = FEBLOCK, true = FEPOINT
};

void WriteFEFromMultiFab(
    const std::string& outfile,
    const amrex::Vector<const amrex::MultiFab*>& state,
    const amrex::Vector<amrex::Geometry>& geom,
    const amrex::Vector<int>& ref_ratio,
    const amrex::Vector<std::string>& var_names,
    amrex::Real time,
    const TecplotFEOptions& opt = TecplotFEOptions());

} // namespace amr_tecplot_fe

#endif
