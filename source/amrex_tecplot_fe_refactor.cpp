// This file contains modified code derived from AMReX-related open-source
// FE export logic.
//
// Original upstream licenses and notices must be preserved where applicable.
//
// Modifications in this version include:
//   - direct export from existing MultiFab data
//   - rank-0 gather and serial write workflow
//   - Tecplot FE ASCII output
//   - optional TecIO binary output support
//
// Copyright (c) 2026 xhf111111

#include <new>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <fstream>
#include <algorithm>

#ifdef USE_TEC_BIN_IO
#include "TECIO.h"
#endif

#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>
#include <AMReX_Vector.H>
#include "amrex_tecplot_fe_refactor.H"

using namespace amrex;

namespace amr_tecplot_fe {

struct Node
{
    enum typeEnum { INIT=0, COVERED=1, VALID=2 };

    Node()
        : level(-1), iv(IntVect(AMREX_D_DECL(-1,-1,-1))), grid(-1), type(Node::INIT) {}

    Node(const IntVect& idx, int lev, int grd, typeEnum typ = INIT)
        : level(lev), iv(idx), grid(grd), type(typ) {}

    bool operator< (const Node& rhs) const
    {
        if (level < rhs.level) return true;
        if ((level == rhs.level) && iv < rhs.iv) return true;
        return false;
    }

    bool operator!= (const Node& rhs) const
    {
        return ((*this) < rhs || rhs < (*this));
    }

    int level;
    IntVect iv;
    int grid;
    typeEnum type;
};

struct Element
{
#if (AMREX_SPACEDIM == 2)
    static constexpr int NNode = 4;
    Element(const Node& a, const Node& b, const Node& c, const Node& d)
    {
        n[0]=&a; n[1]=&b; n[2]=&c; n[3]=&d;
    }
#else
    static constexpr int NNode = 8;
    Element(const Node& a, const Node& b, const Node& c, const Node& d,
            const Node& e, const Node& f, const Node& g, const Node& h)
    {
        n[0]=&a; n[1]=&b; n[2]=&c; n[3]=&d;
        n[4]=&e; n[5]=&f; n[6]=&g; n[7]=&h;
    }
#endif

    const Node* n[NNode];

    bool operator< (const Element& rhs) const
    {
        for (int i = 0; i < NNode; ++i) {
            if (*n[i] != *rhs.n[i]) return *n[i] < *rhs.n[i];
        }
        return false;
    }
};

static BoxArray GetBndryCells(const BoxArray& ba, int ngrow, const Geometry& geom)
{
    BoxList gcells, bcells;

    for (int i = 0; i < ba.size(); ++i) {
        gcells.join(amrex::boxDiff(amrex::grow(ba[i], ngrow), ba[i]));
    }

    for (BoxList::const_iterator it = gcells.begin(); it != gcells.end(); ++it)
    {
        std::vector<std::pair<int,Box> > isects = ba.intersections(*it);

        if (isects.empty()) {
            bcells.push_back(*it);
        } else {
            BoxList pieces;
            for (int i = 0; i < isects.size(); ++i) pieces.push_back(isects[i].second);
            BoxList leftover = amrex::complementIn(*it, pieces);
            bcells.catenate(leftover);
        }
    }

    gcells = amrex::removeOverlap(bcells);
    bcells.clear();

    if (geom.isAnyPeriodic())
    {
#if (AMREX_SPACEDIM == 2)
        Vector<IntVect> pshifts(9);
#else
        Vector<IntVect> pshifts(27);
#endif
        const Box& domain = geom.Domain();

        for (BoxList::const_iterator it = gcells.begin(); it != gcells.end(); ++it)
        {
            if (!domain.contains(*it))
            {
                geom.periodicShift(domain, *it, pshifts);
                for (int i = 0; i < pshifts.size(); ++i)
                {
                    const Box shftbox = *it + pshifts[i];
                    const Box ovlp = domain & shftbox;
                    BoxList bl = amrex::complementIn(ovlp, BoxList(ba));
                    bcells.catenate(bl);
                }
            }
        }
        gcells.catenate(bcells);
    }

    return BoxArray(gcells);
}


static void CollateBlockData(Vector<Real>& nodeRaw,
                             Vector<int>& connData,
                             int nCompPerNode)
{
#ifdef AMREX_USE_MPI
    const int IOProc = ParallelDescriptor::IOProcessorNumber();
    const int nProcs = ParallelDescriptor::NProcs();

    Vector<Real> localNodeRaw = nodeRaw;
    Vector<int>  localConnData = connData;

    Vector<int> nodeCounts(nProcs, 0), nodeOffsets(nProcs, 0);
    int localNodeCount = static_cast<int>(localNodeRaw.size());
    MPI_Gather(&localNodeCount, 1,
               ParallelDescriptor::Mpi_typemap<int>::type(),
               nodeCounts.data(), 1,
               ParallelDescriptor::Mpi_typemap<int>::type(),
               IOProc, ParallelDescriptor::Communicator());

    if (ParallelDescriptor::IOProcessor()) {
        for (int i = 1; i < nProcs; ++i) {
            nodeOffsets[i] = nodeOffsets[i-1] + nodeCounts[i-1];
        }
        nodeRaw.resize(nodeOffsets[nProcs-1] + nodeCounts[nProcs-1]);
    }

    MPI_Gatherv(localNodeRaw.data(), localNodeCount,
                ParallelDescriptor::Mpi_typemap<Real>::type(),
                nodeRaw.data(), nodeCounts.data(), nodeOffsets.data(),
                ParallelDescriptor::Mpi_typemap<Real>::type(),
                IOProc, ParallelDescriptor::Communicator());

    Vector<int> connCounts(nProcs, 0), connOffsets(nProcs, 0);
    int localConnCount = static_cast<int>(localConnData.size());
    MPI_Gather(&localConnCount, 1,
               ParallelDescriptor::Mpi_typemap<int>::type(),
               connCounts.data(), 1,
               ParallelDescriptor::Mpi_typemap<int>::type(),
               IOProc, ParallelDescriptor::Communicator());

    if (ParallelDescriptor::IOProcessor()) {
        for (int i = 1; i < nProcs; ++i) {
            connOffsets[i] = connOffsets[i-1] + connCounts[i-1];
        }
        connData.resize(connOffsets[nProcs-1] + connCounts[nProcs-1]);
    }

    MPI_Gatherv(localConnData.data(), localConnCount,
                ParallelDescriptor::Mpi_typemap<int>::type(),
                connData.data(), connCounts.data(), connOffsets.data(),
                ParallelDescriptor::Mpi_typemap<int>::type(),
                IOProc, ParallelDescriptor::Communicator());

    if (ParallelDescriptor::IOProcessor()) {
        for (int ip = 1; ip < nProcs; ++ip) {
            const int nodeOffset = nodeOffsets[ip] / nCompPerNode;
            for (int j = 0; j < connCounts[ip]; ++j) {
                connData[connOffsets[ip] + j] += nodeOffset;
            }
        }
    }
#else
    amrex::ignore_unused(nCompPerNode);
#endif
}

void WriteFEFromMultiFab(
    const std::string& outfile,
    const Vector<const MultiFab*>& state,
    const Vector<Geometry>& geom,
    const Vector<int>& ref_ratio,
    const Vector<std::string>& var_names,
    Real time,
    const TecplotFEOptions& opt)
{

    BL_ASSERT(!state.empty());
    BL_ASSERT(state.size() == geom.size());
    BL_ASSERT(var_names.size() == state[0]->nComp());
    
    if (!amrex::ParallelDescriptor::IOProcessor()) {
    return;
}
    const int finestLevel = (opt.finest_level >= 0)
        ? std::min(opt.finest_level, static_cast<int>(state.size()) - 1)
        : static_cast<int>(state.size()) - 1;
    const int Nlev = finestLevel + 1;

    Vector<int> comps;
    if (opt.comps.empty()) {
        comps.resize(state[0]->nComp());
        for (int n = 0; n < state[0]->nComp(); ++n) comps[n] = n;
    } else {
        comps = opt.comps;
    }

    Vector<BoxArray> gridArray(Nlev);
    Vector<Box> subboxArray(Nlev);
    for (int lev = 0; lev < Nlev; ++lev) {
        gridArray[lev] = state[lev]->boxArray();
        subboxArray[lev] = geom[lev].Domain();
    }

    const int nGrow = 1;
    using NodeFab = BaseFab<Node>;
    using MultiNodeFab = FabArray<NodeFab>;
    Vector<std::unique_ptr<MultiNodeFab> > nodes(Nlev);
    for (int lev = 0; lev < Nlev; ++lev) {
        nodes[lev] = std::make_unique<MultiNodeFab>(gridArray[lev], state[lev]->DistributionMap(), 1, nGrow);
    }

    using NodeMap = std::map<Node,int>;
    NodeMap nodeMap;
    int cnt = 0;
    
    for (int lev = 0; lev < Nlev; ++lev)
    {
        for (MFIter mfi(*nodes[lev]); mfi.isValid(); ++mfi)
        {
            NodeFab& ifab = (*nodes[lev])[mfi];
            const Box box = ifab.box() & subboxArray[lev];
            for (IntVect iv = box.smallEnd(); iv <= box.bigEnd(); box.next(iv)) {
                ifab(iv,0) = Node(iv, lev, mfi.index(), Node::VALID);
            }
        }

        if (lev != 0)
        {
            const int ref = ref_ratio[lev-1];
            const Box rangeBox(IntVect::TheZeroVector(), (ref-1) * IntVect::TheUnitVector());
            BoxArray bndryCells = GetBndryCells((*nodes[lev]).boxArray(), ref, geom[lev]);

            for (MFIter mfi(*nodes[lev]); mfi.isValid(); ++mfi)
            {
                const Box box = amrex::grow(mfi.validbox(), ref) & subboxArray[lev];
                NodeFab& ifab = (*nodes[lev])[mfi];
                std::vector<std::pair<int,Box> > isects = bndryCells.intersections(box);
                for (int i = 0; i < isects.size(); ++i)
                {
                    const Box& dstBox = isects[i].second;
                    const Box srcBox = amrex::coarsen(dstBox, ref);
                    NodeFab dst(dstBox,1);
                    for (IntVect iv(srcBox.smallEnd()); iv <= srcBox.bigEnd(); srcBox.next(iv))
                    {
                        const IntVect baseIV = ref * iv;
                        for (IntVect ivt(rangeBox.smallEnd()); ivt <= rangeBox.bigEnd(); rangeBox.next(ivt)) {
                            dst(baseIV + ivt,0) = Node(iv, lev-1, -1, Node::VALID);
                        }
                    }
                    const Box ovlp = dstBox & ifab.box();
                    if (ovlp.ok()) ifab.copy(dst, ovlp, 0, ovlp, 0, 1);
                }
            }
        }

        if (lev < finestLevel)
        {
            const BoxArray coarsenedFineBoxes = BoxArray(gridArray[lev+1]).coarsen(ref_ratio[lev]);
            for (MFIter mfi(*nodes[lev]); mfi.isValid(); ++mfi)
            {
                NodeFab& ifab = (*nodes[lev])[mfi];
                const Box& box = ifab.box();
                std::vector<std::pair<int,Box> > isects = coarsenedFineBoxes.intersections(box);
                for (int i = 0; i < isects.size(); ++i)
                {
                    const Box& ovlp = isects[i].second;
                    for (IntVect iv = ovlp.smallEnd(); iv <= ovlp.bigEnd(); ovlp.next(iv)) {
                        ifab(iv,0) = Node(iv, lev, mfi.index(), Node::COVERED);
                    }
                }
            }
        }

        for (MFIter mfi(*nodes[lev]); mfi.isValid(); ++mfi)
        {
            NodeFab& ifab = (*nodes[lev])[mfi];
            const Box box = mfi.validbox() & subboxArray[lev];
            for (IntVect iv(box.smallEnd()); iv <= box.bigEnd(); box.next(iv)) {
                if (ifab(iv,0).type == Node::VALID) nodeMap[ifab(iv,0)] = cnt++;
            }
        }
    }

    using EltSet = std::set<Element>;
    EltSet elements;
    for (int lev = 0; lev < Nlev; ++lev)
    {
        for (MFIter mfi(*nodes[lev]); mfi.isValid(); ++mfi)
        {
        
NodeFab& ifab = (*nodes[lev])[mfi];
Box box = ifab.box() & subboxArray[lev];
for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) box.growHi(dir,-1);

            for (IntVect iv(box.smallEnd()); iv <= box.bigEnd(); box.next(iv))
            {
#if (AMREX_SPACEDIM == 2)
                const Node& n1 = ifab(iv,0);
                const Node& n2 = ifab(IntVect(iv).shift(BASISV(0)),0);
                const Node& n3 = ifab(IntVect(iv).shift(IntVect::TheUnitVector()),0);
                const Node& n4 = ifab(IntVect(iv).shift(BASISV(1)),0);
                if (n1.type==Node::VALID && n2.type==Node::VALID && n3.type==Node::VALID && n4.type==Node::VALID) {
                    elements.insert(Element(n1,n2,n3,n4));
                }
#else
                const IntVect ivu = IntVect(iv).shift(BASISV(2));
                const Node& n1 = ifab(iv,0);
                const Node& n2 = ifab(IntVect(iv).shift(BASISV(0)),0);
                const Node& n3 = ifab(IntVect(iv).shift(BASISV(0)).shift(BASISV(1)),0);
                const Node& n4 = ifab(IntVect(iv).shift(BASISV(1)),0);
                const Node& n5 = ifab(ivu,0);
                const Node& n6 = ifab(IntVect(ivu).shift(BASISV(0)),0);
                const Node& n7 = ifab(IntVect(ivu).shift(BASISV(0)).shift(BASISV(1)),0);
                const Node& n8 = ifab(IntVect(ivu).shift(BASISV(1)),0);
                if (n1.type==Node::VALID && n2.type==Node::VALID && n3.type==Node::VALID && n4.type==Node::VALID &&
                    n5.type==Node::VALID && n6.type==Node::VALID && n7.type==Node::VALID && n8.type==Node::VALID) {
                    elements.insert(Element(n1,n2,n3,n4,n5,n6,n7,n8));
                }
#endif
            }
        }
    }

    const int nElts = opt.connect_cc ? static_cast<int>(elements.size()) : static_cast<int>(nodeMap.size());
    Vector<int> connData(Element::NNode * nElts);

    if (opt.connect_cc)
    {
        cnt = 0;
        for (EltSet::const_iterator it = elements.begin(); it != elements.end(); ++it)
        {
            for (int j = 0; j < Element::NNode; ++j)
            {
                NodeMap::const_iterator noit = nodeMap.find(*((*it).n[j]));
                BL_ASSERT(noit != nodeMap.end());
                connData[cnt++] = noit->second + 1;
            }
        }
    }
    else
    {
        cnt = 1;
        for (int i = 0; i < nElts; ++i) {
            for (int j = 0; j < Element::NNode; ++j) connData[i*Element::NNode+j] = cnt++;
        }
    }

    std::vector<Node> nodeVect(nodeMap.size());
    for (NodeMap::const_iterator it = nodeMap.begin(); it != nodeMap.end(); ++it) {
        nodeVect[it->second] = it->first;
    }

    const int nNodesFINAL = opt.connect_cc ? static_cast<int>(nodeVect.size()) : nElts * Element::NNode;
    const int nState = AMREX_SPACEDIM + static_cast<int>(comps.size());
    Vector<Vector<Real> > out(nState, Vector<Real>(nNodesFINAL, 0.0_rt));

    const auto plo = geom[0].ProbLoArray();
    cnt = 0;
    int levPrev = -1;
    int jGridPrev = -1;
    
    for (int i = 0; i < nodeVect.size(); ++i)
    {
        const Node& node = nodeVect[i];
        const IntVect& iv = node.iv;
        const auto dx = geom[node.level].CellSizeArray();
        const BoxArray& grids = state[node.level]->boxArray();

        int jGrid = node.grid;
        if (jGrid < 0)
        {
            bool found_it = false;
            if (node.level == levPrev && jGridPrev >= 0 && grids[jGridPrev].contains(iv)) {
                jGrid = jGridPrev;
                found_it = true;
            }
            for (int j = 0; j < grids.size() && !found_it; ++j) {
                if (grids[j].contains(iv)) {
                    jGrid = j;
                    found_it = true;
                }
            }
            BL_ASSERT(found_it);
        }
        levPrev = node.level;
        jGridPrev = jGrid;

        Vector<IntVect> ivt;
        if (opt.connect_cc) {
            ivt.resize(1, iv);
        } else {
            ivt.resize(AMREX_D_PICK(1,4,8), iv);
            ivt[1] += BASISV(0);
            ivt[2] = ivt[1] + BASISV(1);
            ivt[3] += BASISV(1);
#if (AMREX_SPACEDIM == 3)
            for (int n = 0; n < 4; ++n) ivt[4+n] = ivt[n] + BASISV(2);
#endif
        }

        for (int k = 0; k < ivt.size(); ++k)
        {
            const Real off = opt.connect_cc ? 0.5_rt : 0.0_rt;
            for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
                out[dir][cnt] = plo[dir] + (static_cast<Real>(ivt[k][dir]) + off) * dx[dir];
            }
            const FArrayBox& fab = (*state[node.level])[jGrid];
            for (int n = 0; n < comps.size(); ++n) {
                out[AMREX_SPACEDIM+n][cnt] = fab(iv, comps[n]);
            }
            ++cnt;
        }
    }
    
    const int nNodesWrite = nNodesFINAL;
    const int nEltsWrite  = nElts;

    std::cout << "IOProc before write:"
              << " nNodesWrite = " << nNodesWrite
              << ", nEltsWrite = " << nEltsWrite
              << ", connData.size() = " << connData.size()
              << std::endl;

std::string vars = AMREX_D_TERM("X", " Y", " Z");
for (int j = 0; j < comps.size(); ++j) {
    vars += " " + var_names[comps[j]];
}

if (opt.write_binary)
{
#ifdef USE_TEC_BIN_IO
    INTEGER4 Debug = 0;
    INTEGER4 VIsDouble = 1;
    INTEGER4 EltID = AMREX_D_PICK(0,1,3);

    const std::string block_or_point = opt.binary_point ? "FEPOINT" : "FEBLOCK";

    amrex::Vector<Real> tecdata(static_cast<std::size_t>(nState) *
                                static_cast<std::size_t>(nNodesWrite));

    if (opt.binary_point)
    {
        for (int i = 0; i < nNodesWrite; ++i) {
            for (int n = 0; n < nState; ++n) {
                tecdata[static_cast<std::size_t>(i) * nState + n] = out[n][i];
            }
        }
    }
    else
    {
        for (int n = 0; n < nState; ++n) {
            for (int i = 0; i < nNodesWrite; ++i) {
                tecdata[static_cast<std::size_t>(n) * nNodesWrite + i] = out[n][i];
            }
        }
    }

    INTEGER4 nPts  = nNodesWrite;
    INTEGER4 nElem = nEltsWrite;
    INTEGER4 nData = static_cast<INTEGER4>(tecdata.size());

    TECINI((char*)"AMReX FE data",
           (char*)vars.c_str(),
           (char*)outfile.c_str(),
           (char*)".",
           &Debug,
           &VIsDouble);

    TECZNE((char*)opt.zone_name.c_str(),
           &nPts,
           &nElem,
           &EltID,
           (char*)block_or_point.c_str(),
           NULL);

    TECDAT(&nData, tecdata.dataPtr(), &VIsDouble);
    TECNOD(connData.dataPtr());
    TECEND();
#else
    amrex::Abort("Need to compile with USE_TEC_BIN_IO defined");
#endif
}
else
{
    std::ofstream os(outfile.c_str());
    os << AMREX_D_TERM("VARIABLES= \"X\"", " \"Y\"", " \"Z\"");
    for (int j = 0; j < comps.size(); ++j) os << " \"" << var_names[comps[j]] << "\"";
    os << "\nZONE T=\"" << opt.zone_name << " time = " << time
       << "\", N=" << nNodesWrite
       << ", E=" << nEltsWrite
       << ", F=FEPOINT, ET=" << AMREX_D_PICK("POINT","QUADRILATERAL","BRICK") << "\n";

    for (int i = 0; i < nNodesWrite; ++i) {
        for (int n = 0; n < nState; ++n) os << out[n][i] << " ";
        os << "\n";
    }
    for (int e = 0; e < nEltsWrite; ++e) {
        for (int j = 0; j < Element::NNode; ++j) os << connData[e*Element::NNode+j] << " ";
        os << "\n";
    }
    os.close();
}
}

} // namespace amr_tecplot_fe

/* ========================= Example call in AmrCoreAdv =========================

#include "amrex_tecplot_fe_refactor.cpp"   // better split to .H/.cpp in your project

void AmrCoreAdv::WriteTecplotFEFile(const std::string& outfile)
{
    Vector<const MultiFab*> state(finest_level + 1);
    Vector<Geometry> geoms(finest_level + 1);
    Vector<int> refs(finest_level);

    for (int lev = 0; lev <= finest_level; ++lev) {
        state[lev] = &phi_new[lev];        // <-- replace with your MultiFab
        geoms[lev] = geom[lev];
        if (lev < finest_level) refs[lev] = refRatio(lev)[0];
    }

    Vector<std::string> names = {
        "rho", "u", "v", "w", "p", "T"
    }; // <-- replace with your variable names, size must equal state[0]->nComp()

    amr_tecplot_fe::TecplotFEOptions opt;
    opt.zone_name = "amr_solution";
    opt.connect_cc = true;

    amr_tecplot_fe::WriteFEFromMultiFab(
        outfile, state, geoms, refs, names, t_new[0], opt);
}

============================================================================= */
