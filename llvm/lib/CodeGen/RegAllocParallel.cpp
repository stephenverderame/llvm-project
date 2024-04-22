//===-- RegAllocBasic.cpp - Basic Register Allocator ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RAParallel function pass, which provides a minimal
// implementation of the basic register allocator.
//
//===----------------------------------------------------------------------===//

#include "AllocationOrder.h"
#include "LiveDebugVariables.h"
#include "RegAllocBase.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <initializer_list>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

static RegisterRegAlloc ParallelRegAlloc("parallel",
                                         "parallel register allocator",
                                         createParallelRegisterAllocator);

namespace {
struct CompSpillWeight {
  bool operator()(const LiveInterval *A, const LiveInterval *B) const {
    return A->weight() < B->weight();
  }
};
} // namespace

namespace {
using IGraph = std::map<Register, std::set<Register>>;

/// A partition tree is a binary tree that represents a partition of a set of
/// registers. Each leaf node contains a set of registers, and each internal
/// node contains two children and no registers.
struct PartitionTree {
  /// Invariant: Either has both children and no regs or is a leaf and has regs.
  std::unique_ptr<PartitionTree> Left, Right;
  std::vector<Register> Regs;
};

/// RAParallel provides a minimal implementation of the basic register
/// allocation algorithm. It prioritizes live virtual registers by spill weight
/// and spills whenever a register is unavailable. This is not practical in
/// production but provides a useful baseline both for measuring other
/// allocators and comparing the speed of the basic algorithm against other
/// styles of allocators.
class RAParallel : public MachineFunctionPass,
                   public RegAllocBase,
                   private LiveRangeEdit::Delegate {
  // context
  MachineFunction *MF = nullptr;

  // state
  std::unique_ptr<Spiller> SpillerInstance;
  std::priority_queue<const LiveInterval *, std::vector<const LiveInterval *>,
                      CompSpillWeight>
      Queue;

  // Scratch space.  Allocated here to avoid repeated malloc calls in
  // selectOrSplit().
  BitVector UsableRegs;

  bool LRE_CanEraseVirtReg(Register) override;
  void LRE_WillShrinkVirtReg(Register) override;

  /// Computes an interference graph from the live intervals in the function.
  IGraph computeInterference();

  /// Allocates physical registers for a set of virtual registers.
  void allocPhysRegsPartial(const std::vector<Register> &VRegs);
  /// Recursively allocates physical registers for a partition tree.
  void allocPhysRegsTree(const PartitionTree &T);

public:
  RAParallel(const RegClassFilterFunc F = allocateAllRegClasses);

  /// Return the pass name.
  StringRef getPassName() const override {
    return "Parallel Register Allocator";
  }

  /// RAParallel analysis usage.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  void releaseMemory() override;

  Spiller &spiller() override { return *SpillerInstance; }

  void enqueueImpl(const LiveInterval *LI) override { Queue.push(LI); }

  const LiveInterval *dequeue() override {
    if (Queue.empty())
      return nullptr;
    const LiveInterval *LI = Queue.top();
    Queue.pop();
    return LI;
  }

  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &SplitVRegs) override;

  /// Perform register allocation.
  bool runOnMachineFunction(MachineFunction &Mf) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  // Helper for spilling all live virtual registers currently unified under preg
  // that interfere with the most recently queried lvr.  Return true if spilling
  // was successful, and append any new spilled/split intervals to splitLVRs.
  bool spillInterferences(const LiveInterval &VirtReg, MCRegister PhysReg,
                          SmallVectorImpl<Register> &SplitVRegs);

  static char ID;
};

/// Gets a perfect (simplicial) elimination order of the interference graph G,
/// in reverse so that index 0 is v_n.
std::vector<Register> perfectElimOrder(const IGraph &G) {
  // maximum cardinality search, generates a perfect elimination order bc
  // ssa interference graphs are chordal
  std::vector<Register> Order;
  std::vector<std::pair<Register, int>> Heap;
  std::map<Register, int> Weights;
  std::set<Register> Numbered;
  for (auto [V, _] : G) {
    Heap.push_back(std::make_pair(V, 0));
    Weights.insert(std::make_pair(V, 0));
  }
  const auto Comp = [](const std::pair<Register, int> &A,
                       const std::pair<Register, int> &B) {
    return A.second < B.second;
  };
  while (!Heap.empty()) {
    std::pop_heap(Heap.begin(), Heap.end(), Comp);
    auto [V, W] = Heap.back();
    Heap.pop_back();
    if (Numbered.find(V) != Numbered.end()) {
      continue;
    }
    Numbered.insert(V);
    Order.push_back(V);
    for (auto U : G.at(V)) {
      if (Numbered.find(U) != Numbered.end()) {
        continue;
      }
      Weights[U] += 1;
      Heap.push_back(std::make_pair(U, Weights[U]));
      std::push_heap(Heap.begin(), Heap.end(), Comp);
    }
  }
  return Order;
}

/// Returns true if Clique is a complete subgraph of G.
bool isCompleteSubgraph(const IGraph &G, const std::set<Register> &Clique) {
  for (auto V : Clique) {
    for (auto U : Clique) {
      if (V == U) {
        continue;
      }
      if (G.at(V).find(U) == G.at(V).end()) {
        return false;
      }
    }
  }
  return true;
}

/// Gets the connected component of G containing X, excluding the registers in
/// Exclude.
std::set<Register> connectedComponent(const IGraph &G,
                                      const std::set<Register> &Exclude,
                                      Register X) {
  std::set<Register> Component;
  std::queue<Register> Q;
  Q.push(X);
  while (!Q.empty()) {
    auto V = Q.front();
    Q.pop();
    if (Component.find(V) != Component.end()) {
      continue;
    }
    Component.insert(V);
    for (auto U : G.at(V)) {
      if (Component.find(U) != Component.end() ||
          Exclude.find(U) != Exclude.end()) {
        continue;
      }
      Q.push(U);
    }
  }
  return Component;
}

/// Returns the induced subgraph of Set in G.
IGraph inducedSubgraph(const IGraph &G, std::set<Register> &Set) {
  IGraph Subgraph;
  for (auto V : Set) {
    Subgraph.insert(std::make_pair(V, std::set<Register>()));
    for (auto U : G.at(V)) {
      if (Set.find(U) != Set.end()) {
        Subgraph[V].insert(U);
      }
    }
  }
  return Subgraph;
}

/// Removes the nodes of S from G.
IGraph operator-(const IGraph &G, const std::set<Register> &S) {
  IGraph H = G;
  for (auto N : S) {
    H.erase(N);
    for (auto &[_, Neighbors] : H) {
      Neighbors.erase(N);
    }
  }
  return H;
}

/// Builds a partition tree from a set of atoms. The tree will be
/// a right-associative binary tree.
PartitionTree partitionTreeFromAtoms(const std::vector<IGraph> &Atoms) {
  PartitionTree T;
  PartitionTree *Last = &T;
  for (unsigned Idx = 0; Idx < Atoms.size(); ++Idx) {
    if (Idx < Atoms.size() - 1) {
      Last->Left = std::make_unique<PartitionTree>();
      for (auto &[N, _] : Atoms[Idx]) {
        Last->Left->Regs.push_back(N);
      }
      Last->Right = std::make_unique<PartitionTree>();
      Last = Last->Right.get();
    } else {
      for (auto &[N, _] : Atoms[Idx]) {
        Last->Regs.push_back(N);
      }
    }
  }
  return T;
}

/// Builds a partition tree from the interference graph G.
/// G is assumed to be chordal, however if it is not, the algorithm will still
/// be correct (probably?), just not optimal.
PartitionTree buildPartitionTree(const IGraph &G) {
  auto GPrime = G;
  auto PEO = perfectElimOrder(G);
  std::map<Register, size_t> Indices;
  for (size_t Idx = 0; Idx < PEO.size(); ++Idx) {
    Indices.insert(std::make_pair(PEO[Idx], Idx));
  }
  std::vector<IGraph> Atoms;
  for (unsigned Idx = 0; Idx < PEO.size(); ++Idx) {
    auto &V = PEO[Idx];
    std::set<Register> Clique;
    for (auto U : G.at(V)) {
      if (Indices[U] > Idx) {
        Clique.insert(U);
      }
    }
    if (isCompleteSubgraph(G, Clique)) {
      const auto Component = connectedComponent(GPrime, Clique, V);
      auto VertexSet = Component;
      for (auto &N : Clique) {
        VertexSet.insert(N);
      }
      Atoms.emplace_back(inducedSubgraph(GPrime, VertexSet));
      GPrime = GPrime - Component;
    }
  }
  return partitionTreeFromAtoms(Atoms);
}

char RAParallel::ID = 0;

} // end anonymous namespace

char &llvm::RAParallelID = RAParallel::ID;

INITIALIZE_PASS_BEGIN(RAParallel, "regallocparallel",
                      "Parallel Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveDebugVariables)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(RegisterCoalescer)
INITIALIZE_PASS_DEPENDENCY(MachineScheduler)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrix)
INITIALIZE_PASS_END(RAParallel, "regallocparallel",
                    "Parallel Register Allocator", false, false)

bool RAParallel::LRE_CanEraseVirtReg(Register VirtReg) {
  LiveInterval &LI = LIS->getInterval(VirtReg);
  if (VRM->hasPhys(VirtReg)) {
    Matrix->unassign(LI);
    aboutToRemoveInterval(LI);
    return true;
  }
  // Unassigned virtreg is probably in the priority queue.
  // RegAllocBase will erase it after dequeueing.
  // Nonetheless, clear the live-range so that the debug
  // dump will show the right state for that VirtReg.
  LI.clear();
  return false;
}

void RAParallel::LRE_WillShrinkVirtReg(Register VirtReg) {
  if (!VRM->hasPhys(VirtReg))
    return;

  // Register is assigned, put it back on the queue for reassignment.
  LiveInterval &LI = LIS->getInterval(VirtReg);
  Matrix->unassign(LI);
  enqueue(&LI);
}

RAParallel::RAParallel(RegClassFilterFunc F)
    : MachineFunctionPass(ID), RegAllocBase(F) {}

void RAParallel::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addRequired<LiveIntervals>();
  AU.addPreserved<LiveIntervals>();
  AU.addPreserved<SlotIndexes>();
  AU.addRequired<LiveDebugVariables>();
  AU.addPreserved<LiveDebugVariables>();
  AU.addRequired<LiveStacks>();
  AU.addPreserved<LiveStacks>();
  AU.addRequired<MachineBlockFrequencyInfo>();
  AU.addPreserved<MachineBlockFrequencyInfo>();
  AU.addRequiredID(MachineDominatorsID);
  AU.addPreservedID(MachineDominatorsID);
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addRequired<VirtRegMap>();
  AU.addPreserved<VirtRegMap>();
  AU.addRequired<LiveRegMatrix>();
  AU.addPreserved<LiveRegMatrix>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

void RAParallel::releaseMemory() { SpillerInstance.reset(); }

// Spill or split all live virtual registers currently unified under PhysReg
// that interfere with VirtReg. The newly spilled or split live intervals are
// returned by appending them to SplitVRegs.
bool RAParallel::spillInterferences(const LiveInterval &VirtReg,
                                    MCRegister PhysReg,
                                    SmallVectorImpl<Register> &SplitVRegs) {
  // Record each interference and determine if all are spillable before mutating
  // either the union or live intervals.
  SmallVector<const LiveInterval *, 8> Intfs;

  // Collect interferences assigned to any alias of the physical register.
  for (MCRegUnit Unit : TRI->regunits(PhysReg)) {
    LiveIntervalUnion::Query &Q = Matrix->query(VirtReg, Unit);
    for (const auto *Intf : reverse(Q.interferingVRegs())) {
      if (!Intf->isSpillable() || Intf->weight() > VirtReg.weight())
        return false;
      Intfs.push_back(Intf);
    }
  }
  LLVM_DEBUG(dbgs() << "spilling " << printReg(PhysReg, TRI)
                    << " interferences with " << VirtReg << "\n");
  assert(!Intfs.empty() && "expected interference");

  // Spill each interfering vreg allocated to PhysReg or an alias.
  // NOLINTNEXTLINE(readability-identifier-naming)
  for (unsigned i = 0, e = Intfs.size(); i != e; ++i) {
    const LiveInterval &Spill = *Intfs[i];

    // Skip duplicates.
    if (!VRM->hasPhys(Spill.reg()))
      continue;

    // Deallocate the interfering vreg by removing it from the union.
    // A LiveInterval instance may not be in a union during modification!
    Matrix->unassign(Spill);

    // Spill the extracted interval.
    LiveRangeEdit LRE(&Spill, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
    spiller().spill(LRE);
  }
  return true;
}

// Driver for the register assignment and splitting heuristics.
// Manages iteration over the LiveIntervalUnions.
//
// This is a minimal implementation of register assignment and splitting that
// spills whenever we run out of registers.
//
// selectOrSplit can only be called once per live virtual register. We then do a
// single interference test for each register the correct class until we find an
// available register. So, the number of interference tests in the worst case is
// |vregs| * |machineregs|. And since the number of interference tests is
// minimal, there is no value in caching them outside the scope of
// selectOrSplit().
MCRegister RAParallel::selectOrSplit(const LiveInterval &VirtReg,
                                     SmallVectorImpl<Register> &SplitVRegs) {
  // Populate a list of physical register spill candidates.
  SmallVector<MCRegister, 8> PhysRegSpillCands;

  // Check for an available register in this class.
  auto Order =
      AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);
  for (MCRegister PhysReg : Order) {
    assert(PhysReg.isValid());
    // Check for interference in PhysReg
    switch (Matrix->checkInterference(VirtReg, PhysReg)) {
    case LiveRegMatrix::IK_Free:
      // PhysReg is available, allocate it.
      return PhysReg;

    case LiveRegMatrix::IK_VirtReg:
      // Only virtual registers in the way, we may be able to spill them.
      PhysRegSpillCands.push_back(PhysReg);
      continue;

    default:
      // RegMask or RegUnit interference.
      continue;
    }
  }

  // Try to spill another interfering reg with less spill weight.
  for (MCRegister &PhysReg : PhysRegSpillCands) {
    if (!spillInterferences(VirtReg, PhysReg, SplitVRegs))
      continue;

    assert(!Matrix->checkInterference(VirtReg, PhysReg) &&
           "Interference after spill.");
    // Tell the caller to allocate to this newly freed physical register.
    return PhysReg;
  }

  // No other spill candidates were found, so spill the current VirtReg.
  LLVM_DEBUG(dbgs() << "spilling: " << VirtReg << '\n');
  if (!VirtReg.isSpillable())
    return ~0u;
  LiveRangeEdit LRE(&VirtReg, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  spiller().spill(LRE);

  // The live virtual register requesting allocation was spilled, so tell
  // the caller not to allocate anything during this round.
  return 0;
}

IGraph RAParallel::computeInterference() {
  IGraph G;
  auto &MRI = MF->getRegInfo();
  // const auto *TRI = MF->getSubtarget().getRegisterInfo();
  for (unsigned Idx = 0; Idx < MRI.getNumVirtRegs(); ++Idx) {
    Register Reg = Register::index2VirtReg(Idx);
    if (MRI.reg_nodbg_empty(Reg)) {
      continue;
    }
    for (unsigned Idx2 = 0; Idx2 < MRI.getNumVirtRegs(); ++Idx2) {
      Register Reg2 = Register::index2VirtReg(Idx2);
      if (MRI.reg_nodbg_empty(Reg2)) {
        continue;
      }
      if (Reg == Reg2) {
        continue;
      }
      if (LIS->getInterval(Reg).overlaps(LIS->getInterval(Reg2)) &&
          MRI.getRegClassOrNull(Reg) == MRI.getRegClassOrNull(Reg2)) {
        if (G.find(Reg) == G.end()) {
          G.insert(std::make_pair(Reg, std::set<Register>()));
        }
        if (G.find(Reg2) == G.end()) {
          G.insert(std::make_pair(Reg2, std::set<Register>()));
        }
        G[Reg].insert(Reg2);
        G[Reg2].insert(Reg);
      }
    }
  }
  return G;
}

bool RAParallel::runOnMachineFunction(MachineFunction &Mf) {
  LLVM_DEBUG(dbgs() << "********** Parallel REGISTER ALLOCATION **********\n"
                    << "********** Function: " << Mf.getName() << '\n');

  MF = &Mf;
  RegAllocBase::init(getAnalysis<VirtRegMap>(), getAnalysis<LiveIntervals>(),
                     getAnalysis<LiveRegMatrix>());
  VirtRegAuxInfo VRAI(*MF, *LIS, *VRM, getAnalysis<MachineLoopInfo>(),
                      getAnalysis<MachineBlockFrequencyInfo>());
  VRAI.calculateSpillWeightsAndHints();

  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM, VRAI));

  const auto T = buildPartitionTree(computeInterference());
  allocPhysRegsTree(T);
  postOptimization();

  // Diagnostic output before rewriting
  LLVM_DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *VRM << "\n");

  releaseMemory();
  return true;
}

FunctionPass *llvm::createParallelRegisterAllocator() {
  return new RAParallel();
}

FunctionPass *llvm::createParallelRegisterAllocator(RegClassFilterFunc F) {
  return new RAParallel(F);
}

void RAParallel::allocPhysRegsTree(const PartitionTree &T) {
  if (T.Left && T.Right) {
    // TODO: spawn task for left subtree
    allocPhysRegsTree(*T.Left);
    allocPhysRegsTree(*T.Right);
    // TODO: merge results
  } else {
    assert(!T.Left && !T.Right && "expected leaf node");
    allocPhysRegsPartial(T.Regs);
  }
}

void RAParallel::allocPhysRegsPartial(const std::vector<Register> &VRegs) {
  for (auto &V : VRegs) {
    if (MRI->reg_nodbg_empty(V)) {
      continue;
    }
    enqueue(&LIS->getInterval(V));
  }

  // Continue assigning vregs one at a time to available physical registers.
  while (const LiveInterval *VirtReg = dequeue()) {
    if (VRM->hasPhys(VirtReg->reg())) {
      // already allocated
      continue;
    }

    // Unused registers can appear when the spiller coalesces snippets.
    if (MRI->reg_nodbg_empty(VirtReg->reg())) {
      LLVM_DEBUG(dbgs() << "Dropping unused " << *VirtReg << '\n');
      aboutToRemoveInterval(*VirtReg);
      LIS->removeInterval(VirtReg->reg());
      continue;
    }

    // Invalidate all interference queries, live ranges could have changed.
    Matrix->invalidateVirtRegs();

    // selectOrSplit requests the allocator to return an available physical
    // register if possible and populate a list of new live intervals that
    // result from splitting.
    LLVM_DEBUG(dbgs() << "\nselectOrSplit "
                      << TRI->getRegClassName(MRI->getRegClass(VirtReg->reg()))
                      << ':' << *VirtReg << " w=" << VirtReg->weight() << '\n');

    using VirtRegVec = SmallVector<Register, 4>;

    VirtRegVec SplitVRegs;
    MCRegister AvailablePhysReg = selectOrSplit(*VirtReg, SplitVRegs);

    if (AvailablePhysReg == ~0u) {
      // selectOrSplit failed to find a register!
      // Probably caused by an inline asm.
      MachineInstr *MI = nullptr;
      for (MachineRegisterInfo::reg_instr_iterator
               I = MRI->reg_instr_begin(VirtReg->reg()),
               E = MRI->reg_instr_end();
           I != E;) {
        MI = &*(I++);
        if (MI->isInlineAsm())
          break;
      }

      const TargetRegisterClass *RC = MRI->getRegClass(VirtReg->reg());
      ArrayRef<MCPhysReg> AllocOrder = RegClassInfo.getOrder(RC);
      if (AllocOrder.empty())
        report_fatal_error("no registers from class available to allocate");
      else if (MI && MI->isInlineAsm()) {
        MI->emitError("inline assembly requires more registers than available");
      } else if (MI) {
        LLVMContext &Context =
            MI->getParent()->getParent()->getMMI().getModule()->getContext();
        Context.emitError("ran out of registers during register allocation");
      } else {
        report_fatal_error("ran out of registers during register allocation");
      }

      // Keep going after reporting the error.
      VRM->assignVirt2Phys(VirtReg->reg(), AllocOrder.front());
    } else if (AvailablePhysReg)
      Matrix->assign(*VirtReg, AvailablePhysReg);

    for (Register Reg : SplitVRegs) {
      assert(LIS->hasInterval(Reg));

      LiveInterval *SplitVirtReg = &LIS->getInterval(Reg);
      assert(!VRM->hasPhys(SplitVirtReg->reg()) && "Register already assigned");
      if (MRI->reg_nodbg_empty(SplitVirtReg->reg())) {
        assert(SplitVirtReg->empty() && "Non-empty but used interval");
        LLVM_DEBUG(dbgs() << "not queueing unused  " << *SplitVirtReg << '\n');
        aboutToRemoveInterval(*SplitVirtReg);
        LIS->removeInterval(SplitVirtReg->reg());
        continue;
      }
      LLVM_DEBUG(dbgs() << "queuing new interval: " << *SplitVirtReg << "\n");
      assert(SplitVirtReg->reg().isVirtual() &&
             "expect split value in virtual register");
      enqueue(SplitVirtReg);
    }
  }
}
