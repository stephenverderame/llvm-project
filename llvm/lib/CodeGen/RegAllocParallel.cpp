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
#include "IGraph.hpp"
#include "LiveDebugVariables.h"
#include "RegAllocBase.h"
#include "RegisterCoalescer.h"
#include "llvm/ADT/DirectedGraph.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervalUnion.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <fstream>
#include <initializer_list>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <variant>

using namespace llvm;

#define DEBUG_TYPE "regalloc"

static RegisterRegAlloc ParallelRegAlloc("parallel",
                                         "parallel register allocator",
                                         createParallelRegisterAllocator);

static cl::opt<std::string> OutputInterferenceGraph{
    "debug-graph", cl::desc("Specify file to write interference graph to"),
    cl::value_desc("filename")};
static cl::opt<std::string>
    OutputPartitionTree("debug-partitions",
                        cl::desc("Specify file to write partition tree to"),
                        cl::value_desc("filename"));

namespace {
struct CompSpillWeight {
  bool operator()(const LiveInterval *A, const LiveInterval *B) const {
    return A->weight() < B->weight();
  }
};
} // namespace

namespace {

struct ColoringResult {
  // Color Union-Find
  class Color {
    struct ColorClass {
      MCRegister PReg;
      std::set<Register> Members;
    };
    // Physical register or parent
    std::variant<ColorClass, std::shared_ptr<Color>> Value;

  private:
    Color *getRoot() {
      return std::visit(
          [this](auto &&V) {
            using T = std::decay_t<decltype(V)>;
            if constexpr (std::is_same_v<ColorClass, T>) {
              return this;
            } else {
              return V->getRoot();
            }
          },
          Value);
    }

    const Color *getRootC() const {
      return std::visit(
          [this](auto &&V) {
            using T = std::decay_t<decltype(V)>;
            if constexpr (std::is_same_v<ColorClass, T>) {
              return this;
            } else {
              return V->getRootC();
            }
          },
          Value);
    }

    /// Gets the root of the current node in the union-find data structure.
    /// Requires that the current node is not a physical register (not a root)
    std::shared_ptr<Color> getRootHelper() const {
      auto Parent = std::get<std::shared_ptr<Color>>(Value);
      return std::visit(
          [Parent](auto &&Val) {
            using T = std::decay_t<decltype(Val)>;
            if constexpr (std::is_same_v<ColorClass, T>) {
              return Parent;
            } else {
              return Parent->getRootHelper();
            }
          },
          Parent->Value);
    }

    /// Adds the members to the color class this node represents
    void addMembers(const std::set<Register> &Members) {
      auto *Root = getRoot();
      auto &Class = std::get<ColorClass>(Root->Value);
      Class.Members.insert(Members.begin(), Members.end());
    }

  public:
    /// Gets the physical register for this color
    MCRegister getPReg() {
      if (std::holds_alternative<ColorClass>(Value)) {
        return std::get<ColorClass>(Value).PReg;
      }
      Value = getRootHelper();
      auto Ptr = std::get<std::shared_ptr<Color>>(Value);
      assert(Ptr != nullptr);
      return Ptr->getPReg();
    }

    MCRegister getPReg() const {
      if (std::holds_alternative<ColorClass>(Value)) {
        return std::get<ColorClass>(Value).PReg;
      }
      return std::get<std::shared_ptr<Color>>(Value)->getPReg();
    }

    /// Sets the color of all nodes in the class to be the specified color
    /// @{
    void setColor(const std::shared_ptr<Color> &C) {
      auto *Root = getRoot();
      auto &Class = std::get<ColorClass>(Root->Value);
      C->addMembers(Class.Members);
      Root->Value = C;
      Value = C;
    }
    void setColor(std::shared_ptr<Color> &&C) { setColor(C); }
    /// @}

    Color(std::shared_ptr<Color> Value) : Value(std::move(Value)) {}
    Color(MCRegister PReq, Register VReg) {
      std::set<Register> Members;
      Members.insert(VReg);
      Value = ColorClass{PReq, Members};
    }
    Color(MCRegister PReq) : Value(ColorClass{PReq, std::set<Register>()}) {}

    const std::set<Register> &members() const {
      auto *Root = getRootC();
      return std::get<ColorClass>(Root->Value).Members;
    }
  };
  /// Set of physical registers used.
  std::vector<std::shared_ptr<Color>> Colors;
  /// Map virtual register to physical register by the index in `Colors`.
  /// A register that maps to `nullptr` is spilled.
  std::map<Register, std::shared_ptr<Color>> RegToColor;

  raw_ostream &print(raw_ostream &OS, const TargetRegisterInfo *TRI) const {
    for (auto &[Reg, Color] : RegToColor) {
      OS << printReg(Reg, TRI) << " -> ";
      if (Color == nullptr) {
        OS << "Spilled\n";
      } else {
        OS << printReg(Color->getPReg(), TRI) << "\t\t(";
        for (auto Unit : TRI->regunits(Color->getPReg())) {
          OS << printRegUnit(Unit, TRI) << ", ";
        }
        OS << ")\n";
      }
    }
    return OS;
  }
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
  [[nodiscard]] IGraph computeInterference();

  /// Allocates physical registers and handles spills for a coloring result.
  void allocPhysRegsFinal(const ColoringResult &Coloring);
  /// Recursively builds a coloring result from a partition tree.
  [[nodiscard]] ColoringResult colorPhysRegsTree(const PartitionTree &T) const;

  /// Performs register allocation on a leaf node of the partition tree
  [[nodiscard]] ColoringResult localColor(const IGraph &G) const;
  /// Returns true if `VReg` interferes with `PReg` given the current
  /// interference graph and coloring.
  [[nodiscard]] bool doesInterfere(const IGraph &G,
                                   const ColoringResult &CurColoring,
                                   Register VReg, MCRegister PReg) const;
  /// Merges two coloring results together of sibilings in the partition tree
  /// Requires that the only shared virtual registers between the two coloring
  /// results are in the clique.
  [[nodiscard]] ColoringResult
  mergeResults(ColoringResult &&A, ColoringResult &&B,
               const std::set<Register> &Clique) const;

  /// Gets an available color for `NodePReg` from `AvailableColors`. If
  /// `MustChange` is true, the color must be different from `CurPReg`. If the
  /// color hasn't been used before, adds it to the `Result.Colors` list.
  /// The returned color is *NOT* removed from `AvailableColors`.
  /// @param VReg The virtual register that needs a color.
  /// @param CurColorClass The current color class of the virtual register.
  /// @param UsedColors The set of physical registers that have been used in the
  /// coloring.
  /// @param Result The current coloring result.
  /// @param ExistingColors A map of physical registers to their color in the
  /// resulting coloring. Updated with new colors as they are added.
  /// @param MustChange If true, the color must be different from `CurPReg`.
  /// @return The color to use for `VPReg`.
  [[nodiscard]] std::shared_ptr<ColoringResult::Color>
  getAvailableColor(Register VReg, const ColoringResult::Color &CurColorClass,
                    const std::set<MCRegister> &UsedColors,
                    ColoringResult &Result,
                    std::map<MCRegister, std::shared_ptr<ColoringResult::Color>>
                        &ExistingColors,
                    bool MustChange = false) const;

  /// Recolors the right subgraph of the partition tree. Registers in the right
  /// subgraph that are in `NeedColors` will be colored with a new color.
  /// Registers in the right subgraph that are already colored will be recolored
  /// if they have not been recolored already (not in `ColorsToChange`).
  /// @param ExistingColors A map of physical registers to their color in the
  /// resulting coloring. Updated with new colors as they are added.
  /// @param UsedColors A set of physical registers that have been used in the
  /// coloring.
  /// @param NeedColors A set of registers in the right subgraph that need their
  /// color changed.
  /// @param ColorsToChange A set of colors in the right subgraph that need to
  /// be changed.
  /// @param Right The coloring result of the right subgraph.
  /// @param Result The current coloring result.
  /// @return The new coloring result with the right subgraph colored and added
  /// to the existing coloring result.
  ColoringResult
  recolorRight(std::map<MCRegister, std::shared_ptr<ColoringResult::Color>>
                   &ExistingColors,
               std::set<MCRegister> &UsedColors, std::set<Register> &NeedColors,
               std::unordered_set<ColoringResult::Color *> &ColorsToChange,
               ColoringResult &Right, ColoringResult &&Result) const;

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

char RAParallel::ID = 0;

/// Returns true if the order contains the register.
bool orderContainsReg(const AllocationOrder &Order, MCRegister Reg) {
  for (MCRegister R : Order)
    if (R == Reg)
      return true;
  return false;
}

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
  // SmallVector<MCRegister, 8> PhysRegSpillCands;

  // // Check for an available register in this class.
  // auto Order =
  //     AllocationOrder::create(VirtReg.reg(), *VRM, RegClassInfo, Matrix);
  // for (MCRegister PhysReg : Order) {
  //   assert(PhysReg.isValid());
  //   // Check for interference in PhysReg
  //   switch (Matrix->checkInterference(VirtReg, PhysReg)) {
  //   case LiveRegMatrix::IK_Free:
  //     // PhysReg is available, allocate it.
  //     return PhysReg;

  //   case LiveRegMatrix::IK_VirtReg:
  //     // Only virtual registers in the way, we may be able to spill them.
  //     PhysRegSpillCands.push_back(PhysReg);
  //     continue;

  //   default:
  //     // RegMask or RegUnit interference.
  //     continue;
  //   }
  // }

  // // Try to spill another interfering reg with less spill weight.
  // for (MCRegister &PhysReg : PhysRegSpillCands) {
  //   if (!spillInterferences(VirtReg, PhysReg, SplitVRegs))
  //     continue;

  //   assert(!Matrix->checkInterference(VirtReg, PhysReg) &&
  //          "Interference after spill.");
  //   // Tell the caller to allocate to this newly freed physical register.
  //   return PhysReg;
  // }

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
        if (G.find(Reg) == G.end()) {
          G.insert(std::make_pair(Reg, std::set<Register>()));
        }
        continue;
      }
      if (LIS->getInterval(Reg).overlaps(LIS->getInterval(Reg2))) {
        if (G.find(Reg) == G.end()) {
          G.insert(std::make_pair(Reg, std::set<Register>()));
        }
        if (G.find(Reg2) == G.end()) {
          G.insert(std::make_pair(Reg2, std::set<Register>()));
        }
        if (LIS->getInterval(Reg).hasSubRanges() &&
            LIS->getInterval(Reg2).hasSubRanges()) {
          for (auto &Range1 : LIS->getInterval(Reg).subranges()) {
            for (auto &Range2 : LIS->getInterval(Reg2).subranges()) {
              if (Range1.overlaps(Range2)) {
                G[Reg].insert(Reg2);
                G[Reg2].insert(Reg);
                goto dbl_break;
              }
            }
          }
        dbl_break:
          // NOLINTNEXTLINE
        } else {
          G[Reg].insert(Reg2);
          G[Reg2].insert(Reg);
        }
      }
    }
  }
  return G;
}

bool RAParallel::doesInterfere(const IGraph &G,
                               const ColoringResult &CurColoring, Register VReg,
                               MCRegister PReg) const {
  for (auto Neighbor : G.at(VReg)) {
    if (auto NeighorCol = CurColoring.RegToColor.find(Neighbor);
        NeighorCol != CurColoring.RegToColor.end() &&
        NeighorCol->second != nullptr) {
      const auto NeighborReg = NeighorCol->second->getPReg();
      if (TRI->regsOverlap(PReg, NeighborReg) ||
          Matrix->checkInterference(LIS->getInterval(VReg), PReg) !=
              LiveRegMatrix::IK_Free) {
        return true;
      }
    }
  }
  return false;
}

/// Gets a `std::shared_ptr<ColoringResult::Color>` for `AvailableColor`.
/// If the color hasn't been used before, adds it to the
/// `Result.Colors` list and updates `ExistingColors`. Otherwise, returns the
/// existing color.
template <typename... T>
auto chooseColor(MCRegister AvailableColor, ColoringResult &Result,
                 std::map<MCRegister, std::shared_ptr<ColoringResult::Color>>
                     &ExistingColors,
                 T &&...ColorArgs) {
  if (auto ExistingColor = ExistingColors.find(AvailableColor);
      ExistingColor != ExistingColors.end()) {
    return ExistingColor->second;
  }
  auto NewColor = std::make_shared<ColoringResult::Color>(
      AvailableColor, std::forward<T>(ColorArgs)...);
  Result.Colors.push_back(NewColor);
  ExistingColors.insert(std::make_pair(AvailableColor, NewColor));
  return NewColor;
}

ColoringResult RAParallel::localColor(const IGraph &G) const {
  // TODO: better coloring algorithm instead of greedy coloring.
  ColoringResult Result;
  std::vector<std::pair<Register, float>> RegHeap;
  const static auto Cmp = [](const std::pair<Register, float> &A,
                             const std::pair<Register, float> &B) {
    return A.second < B.second;
  };
  for (auto &[V, _] : G) {
    // order regs by spill weight to allocate the most costly to spill ones
    // first
    RegHeap.push_back(std::make_pair(V, LIS->getInterval(V).weight()));
    std::push_heap(RegHeap.begin(), RegHeap.end(), Cmp);
  }
  std::map<MCRegister, std::shared_ptr<ColoringResult::Color>> UsedColors;
  while (!RegHeap.empty()) {
    std::pop_heap(RegHeap.begin(), RegHeap.end(), Cmp);
    auto [V, _] = RegHeap.back();
    RegHeap.pop_back();
    const auto PRegs = AllocationOrder::create(V, *VRM, RegClassInfo, Matrix);
    bool Colored = false;
    for (auto PReg : PRegs) {
      if (!doesInterfere(G, Result, V, PReg)) {
        // if we have an available color, assign it, otherwise
        Colored = true;
        Result.RegToColor.insert(
            std::make_pair(V, chooseColor(PReg, Result, UsedColors, V)));
        break;
      }
    }
    if (!Colored) {
      Result.RegToColor.insert(std::make_pair(V, nullptr));
    }
  }
  return Result;
}

/// Marks `C` and all its register units as used in `UsedColors`.
void setColorUsed(ColoringResult::Color &C, std::set<MCRegister> &UsedColors,
                  const TargetRegisterInfo *TRI) {
  UsedColors.insert(C.getPReg());
  for (auto Unit : TRI->regunits(C.getPReg())) {
    UsedColors.insert(Unit);
  }
}

/// Returns true if `Color` is available in `UsedColors`. Checks for
/// `Color` and all its register units in `UsedColors` to determine if it is
/// available.
[[nodiscard]] bool colorIsAvailable(MCRegister Color,
                                    const ColoringResult::Color &CurColorClass,
                                    const std::set<MCRegister> &UsedColors,
                                    const TargetRegisterInfo *TRI,
                                    LiveRegMatrix *Matrix,
                                    const LiveIntervals *LIS) {
  auto &Mem = CurColorClass.members();
  for (auto M : Mem) {
    // TODO: mutation of the matrix
    if (Matrix->checkRegUnitInterference(LIS->getInterval(M), Color)) {
      // a member interferes with precolored registers
      return false;
    }
  }
  if (UsedColors.find(Color) != UsedColors.end()) {
    // UsedColors contains the color
    return false;
  }
  for (auto Unit : TRI->regunits(Color)) {
    if (UsedColors.find(Unit) != UsedColors.end()) {
      // Used colors contains a reg unit of the color
      return false;
    }
  }
  return true;
}

ColoringResult RAParallel::recolorRight(
    std::map<MCRegister, std::shared_ptr<ColoringResult::Color>>
        &ExistingColors,
    std::set<MCRegister> &UsedColors, std::set<Register> &NeedColors,
    std::unordered_set<ColoringResult::Color *> &ColorsToChange,
    ColoringResult &Right, ColoringResult &&Result) const {
  // for nodes in the clique that need colors in the right subgraph
  for (auto &Node : NeedColors) {
    // node is spilled in the left subgraph but not the right subgraph
    auto NodeColor = Right.RegToColor.at(Node);
    // we must change the color
    auto NewColor = getAvailableColor(Node, NodeColor, UsedColors, Result,
                                      ExistingColors, true);
    setColorUsed(*NewColor, UsedColors, TRI);
    ColorsToChange.erase(NodeColor.get());
    LLVM_DEBUG(dbgs() << "Needed to change: "
                      << printReg(NodeColor->getPReg(), TRI) << " to "
                      << printReg(NewColor->getPReg(), TRI) << "\n";);

    assert(NewColor != nullptr);
    NodeColor->setColor(NewColor);
  }
  for (auto &[Reg, Color] : Right.RegToColor) {
    Result.RegToColor.insert(std::make_pair(Reg, Color));
    if (Color != nullptr) {
      if (auto It = ColorsToChange.find(Color.get());
          It != ColorsToChange.end()) {
        ColorsToChange.erase(It);
        auto NewColor =
            getAvailableColor(Reg, *Color, UsedColors, Result, ExistingColors);
        LLVM_DEBUG(dbgs() << "Changing: " << printReg(Color->getPReg(), TRI)
                          << " {";
                   for (auto &M
                        : Color->members()) {
                     dbgs() << printReg(M, TRI) << " ";
                   } dbgs()
                   << "} to " << printReg(NewColor->getPReg(), TRI) << "\n";);
        Color->setColor(NewColor);
        setColorUsed(*Color, UsedColors, TRI);
      }
    }
  }
  return Result;
}

ColoringResult
RAParallel::mergeResults(ColoringResult &&Left, ColoringResult &&Right,
                         const std::set<Register> &Clique) const {
  LLVM_DEBUG(dbgs() << "*** Merging ***\n");
  LLVM_DEBUG(Left.print(dbgs(), TRI) << "with\n");
  LLVM_DEBUG(Right.print(dbgs(), TRI));
  LLVM_DEBUG(dbgs() << "{ "; for (auto &Node
                                  : Clique) {
    dbgs() << printReg(Node, TRI) << " ";
  } dbgs() << "}\n";);
  auto &Result = Left;
  // The set of colors that have been used in the recolored right subgraph
  std::set<MCRegister> UsedColors;
  // Colors in the right subgraph that need to be changed
  std::unordered_set<ColoringResult::Color *> ColorsToChange;
  std::map<MCRegister, std::shared_ptr<ColoringResult::Color>> ExistingColors;
  for (auto &[VReg, Color] : Right.RegToColor) {
    if (Color != nullptr) {
      ColorsToChange.insert(Color.get());
    }
  }
  for (auto &LColor : Left.Colors) {
    ExistingColors.insert(std::make_pair(LColor->getPReg(), LColor));
  }
  // Registers of the right subgraph that need to be colored
  std::set<Register> NeedColors;
  for (auto Node : Clique) {
    const auto LColor = Left.RegToColor.at(Node);
    auto RColor = Right.RegToColor.at(Node);
    if (LColor != nullptr && RColor != nullptr) {
      // replace rcolor by lcolor in right subgraph
      auto OldRColor = RColor->getPReg();
      RColor->setColor(LColor);
      setColorUsed(*LColor, UsedColors, TRI);
      ColorsToChange.erase(RColor.get());
      LLVM_DEBUG(dbgs() << "Replacing " << printReg(OldRColor, TRI) << " with "
                        << printReg(RColor->getPReg(), TRI) << " { ";
                 for (auto &M
                      : LColor->members()) {
                   dbgs() << printReg(M, TRI) << " ";
                 } dbgs()
                 << "}\n";);
    } else if (LColor != nullptr) {
      // Node is spilled in right subgraph
      // Node will be spilled in result
      // TODO: split live range if node is spilled in one subgraph and colored
      // in the other
      Result.RegToColor.insert(std::make_pair(Node, nullptr));
    } else if (RColor != nullptr) {
      // Node is spilled in left subgraph

      // we need to recolor the set of nodes in the right subgraph that shared
      // `Node`'s color
      NeedColors.insert(Node);
    }
  }
  auto Res = recolorRight(ExistingColors, UsedColors, NeedColors,
                          ColorsToChange, Right, std::move(Result));
  LLVM_DEBUG(Res.print(dbgs() << "====\n", TRI) << "\n\n");
  return Res;
}

std::shared_ptr<ColoringResult::Color> RAParallel::getAvailableColor(
    Register VReg, const ColoringResult::Color &CurColorClass,
    const std::set<MCRegister> &UsedColors, ColoringResult &Result,
    std::map<MCRegister, std::shared_ptr<ColoringResult::Color>>
        &ExistingColors,
    bool MustChange) const {
  const auto CurPReg = CurColorClass.getPReg();
  if (!MustChange) {
    // we don't have to change the color, so try to keep the current color
    if (colorIsAvailable(CurPReg, CurColorClass, UsedColors, TRI, Matrix,
                         LIS)) {
      return chooseColor(CurPReg, Result, ExistingColors);
    }
  }
  auto Candidates = AllocationOrder::create(VReg, *VRM, RegClassInfo, Matrix);
  for (auto &[PReg, _] : ExistingColors) {
    if (orderContainsReg(Candidates, PReg) &&
        colorIsAvailable(PReg, CurColorClass, UsedColors, TRI, Matrix, LIS) &&
        (!MustChange || CurPReg != PReg)) {
      return chooseColor(PReg, Result, ExistingColors);
    }
  }
  for (auto PotentialColor : Candidates) {
    if (colorIsAvailable(PotentialColor, CurColorClass, UsedColors, TRI, Matrix,
                         LIS) &&
        (!MustChange || CurPReg != PotentialColor)) {
      return chooseColor(PotentialColor, Result, ExistingColors);
    }
  }
  assert(false);
  return nullptr;
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
  const auto G = computeInterference();
  if (OutputInterferenceGraph.hasArgStr()) {
    debugInterferenceGraph(G, TRI, MF->getName().str(),
                           OutputInterferenceGraph.getValue());
  }
  const auto T = buildPartitionTree(G, TRI);
  if (OutputPartitionTree.hasArgStr()) {
    debugPartitionTree(T, TRI, MF->getName().str(),
                       OutputPartitionTree.getValue());
  }
  auto Coloring = colorPhysRegsTree(T);
  allocPhysRegsFinal(Coloring);
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

ColoringResult RAParallel::colorPhysRegsTree(const PartitionTree &T) const {
  if (T.Left && T.Right) {
    assert(T.Regs.empty() && "expected internal node");
    // TODO: spawn task for left subtree
    auto L = colorPhysRegsTree(*T.Left);
    auto R = colorPhysRegsTree(*T.Right);
    return mergeResults(std::move(L), std::move(R), T.SeparatingClique);
  }
  assert(!T.Left && !T.Right && T.SeparatingClique.empty() &&
         "expected leaf node");
  LLVM_DEBUG(dbgs() << "allocating: "; for (auto &[R, _]
                                            : T.Regs) {
    dbgs() << printReg(R, TRI) << ", ";
  } dbgs() << "\n");
  return localColor(T.Regs);
}

void RAParallel::allocPhysRegsFinal(const ColoringResult &Coloring) {
  for (auto &[Reg, Color] : Coloring.RegToColor) {
    if (MRI->reg_nodbg_empty(Reg)) {
      continue;
    }
    if (Color != nullptr) {
      VRM->assignVirt2Phys(Reg, Color->getPReg());
    } else {
      enqueue(&LIS->getInterval(Reg));
    }
  }

  // Continue assigning vregs one at a time to available physical registers.
  while (const LiveInterval *VirtReg = dequeue()) {
    assert(!VRM->hasPhys(VirtReg->reg()));

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
