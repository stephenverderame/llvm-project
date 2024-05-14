//===-- RegAllocParallel.cpp - Parallel Register Allocator ----------------===//
//
//
//
//===----------------------------------------------------------------------===//
//
// This file defines the RAParallel function pass, which provides a parallel
// implementation of the RABasic register allocator.
//
//===----------------------------------------------------------------------===//

#include "AllocationOrder.h"
#include "IGraph.hpp"
#include "LiveDebugVariables.h"
#include "RegAllocBase.h"
#include "RegisterCoalescer.h"
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
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

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
static cl::opt<std::string>
    OutputOnly("debug-func",
               cl::desc("Specify a comma separated list of "
                        "functions to save debug output for"),
               cl::value_desc("function1,function2,..."));

static cl::opt<std::string>
    DebugOld("debug-old-partitions",
             cl::desc("Specify file to write old partition tree to"),
             cl::value_desc("filename"));
namespace {
struct CompSpillWeight {
  bool operator()(const LiveInterval *A, const LiveInterval *B) const {
    return A->weight() < B->weight();
  }
};
} // namespace

namespace {

/// Gets a `std::shared_ptr<Color>` for `AvailableColor`.
/// If the color hasn't been used before, adds it to the
/// `Result.Colors` list and updates `ExistingColors`. Otherwise, returns the
/// existing color.
template <typename... T>
auto chooseColor(MCRegister AvailableColor,
                 std::map<MCRegister, std::shared_ptr<Color>> &ExistingColors,
                 T &&...ColorArgs) {
  if (auto ExistingColor = ExistingColors.find(AvailableColor);
      ExistingColor != ExistingColors.end()) {
    return ExistingColor->second;
  }
  auto NewColor =
      std::make_shared<Color>(AvailableColor, std::forward<T>(ColorArgs)...);
  // Result.Colors.push_back(NewColor);
  ExistingColors.insert(std::make_pair(AvailableColor, NewColor));
  return NewColor;
}

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

  /// Info needed for merging coloring results.
  struct MergeCtx {
  private:
    /// A set of colors in the right subgraph that still need to be processed
    /// and potentially have their colors aliased
    std::unordered_set<const Color *> ColorsToChange;

  public:
    /// A set of registers that are already in the result mapped to their color
    /// node in the coloring union find
    std::map<MCRegister, std::shared_ptr<Color>> ExistingColors;
    /// A set of reg units that have been used by another equivalence class of
    /// nodes in the interference graph
    std::set<MCRegUnit> UsedColors;
    /// A set of colors in the right subgraph that need their color changed
    std::set<std::shared_ptr<Color>> NeedColors;

    /// Returns true if `C` should be recolored. This occurs when the root node
    /// of `C` is in `ColorsToChange`.
    [[nodiscard]] bool shouldRecolor(const Color &C) const {
      return ColorsToChange.find(C.getRoot()) != ColorsToChange.end();
    }

    /// Marks `C` as recolored by removing its root from `ColorsToChange`.
    /// `C` should be a color in the right subgraph of the partition tree.
    /// Therefore this method should be called BEFORE the color is recolored.
    void setRecolored(const Color &C) { ColorsToChange.erase(C.getRoot()); }

    /// Adds `C` to the set of colors that need to be changed.
    void addColorToChange(const Color &C) {
      ColorsToChange.insert(C.getRoot());
    }

    /// Marks `C` and all its register units as used in `UsedColors`.
    void setColorUsed(const Color &C, const TargetRegisterInfo *TRI) {
      for (auto Unit : TRI->regunits(C.getPReg())) {
        UsedColors.insert(Unit);
      }
      LLVM_DEBUG(dbgs() << "Adding " << printReg(C.getPReg(), TRI)
                        << " UsedColors [ ";
                 for (auto Unit
                      : UsedColors) {
                   dbgs() << printRegUnit(Unit, TRI) << " ";
                 } dbgs()
                 << "]\n";);
    }

    /// Returns the set of physical registers that need to be changed in the
    /// coloring.
    [[nodiscard]] std::set<MCRegister> getColorsToChange() const {
      std::set<MCRegister> Result;
      for (const auto &Color : ColorsToChange) {
        Result.insert(Color->getPReg());
      }
      return Result;
    }

    /// Gets a `std::shared_ptr<Color>` for `AvailableColor`.
    /// If the color hasn't been used before, adds it to the
    /// `Result.Colors` list and updates `ExistingColors`. Otherwise, returns
    /// the existing color.
    template <typename... T>
    auto chooseColor(MCRegister AvailableColor, T &&...ColorArgs) {
      return ::chooseColor(AvailableColor, ExistingColors,
                           std::forward<T>(ColorArgs)...);
    }
  };

  /// Computes an interference graph from the live intervals in the function.
  [[nodiscard]] IGraph computeInterference();

  /// Allocates physical registers and handles spills for a coloring result.
  void setupAllocation(const ColoringResult &Coloring);
  /// Recursively builds a coloring result from a partition tree.
  [[nodiscard]] ColoringResult colorPhysRegsTree(const PartitionTree *T,
                                                 const PRegMap &M) const;

  /// Performs register allocation on a leaf node of the partition tree
  [[nodiscard]] ColoringResult localColor(const IGraph &G,
                                          const PRegMap &M) const;
  /// Returns true if `VReg` interferes with `PReg` given the current
  /// interference graph and coloring.
  [[nodiscard]] bool doesInterfere(const IGraph &G, const PRegMap &M,
                                   const ColoringResult &CurColoring,
                                   Register VReg, MCRegister PReg) const;
  /// Merges two coloring results together of sibilings in the partition tree
  /// Requires that the only shared virtual registers between the two coloring
  /// results are in the clique.
  [[nodiscard]] ColoringResult
  mergeResults(ColoringResult &&A, ColoringResult &&B,
               const std::unordered_set<Register> &Clique,
               const PRegMap &G) const;

  /// Gets an available color for `NodePReg` from `AvailableColors`. If
  /// `MustChange` is true, the color must be different from `CurPReg`. If the
  /// color hasn't been used before, adds it to the `Result.Colors` list.
  /// The returned color is *NOT* removed from `AvailableColors`.
  /// @param VReg The virtual register that needs a color.
  /// @param RightColorClass The current color class of the virtual register in
  /// right subgraph.
  /// @param UsedColors The set of physical registers that have been used in the
  /// coloring.
  /// @param Result The current coloring result.
  /// @param ExistingColors A map of physical registers to their color in the
  /// resulting coloring. Updated with new colors as they are added.
  /// @param MustChange If true, the color must be different from `CurPReg`.
  /// @return The color to use for `VPReg`.
  [[nodiscard]] std::shared_ptr<Color>
  getAvailableColor(Register VReg, const PRegMap &M,
                    const std::shared_ptr<Color> &RightColorClass,
                    MergeCtx &Ctx, bool MustChange = false) const;

  /// Recolors the right subgraph of the partition tree. Registers in the right
  /// subgraph that are in `NeedColors` will be colored with a new color.
  /// Registers in the right subgraph that are already colored will be recolored
  /// if they have not been recolored already (not in `ColorsToChange`).
  /// @param Ctx The merge context that contains the information about the
  /// coloring of the right subgraph.
  /// @param Right The coloring result of the right subgraph.
  /// @param Result The current coloring result.
  /// @return The new coloring result with the right subgraph colored and added
  /// to the existing coloring result.
  ColoringResult recolorRight(MergeCtx &Ctx, const PRegMap &M,
                              ColoringResult &Right,
                              ColoringResult &&Result) const;

  /// Upon recoloring `OldColor` to `NewColor` from the right subgraph,
  /// search through the right subgraph and attempt to recolor any aliases of
  /// `OldColor` to match the corresponding alias of `NewColor`. If aliases
  /// cannot be recolored to the same color as `NewColor`, they will be
  /// ignored.
  void recolorAliases(ColoringResult &Right, MCRegister OldColor,
                      Color &NewColor, MergeCtx &Ctx, const PRegMap &M) const;

  // do nothing and use the coloring to seed the live regs
  void seedLiveRegs() override {}

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

inline void ltrim(std::string &S) {
  S.erase(S.begin(), std::find_if(S.begin(), S.end(), [](unsigned char C) {
            return !std::isspace(C);
          }));
}

// trim from end (in place)
inline void rtrim(std::string &S) {
  S.erase(std::find_if(S.rbegin(), S.rend(),
                       [](unsigned char C) { return !std::isspace(C); })
              .base(),
          S.end());
}

/// Returns true if `Name` matches `Pattern`. `Pattern` can contain a `*` at the
/// end to match any substring.
[[nodiscard]] bool matches(const std::string &Name,
                           const std::string &Pattern) {
  return Pattern == Name ||
         (Pattern.find('*') != std::string::npos &&
          Name.find(Pattern.substr(0, Pattern.find('*'))) == 0);
}

// Determines whether we should output graphs for `FuncName` based on the
// `OutputOnly`
bool canDebug(const StringRef &FuncName) {
  if (OutputOnly.hasArgStr()) {
    auto Trimmed = OutputOnly.getValue();
    ltrim(Trimmed);
    rtrim(Trimmed);
    if (Trimmed.empty()) {
      return true;
    }
    std::istringstream Iss(OutputOnly.getValue());
    const auto Name = demangle(FuncName);
    std::string Opt;
    while (std::getline(Iss, Opt, ',')) {
      if (!Opt.empty()) {
        if (Opt[0] == '-') {
          const auto OptName = Opt.substr(1);
          if (matches(Name, OptName)) {
            return false;
          }
        } else if (matches(Name, Opt)) {
          return true;
        }
      }
    }
    return false;
  }
  return true;
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
  AU.addRequired<MachinePostDominatorTree>();
  AU.addPreserved<MachinePostDominatorTree>();
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
          G.insert(std::make_pair(Reg, std::unordered_set<Register>()));
        }
        continue;
      }
      if (LIS->getInterval(Reg).overlaps(LIS->getInterval(Reg2))) {
        if (G.find(Reg) == G.end()) {
          G.insert(std::make_pair(Reg, std::unordered_set<Register>()));
        }
        if (G.find(Reg2) == G.end()) {
          G.insert(std::make_pair(Reg2, std::unordered_set<Register>()));
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
          (void)G;
        } else {
          G[Reg].insert(Reg2);
          G[Reg2].insert(Reg);
        }
      }
    }
  }
  return G;
}

bool RAParallel::doesInterfere(const IGraph &G, const PRegMap &M,
                               const ColoringResult &CurColoring, Register VReg,
                               MCRegister PReg) const {
  if (!M.isValidAssignment(VReg, PReg)) {
    return true;
  }
  for (auto Neighbor : G.at(VReg)) {
    if (auto NeighorCol = CurColoring.RegToColor.find(Neighbor);
        NeighorCol != CurColoring.RegToColor.end() &&
        NeighorCol->second != nullptr) {
      const auto NeighborReg = NeighorCol->second->getPReg();
      if (TRI->regsOverlap(PReg, NeighborReg)) {
        return true;
      }
    }
  }
  return false;
}

ColoringResult RAParallel::localColor(const IGraph &G, const PRegMap &M) const {
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
  std::map<MCRegister, std::shared_ptr<Color>> UsedColors;
  while (!RegHeap.empty()) {
    std::pop_heap(RegHeap.begin(), RegHeap.end(), Cmp);
    auto [VReg, _] = RegHeap.back();
    RegHeap.pop_back();
    const auto &PRegs = M.getAllocationOrder(VReg);
    bool Colored = false;
    for (auto PReg : PRegs) {
      if (!doesInterfere(G, M, Result, VReg, PReg)) {
        // if we have an available color, assign it, otherwise
        Colored = true;
        Result.RegToColor[VReg] = chooseColor(PReg, UsedColors, VReg);
        break;
      }
    }
    if (!Colored) {
      Result.RegToColor[VReg] = nullptr;
    }
  }
  return Result;
}

/// Returns true if `Color` is available in `UsedColors`. Checks for
/// `Color` and all its register units in `UsedColors` to determine if it is
/// available.
[[nodiscard]] bool colorIsAvailable(MCRegister Col, const Color &CurColorClass,
                                    const std::set<MCRegUnit> &UsedColors,
                                    const TargetRegisterInfo *TRI,
                                    const PRegMap &M) {
  auto &Mem = CurColorClass.members();
  for (auto Member : Mem) {
    if (!M.isValidAssignment(Member, Col)) {
      // a member interferes with precolored registers
      return false;
    }
  }
  for (auto Unit : TRI->regunits(Col)) {
    if (UsedColors.find(Unit) != UsedColors.end()) {
      // Used colors contains a reg unit of the color
      return false;
    }
  }
  return true;
}

void RAParallel::recolorAliases(ColoringResult &Right, MCRegister OldColor,
                                Color &NewColor, MergeCtx &Ctx,
                                const PRegMap &M) const {
  for (auto [Reg, Color] : Right.RegToColor) {
    if (Color != nullptr && !Color->members().empty() &&
        Ctx.shouldRecolor(Color)) {
      const auto ColorPReg = Color->getPReg();
      if (TRI->regsOverlap(OldColor, ColorPReg)) {
        // for each alias of the old color
        std::optional<MCRegister> NewSizedPReg;
        {
          // find corresponding alias of the new color
          auto &Candidates = M.getAllocationOrder(Reg);
          auto NewPReg = NewColor.getPReg();
          for (auto NewPRegChoice : Candidates) {
            if (TRI->regsOverlap(NewPReg, NewPRegChoice)) {
              NewSizedPReg = NewPRegChoice;
              break;
            }
          }
        }
        if (NewSizedPReg.has_value()) {
          auto NewSizedColor = Ctx.chooseColor(NewSizedPReg.value(), Reg);
          Ctx.setRecolored(Color);
          LLVM_DEBUG(dbgs() << "Changing (via alias of "
                            << printReg(NewColor.getPReg(), TRI) << "): "
                            << printReg(Color->getPReg(), TRI) << " { ";
                     for (auto &M
                          : Color->members()) {
                       dbgs() << printReg(M, TRI) << " ";
                     } dbgs()
                     << "} to " << printReg(NewSizedColor->getPReg(), TRI)
                     << "\n";);
          Color->setColor(NewSizedColor);
        } else {
          // NOTE: I think this breaks the proof that we don't need to spill
          // during merging. We could check aliases when we initially color a
          // color class, but I think this is fine, at least for the tests we
          // currently have.
          LLVM_DEBUG(dbgs()
                         << "Ignoring alias recolor of " << printReg(Reg, TRI)
                         << " to " << printReg(ColorPReg, TRI)
                         << " because no available color\n";);
        }
      }
    }
  }
}

ColoringResult RAParallel::recolorRight(MergeCtx &Ctx, const PRegMap &M,
                                        ColoringResult &Right,
                                        ColoringResult &&Result) const {
  // for nodes in the clique that need colors in the right subgraph
  for (auto &NodeColor : Ctx.NeedColors) {
    // node is spilled in the left subgraph but not the right subgraph
    if (Ctx.shouldRecolor(NodeColor) && !NodeColor->members().empty()) {
      // we must change the color
      auto NewColor = getAvailableColor(*NodeColor->members().begin(), M,
                                        NodeColor, Ctx, true);
      assert(NewColor != nullptr);
      assert(NewColor->getPReg() != NodeColor->getPReg());
      Ctx.setColorUsed(*NewColor, TRI);
      Ctx.setRecolored(*NodeColor);
      LLVM_DEBUG(dbgs() << "Needed to change: "
                        << printReg(NodeColor->getPReg(), TRI) << " to "
                        << printReg(NewColor->getPReg(), TRI) << "\n";);
      auto OldPReg = NodeColor->getPReg();
      NodeColor->setColor(NewColor);
      recolorAliases(Right, OldPReg, *NewColor, Ctx, M);
    }
  }
  for (auto &[Reg, Color] : Right.RegToColor) {
    if (Result.RegToColor.find(Reg) == Result.RegToColor.end()) {
      // color the node if its not in the left subgraph
      // if its in the left subgraph, this has been handled already since if
      // its in both it must be in the clique
      Result.RegToColor.insert(std::make_pair(Reg, Color));
    }
  }
  for (auto &[_, Color] : Right.RegToColor) {
    if (Color == nullptr) {
      continue;
    }
    if (Ctx.shouldRecolor(Color)) {
      Ctx.setRecolored(Color);
      if (Color->members().empty()) {
        LLVM_DEBUG(dbgs() << "Color " << printReg(Color->getPReg(), TRI)
                          << " has no members\n";);
        continue;
      }
      auto Mem = *Color->members().begin();
      auto NewColor = getAvailableColor(Mem, M, Color, Ctx);
      LLVM_DEBUG(
          dbgs() << "Changing: " << printReg(Color->getPReg(), TRI) << " { ";
          for (auto &M
               : Color->members()) { dbgs() << printReg(M, TRI) << " "; } dbgs()
          << "} to " << printReg(NewColor->getPReg(), TRI) << "\n";);
      auto OldPReg = Color->getPReg();
      Color->setColor(NewColor);
      recolorAliases(Right, OldPReg, *Color, Ctx, M);
      Ctx.setColorUsed(*NewColor, TRI);
    }
  }
  return Result;
}

ColoringResult
RAParallel::mergeResults(ColoringResult &&Left, ColoringResult &&Right,
                         const std::unordered_set<Register> &Clique,
                         const PRegMap &M) const {
  LLVM_DEBUG(dbgs() << "*** Merging ***\n");
  LLVM_DEBUG(Left.print(dbgs(), TRI) << "with\n");
  LLVM_DEBUG(Right.print(dbgs(), TRI));
  LLVM_DEBUG(dbgs() << "{ "; for (auto &Node
                                  : Clique) {
    dbgs() << printReg(Node, TRI) << " ";
  } dbgs() << "}\n";);
  auto &Result = Left;
  MergeCtx Ctx;
  for (auto &[VReg, Color] : Right.RegToColor) {
    // TODO: we could probably do this worklist style and only recolor something
    // if we create a conflict
    if (Color != nullptr) {
      Ctx.addColorToChange(Color);
    }
  }
  for (auto &[_, LColor] : Left.RegToColor) {
    if (LColor != nullptr) {
      Ctx.ExistingColors.insert(
          std::make_pair(LColor->getPReg(), Color::getRootPtr(LColor)));
    }
  }
  for (auto Node : Clique) {
    const auto LColor = Left.RegToColor.at(Node);
    auto RColor = Right.RegToColor.at(Node);
    if (LColor != nullptr && RColor != nullptr) {
      if (!Ctx.shouldRecolor(RColor)) {
        // with the way we merge cliques in the new method, we might not
        // actually have a clique separator anymore, so it could be the case
        // that a node is part of a color class that we already colored.
        continue;
      }
      // replace rcolor by lcolor in right subgraph
      auto OldRColor = RColor->getPReg();
      Ctx.setRecolored(*RColor);
      RColor->setColor(LColor);
      Ctx.setColorUsed(*LColor, TRI);
      LLVM_DEBUG(dbgs() << "Replacing " << printReg(OldRColor, TRI) << " with "
                        << printReg(RColor->getPReg(), TRI) << " { ";
                 for (auto &M
                      : LColor->members()) {
                   dbgs() << printReg(M, TRI) << " ";
                 } dbgs()
                 << "}\n";);
      recolorAliases(Right, OldRColor, *RColor, Ctx, M);
    } else if (LColor != nullptr) {
      // Node is spilled in right subgraph but colored in the left

      // Node needs to be spilled in result
      // TODO: split live range if node is spilled in one subgraph and colored
      // in the other
      Result.RegToColor.at(Node)->members().erase(Node);
      Result.RegToColor[Node] = nullptr;
      LLVM_DEBUG(dbgs() << "Spilling " << printReg(Node, TRI)
                        << " from left\n";);
    } else if (RColor != nullptr) {
      if (!Ctx.shouldRecolor(RColor)) {
        // see earlier comment
        continue;
      }
      // Node is spilled in left subgraph, but colored in the right
      Right.RegToColor.at(Node)->members().erase(Node);
      Right.RegToColor[Node] = nullptr;
      LLVM_DEBUG(dbgs() << "Spilling " << printReg(Node, TRI)
                        << " from right\n";);

      if (!RColor->members().empty()) {
        // if there are other nodes in the right subgraph with the same color,
        // we need to recolor them
        Ctx.NeedColors.insert(Color::getRootPtr(RColor));
      } else {
        // no other reg has this color, so it's free to use
        Ctx.setRecolored(RColor);
      }
    }
  }
  LLVM_DEBUG(dbgs() << "Colors To Change: [ "; for (auto PReg
                                                    : Ctx.getColorsToChange()) {
    dbgs() << printReg(PReg, TRI) << " ";
  } dbgs() << "]\n");
  // TODO: if we create no conflicts then we don't need to recolor anything
  auto Res = recolorRight(Ctx, M, Right, std::move(Result));
  LLVM_DEBUG(Res.print(dbgs() << "====\n", TRI) << "\n\n");
  return Res;
}

std::shared_ptr<Color>
RAParallel::getAvailableColor(Register VReg, const PRegMap &M,
                              const std::shared_ptr<Color> &RightColorClass,
                              MergeCtx &Ctx, bool MustChange) const {
  const auto CurPReg = RightColorClass->getPReg();
  if (!MustChange) {
    // we don't have to change the color, so try to keep the current color
    if (colorIsAvailable(CurPReg, RightColorClass, Ctx.UsedColors, TRI, M)) {
      // choose a color from the existing set or make a new one in the result
      return Ctx.chooseColor(CurPReg);
    }
  }
  for (auto &[PReg, Color] : Ctx.ExistingColors) {
    if (colorIsAvailable(PReg, RightColorClass, Ctx.UsedColors, TRI, M) &&
        (!MustChange || CurPReg != PReg)) {
      return Color;
    }
  }
  const auto &Candidates = M.getAllocationOrder(VReg);
  for (auto PotentialColor : Candidates) {
    if (colorIsAvailable(PotentialColor, RightColorClass, Ctx.UsedColors, TRI,
                         M) &&
        (!MustChange || CurPReg != PotentialColor)) {
      return Ctx.chooseColor(PotentialColor);
    }
  }
  LLVM_DEBUG(
      dbgs() << "No available colors for changing "
             << printReg(RightColorClass->getPReg(), TRI) << " { ";
      for (auto M
           : RightColorClass->members()) {
        dbgs() << printReg(M, TRI) << " ";
      } dbgs()
      << "} out of [ ";
      for (auto PReg
           : Candidates) { dbgs() << printReg(PReg, TRI) << " "; } dbgs()
      << "]\n";
      dbgs() << "Used colors: [ "; for (auto &Color
                                        : Ctx.UsedColors) {
        dbgs() << printReg(Color, TRI) << " ";
      } dbgs() << "]\n";);
  assert(false);
  return nullptr;
}

bool RAParallel::runOnMachineFunction(MachineFunction &Mf) {
  LLVM_DEBUG(dbgs() << "********** Parallel REGISTER ALLOCATION **********\n"
                    << "********** Function: " << Mf.getName() << '\n';);

  MF = &Mf;
  LLVM_DEBUG(Mf.print(dbgs()); dbgs() << "\n\n\n";);
  RegAllocBase::init(getAnalysis<VirtRegMap>(), getAnalysis<LiveIntervals>(),
                     getAnalysis<LiveRegMatrix>());
  VirtRegAuxInfo VRAI(*MF, *LIS, *VRM, getAnalysis<MachineLoopInfo>(),
                      getAnalysis<MachineBlockFrequencyInfo>());
  VRAI.calculateSpillWeightsAndHints();

  SpillerInstance.reset(createInlineSpiller(*this, *MF, *VRM, VRAI));
  const auto G = computeInterference();
  const auto PregMap = PRegMap{G, *VRM, RegClassInfo, Matrix, LIS};
  LLVM_DEBUG(if (canDebug(Mf.getName())) {
    if (OutputInterferenceGraph.hasArgStr()) {
      debugInterferenceGraph(G, TRI, demangle(MF->getName()),
                             OutputInterferenceGraph.getValue());
    }
  });
  const auto &PDT = getAnalysis<MachinePostDominatorTree>();
  // const auto T = buildPartitionTree(G, TRI);
  const auto T = getPartitions(Mf, *LIS, PDT, TRI, G);
  LLVM_DEBUG(if (canDebug(Mf.getName())) {
    if (OutputPartitionTree.hasArgStr()) {
      debugPartitionTree(*T, TRI, demangle(MF->getName()),
                         OutputPartitionTree.getValue());
    }
  });
  LLVM_DEBUG(if (canDebug(Mf.getName())) {
    if (DebugOld.hasArgStr()) {
      const auto T2 = buildPartitionTree(G, TRI);
      debugPartitionTree(T2, TRI, demangle(MF->getName()), DebugOld.getValue());
    }
  });
  ColoringResult Coloring;
#pragma omp parallel default(shared)
  {
#pragma omp single
    Coloring = colorPhysRegsTree(&*T, PregMap);
  }
  setupAllocation(Coloring);
  allocatePhysRegs();
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

ColoringResult RAParallel::colorPhysRegsTree(const PartitionTree *T,
                                             const PRegMap &G) const {
  if (T->Left && T->Right) {
    assert(T->Regs.empty() && "expected internal node");
    ColoringResult L;
#pragma omp task shared(L)
    L = colorPhysRegsTree(T->Left.get(), G);
    auto R = colorPhysRegsTree(T->Right.get(), G);
#pragma omp taskwait
    return mergeResults(std::move(L), std::move(R), T->SeparatingClique, G);
  }
  assert(!T->Left && !T->Right && T->SeparatingClique.empty() &&
         "expected leaf node");
  LLVM_DEBUG(dbgs() << "++ ALLOCATING: "; for (auto &[R, _]
                                               : T->Regs) {
    dbgs() << printReg(R, TRI) << ", ";
  } dbgs() << "++\n");
  return localColor(T->Regs, G);
}

void RAParallel::setupAllocation(const ColoringResult &Coloring) {
  for (auto &[Reg, Color] : Coloring.RegToColor) {
    if (MRI->reg_nodbg_empty(Reg)) {
      continue;
    }
    if (Color != nullptr) {
      // this also updates the virtual reg map
      Matrix->assign(LIS->getInterval(Reg), Color->getPReg());
    } else {
      enqueue(&LIS->getInterval(Reg));
    }
  }
}
