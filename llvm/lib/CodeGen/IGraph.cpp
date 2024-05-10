#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include <functional>
#include <queue>
#define DEBUG_TYPE "regalloc"
#include "AllocationOrder.h"
#include "IGraph.hpp"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <stack>
#include <unordered_set>

using namespace llvm;

namespace std {
bool operator<(const std::reference_wrapper<const LiveInterval> &A,
               const std::reference_wrapper<const LiveInterval> &B) {
  return A.get() < B.get();
}
} // namespace std

namespace {
/// Computes a perfect elimination order for the interference graph `G`.
/// @return a pair of the perfect elimination order and the set of vertices
/// that are separator generators.
std::pair<std::vector<Register>, std::unordered_set<Register>>
perfectElimOrder(const IGraph &G) {
  // maximum cardinality search, generates a perfect elimination order bc
  // ssa interference graphs are chordal
  std::vector<Register> Order;
  std::vector<std::pair<Register, int>> Heap;
  std::unordered_map<Register, int> Weights;
  // the set of registers that have already been placed in the order.
  // during the loop, for a given register, the set of numbered registers will
  // come after it in the elimination order
  std::unordered_set<Register> Numbered;
  std::unordered_set<Register> SeparatorGenerators;
  for (auto [V, _] : G) {
    Heap.emplace_back(std::make_pair(V, 0));
    Weights[V] = 0;
  }
  Order.resize(G.size());
  const auto Comp = [](const std::pair<Register, int> &A,
                       const std::pair<Register, int> &B) {
    return A.second < B.second;
  };
  int N = G.size() - 1;
  int LastWeight = -1;
  while (!Heap.empty()) {
    std::pop_heap(Heap.begin(), Heap.end(), Comp);
    const auto [V, W] = Heap.back();
    Heap.pop_back();
    if (Numbered.find(V) != Numbered.end()) {
      continue;
    }
    Numbered.insert(V);
    Order[N--] = V;
    if (W <= LastWeight) {
      // if W is less than or equal to the last weight, then V must be in a new
      // component since each vertex in the previous component will be
      // incremented by 1
      SeparatorGenerators.insert(V);
      // the separator generator will be the first vertex in the elimination
      // order in each clique separator
    }
    LastWeight = W;
    for (const auto U : G.at(V)) {
      if (Numbered.find(U) != Numbered.end()) {
        continue;
      }
      Weights[U] += 1;
      Heap.push_back(std::make_pair(U, Weights[U]));
      std::push_heap(Heap.begin(), Heap.end(), Comp);
    }
  }
  return {Order, SeparatorGenerators};
}

/// Returns the set of neighbors of `V` in `G` that are also in `ValidVertices`.
/// If `IncludeV` is true, `V` will be included in the set of neighbors,
/// regardless of whether it is in `ValidVertices`. Otherwise, `V` will never
/// be included in the set of neighbors.
std::unordered_set<Register>
neighbors(const IGraph &G, const std::unordered_set<Register> &ValidVertices,
          const Register V, bool IncludeV = false) {
  std::unordered_set<Register> Neighbors;
  for (auto U : G.at(V)) {
    if (ValidVertices.find(U) != ValidVertices.end()) {
      Neighbors.insert(U);
    }
  }
  if (IncludeV) {
    Neighbors.insert(V);
  }
  return Neighbors;
}

bool isCompleteSubgraph(const IGraph &G,
                        const std::unordered_set<Register> &Clique) {
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

std::unordered_set<Register>
connectedComponent(const IGraph &G, const std::unordered_set<Register> &Exclude,
                   Register X) {
  std::unordered_set<Register> Component;
  std::queue<Register> Q;
  Q.push(X);
  while (!Q.empty()) {
    auto V = Q.front();
    Q.pop();
    if (Component.find(V) != Component.end() ||
        Exclude.find(V) != Exclude.end()) {
      continue;
    }
    Component.insert(V);
    for (auto U : G.at(V)) {
      Q.push(U);
    }
  }
  return Component;
}

IGraph inducedSubgraph(const IGraph &G,
                       const std::unordered_set<Register> &Set) {
  IGraph Subgraph;
  for (auto V : Set) {
    Subgraph.insert(std::make_pair(V, std::unordered_set<Register>()));
    for (auto U : G.at(V)) {
      if (Set.find(U) != Set.end()) {
        Subgraph[V].insert(U);
      }
    }
  }
  return Subgraph;
}

/// Builds a partition tree from a set of atoms. The tree will be
/// a right-associative binary tree.
PartitionTree partitionTreeFromAtoms(
    std::vector<IGraph> &&Atoms,
    const std::vector<std::unordered_set<Register>> &CliqueSeparators) {
  assert(Atoms.size() == CliqueSeparators.size());
  PartitionTree T;
  PartitionTree *Last = &T;
  for (unsigned Idx = 0; Idx < Atoms.size(); ++Idx) {
    if (Idx < Atoms.size() - 1) {
      Last->Left = std::make_unique<PartitionTree>();
      Last->Left->Regs = std::move(Atoms[Idx]);
      Last->Right = std::make_unique<PartitionTree>();
      Last->SeparatingClique = CliqueSeparators[Idx];
      Last = Last->Right.get();
    } else {
      Last->Regs = std::move(Atoms[Idx]);
    }
  }
  return T;
}
} // namespace

namespace llvm {
IGraph operator-(const IGraph &G, const std::unordered_set<Register> &S) {
  IGraph H = G;
  for (auto N : S) {
    H.erase(N);
    for (auto &[_, Neighbors] : H) {
      Neighbors.erase(N);
    }
  }
  return H;
}
} // namespace llvm

namespace {
using LiveInVars =
    std::unordered_map<const MachineBasicBlock *, std::unordered_set<Register>>;

/// Get the live-in variables for each basic block in the function
LiveInVars liveIns(const MachineFunction &MF, const LiveIntervals &LIS) {
  LiveInVars LiveIns;
  std::queue<const MachineBasicBlock *> Q;
  for (auto &BB : MF) {
    Q.push(&BB);
  }
  while (!Q.empty()) {
    auto *MBB = Q.front();
    Q.pop();
    std::unordered_set<const MachineBasicBlock *> SuccChanged;
    for (auto &I : *MBB) {
      for (auto &Def : I.defs()) {
        if (Def.isReg() && Def.getReg().isVirtual() &&
            LIS.isLiveOutOfMBB(LIS.getInterval(Def.getReg()), MBB)) {
          for (const auto *Succ : MBB->successors()) {
            if (LiveIns[Succ].find(Def.getReg()) == LiveIns[Succ].end()) {
              LiveIns[Succ].insert(Def.getReg());
              SuccChanged.insert(Succ);
            }
          }
        }
      }
    }
    for (auto &LiveIn : LiveIns[MBB]) {
      if (LIS.isLiveOutOfMBB(LIS.getInterval(LiveIn), MBB)) {
        for (const auto *Succ : MBB->successors()) {
          if (LiveIns[Succ].find(LiveIn) == LiveIns[Succ].end()) {
            LiveIns[Succ].insert(LiveIn);
            SuccChanged.insert(Succ);
          }
        }
      }
    }
    for (auto *Succ : SuccChanged) {
      Q.push(Succ);
    }
  }
  return LiveIns;
}

/// Returns true if `Clique` is a clique separator of the basic block with
/// pre-intervals `Pre`, post-intervals `Post`, and clique `Clique`.
bool isCliqueSeparator(
    const std::set<std::reference_wrapper<const LiveInterval>> &Pre,
    const std::set<std::reference_wrapper<const LiveInterval>> &Clique,
    const std::set<std::reference_wrapper<const LiveInterval>> &Post) {
  // TODO: constrain the clique separators so that we do not choose too many
  if (Pre.empty() || Post.empty()) {
    return false;
  }
  std::set<std::reference_wrapper<const LiveInterval>> NonOverlappingPost,
      NonOverlappingPre;
  for (auto &C : Clique) {
    bool IsInPre = true;
    for (auto &P : Pre) {
      if (C.get().overlaps(P)) {
        IsInPre = false;
        break;
      }
    }
    if (IsInPre) {
      NonOverlappingPre.insert(C);
    }
    bool IsInPost = true;
    for (auto &P : Post) {
      if (C.get().overlaps(P)) {
        IsInPost = false;
        break;
      }
    }
    if (IsInPost) {
      NonOverlappingPost.insert(C);
    }
  }
  // NonOverlappingPre and NonOverlappingPost are not subsets of eachother
  // Then we can partition Clique into sets CliquePre which contains elements of
  // NonOverlappingPost that are not in NonOverlappingPre, CliquePost which
  // contains elements of NonOverlappingPre that are not in NonOverlappingPost,
  // and CliqueMiddle which contains the rest of the elements
  return std::any_of(NonOverlappingPre.begin(), NonOverlappingPre.end(),
                     [&NonOverlappingPost](const LiveInterval &C) {
                       return NonOverlappingPost.find(C) ==
                              NonOverlappingPost.end();
                     }) &&
         std::any_of(NonOverlappingPost.begin(), NonOverlappingPost.end(),
                     [&NonOverlappingPre](const LiveInterval &C) {
                       return NonOverlappingPre.find(C) ==
                              NonOverlappingPre.end();
                     });
}

class CodeSeparator {
public:
  std::set<std::reference_wrapper<const LiveInterval>> Clique;
  /// Set of live intervals that exist since the last separator
  std::set<std::reference_wrapper<const LiveInterval>> Partition;

private:
  // map from basic block to the start and end slot indexes of the clique
  std::unordered_map<const MachineBasicBlock *, std::pair<SlotIndex, SlotIndex>>
      SlotMap;

public:
  CodeSeparator(
      const MachineBasicBlock *MBB,
      const std::set<std::reference_wrapper<const LiveInterval>> &Clique,
      const std::set<std::reference_wrapper<const LiveInterval>> &Partition,
      const TargetRegisterInfo *TRI)
      : Clique(Clique), Partition(Partition) {
    bool Init = false;
    assert(!this->Clique.empty());
    SlotIndex StartSlot, EndSlot;
    for (auto &I : this->Clique) {
      if (!Init) {
        StartSlot = I.get().beginIndex();
        EndSlot = I.get().endIndex();
        Init = true;
      } else {
        StartSlot = std::min(StartSlot, I.get().beginIndex());
        EndSlot = std::max(EndSlot, I.get().endIndex());
      }
    }
    SlotMap.insert(std::make_pair(MBB, std::make_pair(StartSlot, EndSlot)));
    LLVM_DEBUG(
        dbgs() << "CodeSeparator: " << MBB->getName() << " [" << StartSlot
               << ", " << EndSlot << "]: { ";
        for (auto &C
             : Clique) {
          dbgs() << printReg(C.get().reg(), TRI) << ", ";
        } dbgs()
        << "}\n\tSeparates: { ";
        for (auto &P
             : Partition) {
          dbgs() << printReg(P.get().reg(), TRI) << ", ";
        } dbgs()
        << "}\n";);
  }

  /// Merges a code separator into this one
  CodeSeparator &merge(const CodeSeparator &Other) {
    for (auto &[MBB, Slot] : Other.SlotMap) {
      SlotMap.insert(std::make_pair(MBB, Slot));
    }
    for (auto &C : Other.Clique) {
      Clique.insert(C);
    }
    for (auto &P : Other.Partition) {
      Partition.insert(P);
    }
    return *this;
  }
};

/// Gets a clique separator of a basic block
/// @param MBB the basic block to get the separator of
/// @param LIS the live intervals of the function
/// @param LiveIns the live-in variables of the function
/// @param Partition In: the set of live intervals since the last separator.
/// Out: the set of live intervals since the last separator along this path.
/// @return the clique separators of the basic block
std::vector<CodeSeparator>
bbSeparators(const MachineBasicBlock *MBB, const LiveIntervals &LIS,
             const LiveInVars &LiveIns,
             std::set<std::reference_wrapper<const LiveInterval>> &Partition,
             const TargetRegisterInfo *TRI) {
  std::vector<CodeSeparator> Separators;
  std::set<std::reference_wrapper<const LiveInterval>> Pre, Post, Clique;
  for (auto LI : LiveIns.at(MBB)) {
    Pre.insert(LIS.getInterval(LI));
  }
  for (auto It = MBB->instr_begin(); It != MBB->instr_end(); ++It) {
    const auto &I = *It;
    for (auto &Def : I.defs()) {
      if (!Def.isReg() || !Def.getReg().isVirtual()) {
        continue;
      }

      // remove the current interval from the post set and add to the clique
      Clique.insert(LIS.getInterval(Def.getReg()));
      Post.erase(LIS.getInterval(Def.getReg()));

      // for all intervals that haven't started yet and overlap with the current
      // interval, add to the post set
      auto It2 = It;
      for (++It2; It2 != MBB->instr_end(); ++It2) {
        for (auto &D2 : It2->defs()) {
          if (D2.isReg() && D2.getReg().isVirtual() &&
              LIS.getInterval(D2.getReg())
                  .overlaps(LIS.getInterval(Def.getReg()))) {
            Post.insert(LIS.getInterval(D2.getReg()));
          }
        }
      }
    }

    // for all intervals ending at the current program point,
    // remove from clique and add to pre
    std::unordered_set<const LiveInterval *> ToRemove;
    for (auto &C : Clique) {
      bool IsEnding = true;
      for (auto &D : I.defs()) {
        if (D.isReg() && D.getReg().isVirtual() &&
            (C.get().overlaps(LIS.getInterval(D.getReg())) ||
             C.get().reg() == D.getReg())) {
          IsEnding = false;
          break;
        }
      }
      if (IsEnding) {
        Pre.insert(C);
        ToRemove.insert(&C.get());
      }
    }
    for (const auto *C : ToRemove) {
      Clique.erase(*C);
    }
    ToRemove.clear();

    // for all spans that no longer overlap a member of the clique,
    // remove it from the Pre set
    for (auto &P : Pre) {
      bool CanRemove = true;
      for (auto &C : Clique) {
        if (P.get().overlaps(C.get())) {
          CanRemove = false;
          break;
        }
      }
      if (CanRemove) {
        ToRemove.insert(&P.get());
      }
    }
    for (const auto *P : ToRemove) {
      Pre.erase(*P);
    }

    if (isCliqueSeparator(Pre, Clique, Post)) {
      Separators.emplace_back(MBB, Clique, Partition, TRI);
      Partition.clear();
    } else {
      for (auto &Def : I.defs()) {
        if (Def.isReg() && Def.getReg().isVirtual()) {
          Partition.insert(LIS.getInterval(Def.getReg()));
        }
      }
    }
  }
  return Separators;
}

/// Gets a list of separators, in sequential order, along all paths from `Start`
/// to `End` in the control flow graph. `Start` is inclusive, `End` is
/// exclusive. Also returns the set of live intervals that are in the final
/// partition.
/// @param Start the start of the path
/// @param End the end of the path or nullptr if the path ends at the end of the
/// function
/// @param LIS the live intervals of the function
/// @param LiveIns the live-in variables of the function
/// @param PDT the post-dominator tree of the function
/// @param Visited the set of basic blocks that have already been visited in
/// some path
/// @param Partition the set of live intervals since the last separator
std::pair<std::vector<CodeSeparator>,
          std::set<std::reference_wrapper<const LiveInterval>>>
getSeparatorsAlongPath(
    const MachineBasicBlock *Start, const MachineBasicBlock *End,
    const LiveIntervals &LIS, const LiveInVars &LiveIns,
    const MachinePostDominatorTree &PDT,
    std::unordered_set<const MachineBasicBlock *> Visited,
    std::set<std::reference_wrapper<const LiveInterval>> Partition,
    const TargetRegisterInfo *TRI) {
  // TODO: clean this up
  std::vector<CodeSeparator> Separators;
  std::queue<const MachineBasicBlock *> Q;
  Q.push(Start);
  while (!Q.empty()) {
    auto *MBB = Q.front();
    Q.pop();
    if (MBB == End || Visited.find(MBB) != Visited.end()) {
      break;
    }
    const auto BlockSeps = bbSeparators(MBB, LIS, LiveIns, Partition, TRI);
    Separators.insert(Separators.end(), BlockSeps.begin(), BlockSeps.end());
    Visited.insert(MBB);
    if (MBB->succ_empty()) {
      assert(Q.empty());
      break;
    }
    std::vector<MachineBasicBlock *> Succs;
    for (auto *Succ : MBB->successors()) {
      Succs.push_back(Succ);
    }
    // merge point of all successors
    const auto *NextBlock = PDT.findNearestCommonDominator(Succs);
    if (NextBlock != nullptr) {
      // vector of pathsep[pathidx][sepidx]
      std::vector<std::vector<CodeSeparator>> PathSeps;
      // vector of the values since the last separator for each path
      std::vector<std::set<std::reference_wrapper<const LiveInterval>>>
          PathPartitions;
      auto MinPathSeps = std::numeric_limits<size_t>::max();
      for (auto *Succ : MBB->successors()) {
        // SSA, so there can't be a live interval that "wraps" around via a loop
        auto [Seps, Part] = getSeparatorsAlongPath(
            Succ, NextBlock, LIS, LiveIns, PDT, Visited, Partition, TRI);
        if (Seps.size() < MinPathSeps) {
          MinPathSeps = Seps.size();
        }
        PathSeps.push_back(std::move(Seps));
        PathPartitions.push_back(std::move(Part));
      }
      if (MinPathSeps != std::numeric_limits<size_t>::max()) {
        // clear partition bc we are going to reset it based on the child paths
        Partition.clear();
        for (size_t SepIdx = 0; SepIdx < MinPathSeps; ++SepIdx) {
          auto Sep = PathSeps[0][SepIdx];
          for (size_t PathIdx = 1; PathIdx < PathSeps.size(); ++PathIdx) {
            Sep.merge(PathSeps[PathIdx][SepIdx]);
            LLVM_DEBUG(
                dbgs() << "Meging [ ";
                for (auto &C
                     : PathSeps[PathIdx][SepIdx].Clique) {
                  dbgs() << printReg(C.get().reg(), TRI) << ", ";
                } dbgs()
                << "] into [ ";
                for (auto &C
                     : Sep.Clique) {
                  dbgs() << printReg(C.get().reg(), TRI) << ", ";
                } dbgs()
                << "]\n";);
          }
          Separators.emplace_back(std::move(Sep));
        }
        // add the values since the last separator of each path to the current
        // partition
        for (auto &Path : PathSeps) {
          for (size_t SepIdx = MinPathSeps; SepIdx < Path.size(); ++SepIdx) {
            Partition.insert(Path[SepIdx].Partition.begin(),
                             Path[SepIdx].Partition.end());
          }
        }
        for (auto &P : PathPartitions) {
          Partition.insert(P.begin(), P.end());
        }
      }
      Q.push(NextBlock);
    } else {
      // one of the paths leads to an end
      // this is probably ok because they diverge, the whole merging this is
      // to prevent one branch from having more separators than the other
      assert(MBB->succ_size() == 2 && "Early return has unexpected succ size");
      // TODO: since the successors diverge, they don't interfere with eachother
      // so we could create more partitions instead of aligning them

      std::array<
          std::pair<std::vector<CodeSeparator>,
                    std::set<std::reference_wrapper<const LiveInterval>>>,
          2>
          PathSeps;
      int Idx = 0;
      for (auto *Succ : MBB->successors()) {
        PathSeps[Idx++] = getSeparatorsAlongPath(Succ, nullptr, LIS, LiveIns,
                                                 PDT, Visited, Partition, TRI);
      }
      Partition.clear();
      const auto NumSharedSeps =
          std::min(PathSeps[0].first.size(), PathSeps[1].first.size());
      const auto MinPathIdx = PathSeps[0].first.size() == NumSharedSeps ? 0 : 1;
      const auto MaxPathIdx = 1 - MinPathIdx;
      // merge the shared separators
      for (size_t SepIdx = 0; SepIdx < NumSharedSeps; ++SepIdx) {
        auto &Sep = PathSeps[0].first[SepIdx];
        LLVM_DEBUG(
            dbgs() << "Meging [ "; for (auto &C
                                        : PathSeps[1].first[SepIdx].Clique) {
              dbgs() << printReg(C.get().reg(), TRI) << ", ";
            } dbgs() << "] into [ ";
            for (auto &C
                 : Sep.Clique) {
              dbgs() << printReg(C.get().reg(), TRI) << ", ";
            } dbgs()
            << "]\n";);
        Sep.merge(PathSeps[1].first[SepIdx]);
        Separators.emplace_back(std::move(Sep));
      }
      auto &[ShortPath, ShortPathPartition] = PathSeps[MinPathIdx];
      auto &[LongPath, LongPathPartition] = PathSeps[MaxPathIdx];
      if (NumSharedSeps < LongPath.size()) {
        // add last partition of shorter path to the next separator's
        // partition in the longer path
        LongPath[NumSharedSeps].Partition.insert(ShortPathPartition.begin(),
                                                 ShortPathPartition.end());
        // add the reamining separators of the longer path
        for (size_t SepIdx = NumSharedSeps; SepIdx < LongPath.size();
             ++SepIdx) {
          Separators.emplace_back(std::move(LongPath[SepIdx]));
        }
        Partition = std::move(LongPathPartition);
      } else {
        // both paths have the same number of separators
        Partition = std::move(ShortPathPartition);
        Partition.insert(LongPathPartition.begin(), LongPathPartition.end());
      }
    }
  }
  return {Separators, Partition};
}

/// Builds a node of the partition tree by recursively building the left and
/// right children by splitting the list of separators in half.
/// @param Clique the set of registers that are in the clique for the root
/// partition.
/// @param LastPart the set of live intervals that are in the last partition
/// of the parent separator.
/// @param G the interference graph of the function
/// @param Seps the list of separators along the path
/// @param Start the start index of the list of separators, inclusive
/// @param End the end index of the list of separators, exclusive
/// @return the node of the partition tree
std::unique_ptr<PartitionTree>
buildNode(const std::set<std::reference_wrapper<const LiveInterval>> &LastPart,
          const IGraph &G, const std::vector<CodeSeparator> &Seps, size_t Start,
          size_t End) {
  if (Start == End) {
    auto Node = std::make_unique<PartitionTree>();
    std::unordered_set<Register> S;
    for (auto &C : LastPart) {
      S.insert(C.get().reg());
    }
    // include clique from both adjacent separators
    if (Start > 0) {
      for (auto &C : Seps[Start - 1].Clique) {
        S.insert(C.get().reg());
      }
    }
    if (End < Seps.size()) {
      for (auto &C : Seps[End].Clique) {
        S.insert(C.get().reg());
      }
    }
    Node->Regs = inducedSubgraph(G, S);
    return Node;
  }
  assert(Start < End);
  const auto Mdpt = (Start + End) / 2;
  auto Node = std::make_unique<PartitionTree>();
  for (auto &C : Seps[Mdpt].Clique) {
    Node->SeparatingClique.insert(C.get().reg());
  }
  Node->Left = buildNode(Seps[Mdpt].Partition, G, Seps, Start, Mdpt);
  Node->Right = buildNode(LastPart, G, Seps, std::min(Mdpt + 1, End), End);
  return Node;
}
} // namespace

std::unique_ptr<PartitionTree>
llvm::getPartitions(const MachineFunction &MF, const LiveIntervals &LIS,
                    const MachinePostDominatorTree &PDT,
                    const TargetRegisterInfo *TRI, const IGraph &G) {
  const auto LiveIns = liveIns(MF, LIS);
  auto [Seps, LastPartition] = getSeparatorsAlongPath(
      &*MF.begin(), nullptr, LIS, LiveIns, PDT, {}, {}, TRI);
  return buildNode(LastPartition, G, Seps, 0, Seps.size());
}

PartitionTree llvm::buildPartitionTree(const IGraph &G,
                                       const TargetRegisterInfo *TRI) {
  auto GPrime = G;
  LLVM_DEBUG(dbgs() << "***** Elimination Order: *****\n");
  auto [PEO, SeparatorGenerators] = perfectElimOrder(G);
  for (auto &R : PEO) {
    LLVM_DEBUG(dbgs() << printReg(R, TRI) << ", ");
  }
  LLVM_DEBUG(dbgs() << "\n");
  std::unordered_map<Register, size_t> Indices;
  for (size_t Idx = 0; Idx < PEO.size(); ++Idx) {
    Indices.insert(std::make_pair(PEO[Idx], Idx));
  }
  std::vector<IGraph> Atoms;
  std::vector<std::unordered_set<Register>> CliqueSeparators;
  for (unsigned Idx = 0; Idx < PEO.size(); ++Idx) {
    auto &V = PEO[Idx];
    // if (SeparatorGenerators.find(V) == SeparatorGenerators.end()) {
    //   continue;
    // ?
    // }
    // neighbors of V st vertices appear after V in the elimination order
    std::unordered_set<Register> Clique;
    for (auto U : G.at(V)) {
      if (Indices[U] > Idx) {
        Clique.insert(U);
      }
    }
    if (isCompleteSubgraph(G, Clique)) {
      // A in Tarjan's paper
      const auto Component = connectedComponent(GPrime, Clique, V);
      // VertexSet is A \cup C(v) in Tarjan's paper
      auto VertexSet = Component;
      for (auto &N : Clique) {
        VertexSet.insert(N);
      }
      if (VertexSet.size() < GPrime.size()) {
        Atoms.emplace_back(inducedSubgraph(GPrime, VertexSet));
        CliqueSeparators.emplace_back(std::move(Clique));
        GPrime = GPrime - Component;
      }
    }
  }
  if (!GPrime.empty()) {
    Atoms.emplace_back(GPrime);
    CliqueSeparators.emplace_back(std::unordered_set<Register>());
  }
  return partitionTreeFromAtoms(std::move(Atoms), CliqueSeparators);
}

// an attempt at the approach from the chordal graph minimal separator paper
PartitionTree /*llvm:: */ buildPartitionTree(const IGraph &G,
                                             const TargetRegisterInfo *TRI) {
  // requires G is a chordal graph
  std::vector<std::pair<Register, int>> Heap;
  std::unordered_map<Register, int> Weights;
  // numbered vertices
  std::unordered_set<Register> VNum;
  for (auto [V, _] : G) {
    Heap.push_back(std::make_pair(V, 0));
    Weights.insert(std::make_pair(V, 0));
  }
  // the weight of the last vertex in the order
  std::optional<int> LastWeight;
  // set of maximal cliques
  std::vector<IGraph> Atoms;
  // set of minimal separators
  std::vector<std::unordered_set<Register>> CliqueSeparators;
  // the previous register taken from the heap
  std::optional<Register> LastReg;
  std::stack<std::pair<Register, int>> DebugStack;
  while (!Heap.empty()) {
    std::pop_heap(Heap.begin(), Heap.end());
    const auto [V, W] = Heap.back();
    Heap.pop_back();
    if (VNum.find(V) != VNum.end()) {
      continue;
    }
    VNum.insert(V);
    DebugStack.push(std::make_pair(V, W));
    // we are inserting vertices in decreading elimination order
    if (LastWeight && W <= *LastWeight) {
      // V is a min separator generator and the next vertex is a maximal
      // clique generator
      const auto CS = neighbors(G, VNum, V);
      CliqueSeparators.push_back(CS);
      // bc we have a peo, LatReg and its neighbors that are in VNum form a
      // clique. LastReg is the vertex in the elimination ordering AFTER V.
      // Ie. if V = x_i then LastReg is x_{i+1}
      auto A = neighbors(G, VNum, *LastReg, true);
      for (auto N : CS) {
        A.insert(N);
      }
      Atoms.emplace_back(inducedSubgraph(G, A));
    }
    LastReg = V;
    LastWeight = W;
    for (auto N : G.at(V)) {
      if (VNum.find(N) != VNum.end()) {
        continue;
      }
      Weights[N] += 1;
      Heap.push_back(std::make_pair(N, Weights[N]));
      std::push_heap(Heap.begin(), Heap.end());
    }
  }
  std::unordered_set<Register> IncludedVerts;
  for (auto &A : Atoms) {
    for (auto &[V, _] : A) {
      IncludedVerts.insert(V);
    }
  }

  if (IncludedVerts.size() < G.size()) {
    CliqueSeparators.emplace_back();
    // Atoms.emplace_back(inducedSubgraph(G, neighbors(G, VNum, *LastReg,
    // true)));
    Atoms.emplace_back(G - IncludedVerts);
  }
  DebugStack.push(std::make_pair(*LastReg, *LastWeight));
  LLVM_DEBUG(
      dbgs() << "***** Elimination Order: *****\n"; while (
                                                        !DebugStack.empty()) {
        auto P = DebugStack.top();
        DebugStack.pop();
        dbgs() << "(" << printReg(P.first, TRI) << ", " << P.second << "), ";
      } dbgs() << "\n");
  return partitionTreeFromAtoms(std::move(Atoms), CliqueSeparators);
}

void llvm::debugInterferenceGraph(const IGraph &G,
                                  const TargetRegisterInfo *TRI,
                                  const std::string &FuncName,
                                  const std::string &FileName) {
  if (FileName.find(".") == std::string::npos) {
    return;
  }
  std::string OutTxt;
  llvm::raw_string_ostream Out(OutTxt);
  Out << "graph Interference {\n";
  Out << "\tnode [fontname=\"Helvetica\"];\n";
  Out << "\tedge [fontname=\"Helvetica\"];\n";
  Out << "\tlayout=fdp;\n";
  std::set<std::pair<Register, Register>> ProcessedEdges;
  for (auto &[V, _] : G) {
    Out << "\tn" << V.id() << " [label=\"" << printReg(V, TRI) << "\"];\n";
  }
  for (auto &[V, Neighbors] : G) {
    for (auto U : Neighbors) {
      if (ProcessedEdges.find(std::make_pair(U, V)) != ProcessedEdges.end()) {
        continue;
      }
      Out << "\tn" << V.id() << " -- n" << U.id() << ";\n";
      ProcessedEdges.insert(std::make_pair(V, U));
      ProcessedEdges.insert(std::make_pair(U, V));
    }
  }
  Out << "}\n";
  Out.flush();
  std::ofstream OutFile(FuncName + "_" + FileName);
  OutFile << OutTxt << std::endl;
}

void llvm::debugPartitionTree(const PartitionTree &T,
                              const TargetRegisterInfo *TRI,
                              const std::string &FuncName,
                              const std::string &FileName) {
  if (FileName.find(".") == std::string::npos) {
    return;
  }
  std::string OutTxt;
  llvm::raw_string_ostream Out(OutTxt);
  Out << "digraph PartitionTree {\n";
  Out << "\tnode [fontname=\"Helvetica\"];\n";
  Out << "\tedge [fontname=\"Helvetica\"];\n";
  Out << "\tlayout=dot;\n";
  Out << "\trankdir=\"TB\";\n";
  std::vector<std::pair<const PartitionTree *, int>> Q;
  Q.push_back(std::make_pair(&T, 0));
  int GlobalId = 0;
  while (!Q.empty()) {
    auto [Node, Id] = Q.back();
    Q.pop_back();
    Out << "\tn" << Id << " [label=";
    if (Node->Left && Node->Right) {
      assert(Node->Regs.empty());
      Out << "\"{";
      unsigned Cnt = 0;
      for (auto N : Node->SeparatingClique) {
        Out << printReg(N, TRI);
        if (++Cnt < Node->SeparatingClique.size()) {
          Out << ", ";
        }
      }
      Out << "}\"];\n";
      const auto LId = ++GlobalId;
      const auto RId = ++GlobalId;
      Out << "\tn" << Id << " -> n" << LId << ";\n";
      Out << "\tn" << Id << " -> n" << RId << ";\n";
      Q.emplace_back(Node->Left.get(), LId);
      Q.emplace_back(Node->Right.get(), RId);
    } else {
      Out << "\"{";
      // NOLINTNEXTLINE(readability-identifier-naming)
      unsigned i = 0;
      for (auto &[R, _] : Node->Regs) {
        Out << printReg(R, TRI);
        if (i++ < Node->Regs.size() - 1) {
          Out << ", ";
        }
      }
      Out << "}\"];\n";
    }
    ++Id;
  }
  Out << "}\n";
  Out.flush();
  std::ofstream OutFile(FuncName + "_" + FileName);
  OutFile << OutTxt << std::endl;
}

llvm::PRegMap::PRegMap(const IGraph &G, const VirtRegMap &VRM,
                       const RegisterClassInfo &RegClassInfo,
                       LiveRegMatrix *Matrix, const LiveIntervals *LIS) {
  for (auto &[VReg, _] : G) {
    auto Order = AllocationOrder::create(VReg, VRM, RegClassInfo, Matrix);
    std::vector<MCRegister> CorrectedOrder;
    for (auto PReg : Order) {
      if (Matrix->checkInterference(LIS->getInterval(VReg), PReg) ==
          LiveRegMatrix::IK_Free) {
        CorrectedOrder.push_back(PReg);
      }
    }
    OrderMap[VReg] = CorrectedOrder;
    SetMap[VReg] = std::unordered_set<MCRegister>(CorrectedOrder.begin(),
                                                  CorrectedOrder.end());
  }
}

bool llvm::PRegMap::isValidAssignment(Register VReg, MCRegister PReg) const {
  return SetMap.at(VReg).find(PReg) != SetMap.at(VReg).end();
}

const std::vector<MCRegister> &
llvm::PRegMap::getAllocationOrder(Register VReg) const {
  return OrderMap.at(VReg);
}

std::shared_ptr<Color> llvm::Color::getRootHelper() const {
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

Color *llvm::Color::getRootMut() {
  return std::visit(
      [this](auto &&V) {
        using T = std::decay_t<decltype(V)>;
        if constexpr (std::is_same_v<ColorClass, T>) {
          return this;
        } else {
          return V->getRootMut();
        }
      },
      Value);
}

void llvm::Color::addMembers(const std::unordered_set<Register> &Members) {
  auto *Root = getRootMut();
  auto &Class = std::get<ColorClass>(Root->Value);
  Class.Members.insert(Members.begin(), Members.end());
}

MCRegister llvm::Color::getPReg() {
  if (std::holds_alternative<ColorClass>(Value)) {
    return std::get<ColorClass>(Value).PReg;
  }
  Value = getRootHelper();
  auto Ptr = std::get<std::shared_ptr<Color>>(Value);
  assert(Ptr != nullptr);
  return Ptr->getPReg();
}

MCRegister llvm::Color::getPReg() const {
  if (std::holds_alternative<ColorClass>(Value)) {
    return std::get<ColorClass>(Value).PReg;
  }
  return std::get<std::shared_ptr<Color>>(Value)->getPReg();
}

void llvm::Color::setColor(const std::shared_ptr<Color> &C) {
  auto *Root = getRootMut();
  if (Root != C->getRootMut()) {
    auto &CurClass = std::get<ColorClass>(Root->Value);
    C->addMembers(CurClass.Members);
    Root->Value = C;
    Value = C;
  }
}

void llvm::Color::setColor(std::shared_ptr<Color> &&C) { setColor(C); }

const Color *Color::getRoot() const {
  return std::visit(
      [this](auto &&V) -> const Color * {
        using T = std::decay_t<decltype(V)>;
        if constexpr (std::is_same_v<ColorClass, T>) {
          return this;
        } else {
          return V->getRoot();
        }
      },
      Value);
}

std::shared_ptr<Color> Color::getRootPtr(const std::shared_ptr<Color> &C) {
  return std::visit(
      [C](auto &&V) -> std::shared_ptr<Color> {
        using T = std::decay_t<decltype(V)>;
        if constexpr (std::is_same_v<ColorClass, T>) {
          return C;
        } else {
          return V->getRootPtr(C);
        }
      },
      C->Value);
}

raw_ostream &ColoringResult::print(raw_ostream &OS,
                                   const TargetRegisterInfo *TRI) const {
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