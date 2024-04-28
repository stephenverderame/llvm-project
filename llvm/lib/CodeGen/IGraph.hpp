#pragma once
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
namespace llvm {
using IGraph = std::map<Register, std::set<Register>>;

/// A partition tree is a binary tree that represents a partition of a set of
/// registers. Each leaf node contains a set of registers, and each internal
/// node contains two children, a separating clique, and no registers.
struct PartitionTree {
  /// Invariant: Either has both children and no regs or is a leaf and has regs.
  std::unique_ptr<PartitionTree> Left, Right;
  std::set<Register> SeparatingClique;
  IGraph Regs;
};
/// Gets a perfect (simplicial) elimination order of the interference graph G.
std::vector<Register> perfectElimOrder(const IGraph &G);

/// Returns true if Clique is a complete subgraph of G.
bool isCompleteSubgraph(const IGraph &G, const std::set<Register> &Clique);

/// Gets the connected component of G containing X, excluding the registers in
/// Exclude.
std::set<Register> connectedComponent(const IGraph &G,
                                      const std::set<Register> &Exclude,
                                      Register X);

/// Returns the induced subgraph of Set in G.
IGraph inducedSubgraph(const IGraph &G, std::set<Register> &Set);

/// Removes the nodes of S from G.
IGraph operator-(const IGraph &G, const std::set<Register> &S);

/// Builds a partition tree from the interference graph G.
/// G is assumed to be chordal, however if it is not, the algorithm will still
/// be correct (probably?), just not optimal.
PartitionTree buildPartitionTree(const IGraph &G,
                                 const TargetRegisterInfo *TRI);

/// Writes the interference to a file if one is supplied
void debugInterferenceGraph(const IGraph &G, const TargetRegisterInfo *TRI,
                            const std::string &FuncName,
                            const std::string &FileName);

/// Writes the partition tree to a file if one is supplied
void debugPartitionTree(const PartitionTree &T, const TargetRegisterInfo *TRI,
                        const std::string &FuncName,
                        const std::string &FileName);
} // namespace llvm