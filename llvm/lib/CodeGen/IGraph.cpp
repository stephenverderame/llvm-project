#include "IGraph.hpp"
#include "AllocationOrder.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#define DEBUG_TYPE "igraph"

using namespace llvm;
std::vector<Register> llvm::perfectElimOrder(const IGraph &G) {
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
  Order.resize(G.size());
  const auto Comp = [](const std::pair<Register, int> &A,
                       const std::pair<Register, int> &B) {
    return A.second < B.second;
  };
  int N = G.size() - 1;
  while (!Heap.empty()) {
    std::pop_heap(Heap.begin(), Heap.end(), Comp);
    auto [V, W] = Heap.back();
    Heap.pop_back();
    if (Numbered.find(V) != Numbered.end()) {
      continue;
    }
    Numbered.insert(V);
    Order[N--] = V;
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

bool llvm::isCompleteSubgraph(const IGraph &G,
                              const std::set<Register> &Clique) {
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

std::set<Register> llvm::connectedComponent(const IGraph &G,
                                            const std::set<Register> &Exclude,
                                            Register X) {
  std::set<Register> Component;
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

IGraph llvm::inducedSubgraph(const IGraph &G, std::set<Register> &Set) {
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

IGraph llvm::operator-(const IGraph &G, const std::set<Register> &S) {
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
PartitionTree partitionTreeFromAtoms(
    std::vector<IGraph> &&Atoms,
    const std::vector<std::set<Register>> &CliqueSeparators) {
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

PartitionTree llvm::buildPartitionTree(const IGraph &G,
                                       const TargetRegisterInfo *TRI) {
  auto GPrime = G;
  LLVM_DEBUG(dbgs() << "***** Elimination Order: *****\n");
  auto PEO = perfectElimOrder(G);
  for (auto &R : PEO) {
    LLVM_DEBUG(dbgs() << printReg(R, TRI) << ", ");
  }
  LLVM_DEBUG(dbgs() << "\n");
  std::map<Register, size_t> Indices;
  for (size_t Idx = 0; Idx < PEO.size(); ++Idx) {
    Indices.insert(std::make_pair(PEO[Idx], Idx));
  }
  std::vector<IGraph> Atoms;
  std::vector<std::set<Register>> CliqueSeparators;
  for (unsigned Idx = 0; Idx < PEO.size(); ++Idx) {
    auto &V = PEO[Idx];
    std::set<Register> Clique;
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
    CliqueSeparators.emplace_back(std::set<Register>());
  }
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
  Out << "graph Interferance {\n";
  Out << "\tfontname=\"Helvetica\";\n";
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
  Out << "\tfontname=\"Helvetica\";\n";
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
      for (auto N : Node->SeparatingClique) {
        Out << printReg(N, TRI);
        if (N != *Node->SeparatingClique.rbegin()) {
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
    SetMap[VReg] =
        std::set<MCRegister>(CorrectedOrder.begin(), CorrectedOrder.end());
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

void llvm::Color::addMembers(const std::set<Register> &Members) {
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