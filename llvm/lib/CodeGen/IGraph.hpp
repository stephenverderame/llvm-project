#pragma once
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCRegister.h"
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace std {
template <> struct hash<llvm::Register> {
  std::size_t operator()(const llvm::Register &R) const {
    return std::hash<unsigned>()(R.id());
  }
};

template <> struct hash<llvm::MCRegister> {
  std::size_t operator()(const llvm::MCRegister &R) const {
    return std::hash<unsigned>()(R.id());
  }
};
} // namespace std
namespace llvm {
using IGraph = std::unordered_map<Register, std::unordered_set<Register>>;

/// A partition tree is a binary tree that represents a partition of a set of
/// registers. Each leaf node contains a set of registers, and each internal
/// node contains two children, a separating clique, and no registers.
struct PartitionTree {
  /// Invariant: Either has both children and no regs or is a leaf and has regs.
  std::unique_ptr<PartitionTree> Left, Right;
  std::unordered_set<Register> SeparatingClique;
  IGraph Regs;
};

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

/// An interference graph between virtual and physical registers.
/// The graph is bipartite with bipartite sets for virtual and physical
/// registers.
class PRegMap {
  std::unordered_map<Register, std::vector<MCRegister>> OrderMap;
  std::unordered_map<Register, std::unordered_set<MCRegister>> SetMap;

public:
  /// For each register in the interference graph, returns a list of physical
  /// registers that could potentially be assigned to it, taking into
  /// account precolored register interference. The order of the physical
  /// registers is the order in which they are preferred.
  PRegMap(const IGraph &G, const VirtRegMap &VRM,
          const RegisterClassInfo &RegClassInfo, LiveRegMatrix *Matrix,
          const LiveIntervals *LIS);

  /// Gets the allocation order of physical registers for a given virtual
  /// register, taking into account precolored register interference.
  const std::vector<MCRegister> &getAllocationOrder(Register VReg) const;

  /// Returns true if the physical register is a valid assignment for the
  /// virtual register. A valid assignment is a preg that doesn't interfere with
  /// the vreg and has the same "type"
  bool isValidAssignment(Register VReg, MCRegister PReg) const;
};

/// A register color union-find
class Color {
  /// The root node of the union-find data structure.
  struct ColorClass {
    MCRegister PReg;
    std::unordered_set<Register> Members;
  };
  // Physical register or parent
  std::variant<ColorClass, std::shared_ptr<Color>> Value;

private:
  /// Gets the root of the current node in the union-find data structure.
  /// Requires that the current node is not a physical register (not a root)
  std::shared_ptr<Color> getRootHelper() const;

  /// Gets the root color of the equivalence class this node is in
  Color *getRootMut();

  /// Adds the members to the color class this node represents
  void addMembers(const std::unordered_set<Register> &Members);

public:
  /// Gets the physical register for this color
  /// @{
  MCRegister getPReg();
  MCRegister getPReg() const;
  /// @}

  /// Sets the color of all nodes in the class to be the specified color.
  /// Does nothing if the two colors are already the same.
  /// @{
  void setColor(const std::shared_ptr<Color> &C);
  void setColor(std::shared_ptr<Color> &&C);
  /// @}

  Color(std::shared_ptr<Color> Value) : Value(std::move(Value)) {}
  Color(MCRegister PReq, Register VReg) {
    std::unordered_set<Register> Members;
    Members.insert(VReg);
    Value = ColorClass{PReq, Members};
  }
  Color(MCRegister PReq)
      : Value(ColorClass{PReq, std::unordered_set<Register>()}) {}

  /// Gets the root color of the equivalence class this node is in
  const Color *getRoot() const;

  inline const std::unordered_set<Register> &members() const {
    auto *Root = getRoot();
    return std::get<ColorClass>(Root->Value).Members;
  }

  inline std::unordered_set<Register> &members() {
    auto *Root = getRootMut();
    return std::get<ColorClass>(Root->Value).Members;
  }

  /// Gets a shared pointer to the root of the current node in the union-find
  /// data structure.
  static std::shared_ptr<Color> getRootPtr(const std::shared_ptr<Color> &C);
};

/// Result of performing graph coloring on an interference graph.
struct ColoringResult {
  /// Set of physical registers used.
  // std::vector<std::shared_ptr<Color>> Colors;

  /// Map virtual register to physical register by the index in `Colors`.
  /// A register that maps to `nullptr` is spilled.
  std::map<Register, std::shared_ptr<Color>> RegToColor;

  raw_ostream &print(raw_ostream &OS, const TargetRegisterInfo *TRI) const;
};
} // namespace llvm