/*
 * Implementation of IR utility functions
 */

#include "split/ir_utils.h"

#include <stdexcept>
#include <string>

namespace middleware {
namespace ir_utils {

std::unique_ptr<ir_sql_converter::AQPExpr>
CloneExpr(const ir_sql_converter::AQPExpr *expr) {
  if (!expr)
    return nullptr;

  auto node_type = expr->GetNodeType();

  if (node_type == ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarConstComparison *>(
            expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestVarConstComparison>(
          e->GetSimplestExprType(), CloneAttr(e->attr),
          std::make_unique<ir_sql_converter::SimplestConstVar>(*e->const_var));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarComparison *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestVarComparison>(
          e->GetSimplestExprType(), CloneAttr(e->left_attr),
          CloneAttr(e->right_attr));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::IsNullExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestIsNullExpr>(
          e->GetSimplestExprType(), CloneAttr(e->attr));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarParamComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarParamComparison *>(
            expr);
    if (e && e->param_var) {
      auto cloned_param = std::make_unique<ir_sql_converter::SimplestParam>(
          e->param_var->GetType(), e->param_var->GetParamId());
      return std::make_unique<ir_sql_converter::SimplestVarParamComparison>(
          e->GetSimplestExprType(), CloneAttr(e->attr),
          std::move(cloned_param));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestLogicalExpr>(
          e->GetLogicalOp(), CloneExpr(e->left_expr.get()),
          CloneExpr(e->right_expr.get()));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::SingleAttrExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestSingleAttrExpr *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestSingleAttrExpr>(
          CloneAttr(e->attr));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::InExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    if (e && e->attr) {
      std::vector<std::unique_ptr<ir_sql_converter::SimplestConstVar>> vals;
      for (const auto &v : e->values)
        vals.push_back(
            std::make_unique<ir_sql_converter::SimplestConstVar>(*v));
      return std::make_unique<ir_sql_converter::SimplestInExpr>(
          CloneAttr(e->attr), std::move(vals), e->negated);
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ArithExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestArithExpr *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestArithExpr>(
          e->arith_op, CloneExpr(e->left.get()), CloneExpr(e->right.get()),
          e->result_type);
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::CastExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestCastExpr *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestCastExpr>(
          CloneExpr(e->child.get()), e->target_type);
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestGeneralComparison *>(expr);
    if (e) {
      return std::make_unique<ir_sql_converter::SimplestGeneralComparison>(
          e->GetSimplestExprType(), CloneExpr(e->left_expr.get()),
          CloneExpr(e->right_expr.get()));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ConstVarNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestConstExpr *>(expr);
    if (e && e->value) {
      return std::make_unique<ir_sql_converter::SimplestConstExpr>(
          std::make_unique<ir_sql_converter::SimplestConstVar>(*e->value));
    }
  }

  if (node_type == ir_sql_converter::SimplestNodeType::FunctionExprNodeType) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestFunctionExpr *>(expr);
    if (e) {
      std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> cloned_args;
      for (const auto &arg : e->args)
        cloned_args.push_back(CloneExpr(arg.get()));
      return std::make_unique<ir_sql_converter::SimplestFunctionExpr>(
          e->fn_name, std::move(cloned_args));
    }
  }

  throw std::runtime_error("ir_utils::CloneExpr unsupported: expression node "
                           "type " +
                           std::to_string(static_cast<int>(node_type)));
}

bool ExprInvolvesOnlyTables(const ir_sql_converter::AQPExpr *expr,
                            const std::set<unsigned int> &tables) {
  if (!expr)
    return true;

  auto node_type = expr->GetNodeType();

  if (node_type == ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarConstComparison *>(
            expr);
    return e && tables.count(e->attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarComparison *>(expr);
    if (!e)
      return false;
    return tables.count(e->left_attr->GetTableIndex()) > 0 &&
           tables.count(e->right_attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::IsNullExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    return e && tables.count(e->attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarParamComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarParamComparison *>(
            expr);
    return e && tables.count(e->attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (!e)
      return false;
    return ExprInvolvesOnlyTables(e->left_expr.get(), tables) &&
           ExprInvolvesOnlyTables(e->right_expr.get(), tables);
  }

  if (node_type == ir_sql_converter::SimplestNodeType::SingleAttrExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestSingleAttrExpr *>(expr);
    return e && tables.count(e->attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::InExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    return e && e->attr && tables.count(e->attr->GetTableIndex()) > 0;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ArithExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestArithExpr *>(expr);
    if (!e)
      return false;
    return ExprInvolvesOnlyTables(e->left.get(), tables) &&
           ExprInvolvesOnlyTables(e->right.get(), tables);
  }

  if (node_type == ir_sql_converter::SimplestNodeType::CastExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestCastExpr *>(expr);
    if (!e)
      return false;
    return ExprInvolvesOnlyTables(e->child.get(), tables);
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestGeneralComparison *>(expr);
    if (!e)
      return false;
    return ExprInvolvesOnlyTables(e->left_expr.get(), tables) &&
           ExprInvolvesOnlyTables(e->right_expr.get(), tables);
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ConstVarNode) {
    return true;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::FunctionExprNodeType) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestFunctionExpr *>(expr);
    if (!e)
      return false;
    for (const auto &arg : e->args) {
      if (!ExprInvolvesOnlyTables(arg.get(), tables))
        return false;
    }
    return true;
  }

  return false;
}

void CollectAttrsFromExpr(
    const ir_sql_converter::AQPExpr *expr,
    const std::set<unsigned int> &target_tables,
    std::set<std::pair<unsigned int, unsigned int>> &seen,
    std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> &attrs) {

  if (!expr)
    return;

  auto addAttr = [&](const ir_sql_converter::SimplestAttr *attr) {
    if (!attr)
      return;
    if (target_tables.count(attr->GetTableIndex()) == 0)
      return;
    auto key = std::make_pair(attr->GetTableIndex(), attr->GetColumnIndex());
    if (seen.find(key) == seen.end()) {
      seen.insert(key);
      attrs.push_back(CloneAttr(attr));
    }
  };

  auto node_type = expr->GetNodeType();

  if (node_type == ir_sql_converter::SimplestNodeType::VarConstComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarConstComparison *>(
            expr);
    if (e)
      addAttr(e->attr.get());
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarComparison *>(expr);
    if (e) {
      addAttr(e->left_attr.get());
      addAttr(e->right_attr.get());
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::IsNullExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    if (e)
      addAttr(e->attr.get());
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::VarParamComparisonNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestVarParamComparison *>(
            expr);
    if (e)
      addAttr(e->attr.get());
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (e) {
      CollectAttrsFromExpr(e->left_expr.get(), target_tables, seen, attrs);
      CollectAttrsFromExpr(e->right_expr.get(), target_tables, seen, attrs);
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::SingleAttrExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestSingleAttrExpr *>(expr);
    if (e)
      addAttr(e->attr.get());
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::InExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    if (e)
      addAttr(e->attr.get());
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ArithExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestArithExpr *>(expr);
    if (e) {
      CollectAttrsFromExpr(e->left.get(), target_tables, seen, attrs);
      CollectAttrsFromExpr(e->right.get(), target_tables, seen, attrs);
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::CastExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestCastExpr *>(expr);
    if (e)
      CollectAttrsFromExpr(e->child.get(), target_tables, seen, attrs);
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ExprNode) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestGeneralComparison *>(expr);
    if (e) {
      CollectAttrsFromExpr(e->left_expr.get(), target_tables, seen, attrs);
      CollectAttrsFromExpr(e->right_expr.get(), target_tables, seen, attrs);
    }
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::ConstVarNode) {
    // SimplestConstExpr: no attrs to collect
    return;
  }

  if (node_type == ir_sql_converter::SimplestNodeType::FunctionExprNodeType) {
    auto *e =
        dynamic_cast<const ir_sql_converter::SimplestFunctionExpr *>(expr);
    if (e) {
      for (const auto &arg : e->args)
        CollectAttrsFromExpr(arg.get(), target_tables, seen, attrs);
    }
    return;
  }

  throw std::runtime_error(
      "ir_utils::CollectAttrsFromExpr unsupported: expression node type " +
      std::to_string(static_cast<int>(node_type)) +
      "; required columns would be missed");
}

// ===== AND-Splitting for Filter Conditions =====

std::unique_ptr<ir_sql_converter::AQPExpr>
ExtractConjunctsForTables(const ir_sql_converter::AQPExpr *expr,
                          const std::set<unsigned int> &tables) {
  if (!expr)
    return nullptr;

  auto node_type = expr->GetNodeType();

  // Handle LogicalExpr (AND/OR)
  if (node_type == ir_sql_converter::SimplestNodeType::LogicalExprNode) {
    auto *e = dynamic_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    if (!e)
      return nullptr;

    if (e->GetLogicalOp() == ir_sql_converter::SimplestLogicalOp::LogicalAnd) {
      // AND: recursively extract from both sides, combine valid parts
      auto left_extracted =
          ExtractConjunctsForTables(e->left_expr.get(), tables);
      auto right_extracted =
          ExtractConjunctsForTables(e->right_expr.get(), tables);

      if (left_extracted && right_extracted) {
        // Both sides valid → combine with AND
        return std::make_unique<ir_sql_converter::SimplestLogicalExpr>(
            ir_sql_converter::SimplestLogicalOp::LogicalAnd,
            std::move(left_extracted), std::move(right_extracted));
      } else if (left_extracted) {
        // Only left valid → return left
        return left_extracted;
      } else if (right_extracted) {
        // Only right valid → return right
        return right_extracted;
      } else {
        // Neither valid → return nullptr
        return nullptr;
      }
    } else if (e->GetLogicalOp() ==
               ir_sql_converter::SimplestLogicalOp::LogicalOr) {
      // OR: can only include if ALL referenced tables are in the set
      // (can't split OR - must keep entire clause together)
      if (ExprInvolvesOnlyTables(expr, tables)) {
        return CloneExpr(expr);
      }
      return nullptr;
    }
  }

  // For leaf expressions: include if all referenced tables are in the set
  if (ExprInvolvesOnlyTables(expr, tables)) {
    return CloneExpr(expr);
  }

  return nullptr;
}

// ===== Collection Functions =====

std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>>
CollectFilterConditions(ir_sql_converter::AQPStmt *ir,
                        const std::set<unsigned int> &tables) {
  std::vector<std::unique_ptr<ir_sql_converter::AQPExpr>> filters;

  if (!ir)
    return filters;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::SetOpNode)
    return filters;

  // Check qual_vec for filter conditions in this node
  for (const auto &qual : ir->qual_vec) {
    // Use AND-splitting: extract conjuncts that involve only cluster tables
    auto extracted = ExtractConjunctsForTables(qual.get(), tables);
    if (extracted) {
      filters.push_back(std::move(extracted));
    }
  }

  // Recurse into children
  for (auto &child : ir->children) {
    auto child_filters = CollectFilterConditions(child.get(), tables);
    for (auto &f : child_filters) {
      filters.push_back(std::move(f));
    }
  }

  return filters;
}

std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
CollectJoinConditions(ir_sql_converter::AQPStmt *ir,
                      const std::set<unsigned int> &tables) {
  std::vector<std::unique_ptr<ir_sql_converter::SimplestVarComparison>>
      conditions;

  if (!ir)
    return conditions;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::SetOpNode)
    return conditions;

  // If this is a Join node, check its join conditions
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode) {
    auto *join = dynamic_cast<ir_sql_converter::SimplestJoin *>(ir);
    if (join) {
      for (const auto &cond : join->join_conditions) {
        unsigned int left_table = cond->left_attr->GetTableIndex();
        unsigned int right_table = cond->right_attr->GetTableIndex();
#ifndef NDEBUG
        std::cout << "[ir_utils::CollectJoinConditions] cond "
                  << left_table << "." << cond->left_attr->GetColumnName()
                  << " vs " << right_table << "."
                  << cond->right_attr->GetColumnName()
                  << " in_cluster: " << tables.count(left_table)
                  << "/" << tables.count(right_table) << std::endl;
#endif
        // Include condition if BOTH tables are in our cluster
        if (tables.count(left_table) && tables.count(right_table)) {
          auto cloned = CloneVarComparison(cond.get());
          if (cloned) {
            conditions.push_back(std::move(cloned));
          }
        }
      }
    }
  }

  // Recurse into children
  for (auto &child : ir->children) {
    auto child_conds = CollectJoinConditions(child.get(), tables);
    for (auto &c : child_conds) {
      conditions.push_back(std::move(c));
    }
  }

  return conditions;
}

std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>>
CollectFilterAttrs(ir_sql_converter::AQPStmt *ir,
                   const std::set<unsigned int> &tables) {
  std::vector<std::unique_ptr<ir_sql_converter::SimplestAttr>> attrs;
  std::set<std::pair<unsigned int, unsigned int>> seen;

  if (!ir)
    return attrs;

  // Collect from qual_vec in this node
  for (const auto &qual : ir->qual_vec) {
    CollectAttrsFromExpr(qual.get(), tables, seen, attrs);
  }

  // Recurse into children
  for (auto &child : ir->children) {
    auto child_attrs = CollectFilterAttrs(child.get(), tables);
    for (auto &attr : child_attrs) {
      auto key = std::make_pair(attr->GetTableIndex(), attr->GetColumnIndex());
      if (seen.find(key) == seen.end()) {
        seen.insert(key);
        attrs.push_back(std::move(attr));
      }
    }
  }

  return attrs;
}

// ===== IR Tree Cloning =====

std::unique_ptr<ir_sql_converter::AQPStmt>
CloneIR(const ir_sql_converter::AQPStmt *node) {
  using namespace ir_sql_converter;
  if (!node)
    return nullptr;

  std::vector<std::unique_ptr<AQPStmt>> children;
  for (const auto &c : node->children) {
    auto cc = CloneIR(c.get());
    if (c && !cc)
      return nullptr;
    children.push_back(std::move(cc));
  }

  std::vector<std::unique_ptr<SimplestAttr>> targets;
  for (const auto &a : node->target_list)
    targets.push_back(std::make_unique<SimplestAttr>(*a));

  std::vector<std::unique_ptr<AQPExpr>> quals;
  for (const auto &q : node->qual_vec) {
    auto cq = CloneExpr(q);
    if (q && !cq)
      return nullptr;
    quals.push_back(std::move(cq));
  }

  auto base = std::make_unique<AQPStmt>(std::move(children), std::move(targets),
                                        std::move(quals), node->GetNodeType());
  if (!node->expr_target_list.empty()) {
    for (const auto &e : node->expr_target_list)
      base->expr_target_list.push_back(e ? CloneExpr(e.get()) : nullptr);
  }

  std::unique_ptr<AQPStmt> out;
  switch (node->GetNodeType()) {
  case SimplestNodeType::ProjectionNode: {
    auto *p = dynamic_cast<const SimplestProjection *>(node);
    if (!p)
      return nullptr;
    out = std::make_unique<SimplestProjection>(std::move(base), p->GetIndex());
    break;
  }
  case SimplestNodeType::JoinNode: {
    auto *j = dynamic_cast<const SimplestJoin *>(node);
    if (!j)
      return nullptr;
    std::vector<std::unique_ptr<SimplestVarComparison>> conds;
    for (const auto &cond : j->join_conditions) {
      if (!cond || !cond->left_attr || !cond->right_attr)
        return nullptr;
      conds.push_back(std::make_unique<SimplestVarComparison>(
          cond->GetSimplestExprType(),
          std::make_unique<SimplestAttr>(*cond->left_attr),
          std::make_unique<SimplestAttr>(*cond->right_attr)));
    }
    auto join = std::make_unique<SimplestJoin>(std::move(base), std::move(conds),
                                              j->GetSimplestJoinType());
    join->SetMarkIndex(j->GetMarkIndex());
    join->SetBuildChild(j->GetBuildChild());
    out = std::move(join);
    break;
  }
  case SimplestNodeType::FilterNode:
    out = std::make_unique<SimplestFilter>(std::move(base));
    break;
  case SimplestNodeType::ScanNode: {
    auto *s = dynamic_cast<const SimplestScan *>(node);
    if (!s)
      return nullptr;
    out = std::make_unique<SimplestScan>(std::move(base), s->GetTableIndex(),
                                        s->GetTableName());
    break;
  }
  case SimplestNodeType::ChunkNode: {
    auto *c = dynamic_cast<const SimplestChunk *>(node);
    if (!c)
      return nullptr;
    out = std::make_unique<SimplestChunk>(std::move(base), c->GetTableIndex(),
                                         c->GetChunkName(), c->GetContents());
    break;
  }
  case SimplestNodeType::CrossProductNode:
    out = std::make_unique<SimplestCrossProduct>(std::move(base));
    break;
  case SimplestNodeType::AggregateNode: {
    auto *a = dynamic_cast<const SimplestAggregate *>(node);
    if (!a)
      return nullptr;
    agg_fn_pair cloned_fns;
    for (const auto &fn : a->agg_fns)
      cloned_fns.emplace_back(
          fn.first ? std::make_unique<SimplestAttr>(*fn.first) : nullptr,
          fn.second);
    std::vector<std::unique_ptr<SimplestAttr>> cloned_groups;
    for (const auto &g : a->groups)
      cloned_groups.push_back(std::make_unique<SimplestAttr>(*g));
    auto agg = std::make_unique<SimplestAggregate>(
        std::move(base), std::move(cloned_fns), std::move(cloned_groups),
        a->GetAggIndex(), a->GetGroupIndex());
    for (const auto &e : a->agg_fn_exprs)
      agg->agg_fn_exprs.push_back(e ? CloneExpr(e.get()) : nullptr);
    agg->agg_distinct = a->agg_distinct;
    for (const auto &e : a->group_exprs)
      agg->group_exprs.push_back(e ? CloneExpr(e.get()) : nullptr);
    agg->grouping_sets = a->grouping_sets;
    out = std::move(agg);
    break;
  }
  case SimplestNodeType::OrderNode: {
    auto *o = dynamic_cast<const SimplestOrderBy *>(node);
    if (!o)
      return nullptr;
    std::vector<OrderStruct> cloned_orders;
    for (const auto &ord : o->orders)
      cloned_orders.push_back(
          {ord.order_type,
           ord.attr ? std::make_unique<SimplestAttr>(*ord.attr) : nullptr});
    auto ob = std::make_unique<SimplestOrderBy>(std::move(base),
                                                std::move(cloned_orders));
    ob->projections = o->projections;
    out = std::move(ob);
    break;
  }
  case SimplestNodeType::LimitNode: {
    auto *l = dynamic_cast<const SimplestLimit *>(node);
    if (!l)
      return nullptr;
    out = std::make_unique<SimplestLimit>(std::move(base), l->limit_val,
                                         l->offset_val);
    break;
  }
  case SimplestNodeType::DistinctNode:
    out = std::make_unique<SimplestDistinct>(std::move(base));
    break;
  case SimplestNodeType::SetOpNode: {
    auto *s = dynamic_cast<const SimplestSetOp *>(node);
    if (!s)
      return nullptr;
    // SetOp stores children in the base; already cloned above.
    // Reconstruct with the right metadata.
    auto setop = std::make_unique<AQPStmt>(
        std::move(base->children), std::move(base->target_list), SetOpNode);
    // Can't use the 4-arg SetOp ctor easily; set fields manually via cast.
    // Return nullptr: SetOp in remaining IR is rare; skip speculation.
    return nullptr;
  }
  default:
    return nullptr;
  }
  out->SetEstimatedCardinality(node->GetEstimatedCardinality());
  return out;
}

} // namespace ir_utils
} // namespace middleware
