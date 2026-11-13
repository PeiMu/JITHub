/*
 * DuckDB adapter for binding IR to the DuckDB engine
 * */

#include "adapters/duckdb_adapter.h"
#include "util/util.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "duckdb/common/enums/physical_operator_type.hpp"

#ifdef HAVE_LLVM
#include "duckdb/execution/operator/aggregate/physical_hash_aggregate.hpp"
#include "duckdb/execution/operator/aggregate/physical_ungrouped_aggregate.hpp"
#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/join/physical_hash_join.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/operator/scan/physical_table_scan.hpp"
#include "duckdb/execution/operator/scan/physical_column_data_scan.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/table_scan.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/parser/tableref/basetableref.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "jit/aqp_jit_hashtable.h"
#include "qjit/query_jit_steps.h"
#include "simplest_ir.h"
#include "storage/storage_plan.h"
#include "duckdb/common/types/hash.hpp"

// libpg_query's pg_functions.hpp does `#define fprintf(...)`, silently
// deleting every fprintf in this file; restore the real function.
#ifdef fprintf
#undef fprintf
#endif

// Resolve jit_flags to FastCompileBackend for IrToLlvmCompiler
#endif

// Write one JIT timing value (in ms) as a CSV column.
static void WriteJitTimingColumn(long us) {
  std::ofstream log_file;
  log_file.open(g_timing_log_name, std::ios_base::app);
  log_file << std::fixed << std::setprecision(3) << (us / 1000.0) << ", ";
  log_file.close();
}

namespace middleware {

#if IN_MEM_TMP_TABLE
// Anonymous namespace for table function internal state
namespace {

// Bind data: holds pointer to collection and override cardinality
struct TempCollectionFunctionData : public duckdb::FunctionData {
  duckdb::ColumnDataCollection *collection = nullptr;
  bool has_override_cardinality = false;
  uint64_t override_cardinality = 0;
  std::string table_name;

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
    auto result = duckdb::make_uniq<TempCollectionFunctionData>();
    result->collection = collection;
    result->has_override_cardinality = has_override_cardinality;
    result->override_cardinality = override_cardinality;
    result->table_name = table_name;
    return std::move(result);
  }

  bool Equals(const duckdb::FunctionData &other_p) const override {
    auto &other = other_p.Cast<TempCollectionFunctionData>();
    return collection == other.collection;
  }
};

// Global state: holds scan state for ColumnDataCollection
struct TempCollectionGlobalState : public duckdb::GlobalTableFunctionState {
  duckdb::ColumnDataScanState scan_state;
  bool initialized = false;
  duckdb::vector<duckdb::column_t> column_ids;
};

// Bind data for kernel temp table scan
struct KernelTempFunctionData : public duckdb::FunctionData {
  const middleware::storage::FlatTable *flat_table = nullptr;
  std::string table_name;

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
    auto result = duckdb::make_uniq<KernelTempFunctionData>();
    result->flat_table = flat_table;
    result->table_name = table_name;
    return std::move(result);
  }

  bool Equals(const duckdb::FunctionData &other_p) const override {
    auto &other = other_p.Cast<KernelTempFunctionData>();
    return flat_table == other.flat_table;
  }
};

struct KernelTempGlobalState : public duckdb::GlobalTableFunctionState {
  uint64_t current_row = 0;
  duckdb::vector<duckdb::column_t> column_ids;
};

// TableFunctionInfo for kernel temp tables
struct KernelTempScanInfo : public duckdb::TableFunctionInfo {
  explicit KernelTempScanInfo(
      std::unordered_map<std::string, const middleware::storage::FlatTable *> *tables)
      : kernel_temps(tables) {}
  std::unordered_map<std::string, const middleware::storage::FlatTable *> *kernel_temps;
};

#ifdef HAVE_LLVM
// Bind data for qjit temp table scan (scan_qjit_temp)
struct QjitTempFunctionData : public duckdb::FunctionData {
  const qjit::QjitTable *table = nullptr;
  std::string table_name;
  bool has_override_cardinality = false;
  uint64_t override_cardinality = 0;

  duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
    auto result = duckdb::make_uniq<QjitTempFunctionData>();
    result->table = table;
    result->table_name = table_name;
    result->has_override_cardinality = has_override_cardinality;
    result->override_cardinality = override_cardinality;
    return std::move(result);
  }

  bool Equals(const duckdb::FunctionData &other_p) const override {
    auto &other = other_p.Cast<QjitTempFunctionData>();
    return table == other.table;
  }
};

struct QjitTempGlobalState : public duckdb::GlobalTableFunctionState {
  uint64_t current_row = 0;
  duckdb::vector<duckdb::column_t> column_ids;
};

// TableFunctionInfo for qjit temp tables
struct QjitTempScanInfo : public duckdb::TableFunctionInfo {
  QjitTempScanInfo(
      std::unordered_map<std::string, std::unique_ptr<qjit::QjitTable>> *tables,
      std::unordered_map<std::string, middleware::QjitTempMeta> *meta,
      std::shared_timed_mutex *temp_state_mutex)
      : qjit_temps(tables), qjit_meta(meta),
        temp_state_mutex(temp_state_mutex) {}
  std::unordered_map<std::string, std::unique_ptr<qjit::QjitTable>> *qjit_temps;
  std::unordered_map<std::string, middleware::QjitTempMeta> *qjit_meta;
  std::shared_timed_mutex *temp_state_mutex;
};
#endif

} // anonymous namespace
#endif

DuckDBAdapter::DuckDBAdapter(const std::string &db_path) {
  db = std::make_unique<duckdb::DuckDB>(db_path);
  conn = std::make_unique<duckdb::Connection>(*db);
#if IN_MEM_TMP_TABLE
  RegisterTempCollectionScan();
#endif
}

DuckDBAdapter::~DuckDBAdapter() { CleanUp(); }

#if IN_MEM_TMP_TABLE
void DuckDBAdapter::RegisterTempCollectionScan() {
  auto context = GetClientContext();

  // Create the table function
  duckdb::TableFunction func(
      "scan_temp_collection", {duckdb::LogicalType::VARCHAR},
      TempCollectionScanFunc, TempCollectionBind, TempCollectionInitGlobal);
  func.cardinality = TempCollectionCardinality;
  func.projection_pushdown = true;
  func.function_info = duckdb::make_shared_ptr<TempCollectionScanInfo>(
      &temp_collections_, &temp_state_mutex_);

  // Register the table functions in the catalog
  duckdb::CreateTableFunctionInfo info(func);
  auto &catalog = duckdb::Catalog::GetSystemCatalog(*context);
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }
  catalog.CreateTableFunction(*context, info);

  // Register the kernel temp table function
  duckdb::TableFunction kernel_func(
      "scan_kernel_temp", {duckdb::LogicalType::VARCHAR},
      KernelTempScanFunc, KernelTempBind, KernelTempInitGlobal);
  kernel_func.cardinality = KernelTempCardinality;
  kernel_func.projection_pushdown = true;
  kernel_func.function_info =
      duckdb::make_shared_ptr<KernelTempScanInfo>(&kernel_temp_tables_);

  duckdb::CreateTableFunctionInfo kernel_info(kernel_func);
  catalog.CreateTableFunction(*context, kernel_info);

#ifdef HAVE_LLVM
  // Register the qjit temp table function (serves query-jit results
  // directly from the in-memory QjitTable — no CDC copy)
  duckdb::TableFunction qjit_func(
      "scan_qjit_temp", {duckdb::LogicalType::VARCHAR}, QjitTempScanFunc,
      QjitTempBind, QjitTempInitGlobal);
  qjit_func.cardinality = QjitTempCardinality;
  qjit_func.projection_pushdown = true;
  qjit_func.function_info = duckdb::make_shared_ptr<QjitTempScanInfo>(
      &qjit_temps_, &qjit_temp_meta_, &temp_state_mutex_);

  duckdb::CreateTableFunctionInfo qjit_info(qjit_func);
  catalog.CreateTableFunction(*context, qjit_info);
#endif

  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }

  // Register the replacement scan
  auto &db_config = duckdb::DBConfig::GetConfig(*context);
  auto scan_data = duckdb::make_uniq<TempCollectionScanData>(
      &temp_collections_, &kernel_temp_tables_, &qjit_temp_meta_,
      &temp_state_mutex_);
  db_config.replacement_scans.emplace_back(TempCollectionReplacementScan,
                                           std::move(scan_data));
}

// Table function callbacks
duckdb::unique_ptr<duckdb::FunctionData> DuckDBAdapter::TempCollectionBind(
    duckdb::ClientContext &context, duckdb::TableFunctionBindInput &input,
    duckdb::vector<duckdb::LogicalType> &return_types,
    duckdb::vector<duckdb::string> &names) {

  auto &info = input.info->Cast<TempCollectionScanInfo>();
  auto table_name = input.inputs[0].GetValue<duckdb::string>();

  // See TempCollectionReplacementScan: bg binds vs main-thread store.
  std::shared_lock<std::shared_timed_mutex> guard;
  if (info.temp_state_mutex)
    guard = std::shared_lock<std::shared_timed_mutex>(*info.temp_state_mutex);

  auto it = info.temp_collections->find(table_name);
  if (it == info.temp_collections->end()) {
    throw duckdb::BinderException("Temp collection '%s' not found", table_name);
  }

  auto &stored = it->second;
  return_types = stored.collection->Types();
  for (auto &col_name : stored.column_names) {
    names.push_back(col_name);
  }

  auto result = duckdb::make_uniq<TempCollectionFunctionData>();
  result->collection = stored.collection.get();
  result->has_override_cardinality = stored.has_override_cardinality;
  result->override_cardinality = stored.override_cardinality;
  result->table_name = table_name;
  return std::move(result);
}

duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
DuckDBAdapter::TempCollectionInitGlobal(duckdb::ClientContext &context,
                                        duckdb::TableFunctionInitInput &input) {
  auto state = duckdb::make_uniq<TempCollectionGlobalState>();
  state->column_ids = input.column_ids;
  return std::move(state);
}

void DuckDBAdapter::TempCollectionScanFunc(duckdb::ClientContext &context,
                                           duckdb::TableFunctionInput &data,
                                           duckdb::DataChunk &output) {
  auto &bind_data = data.bind_data->Cast<TempCollectionFunctionData>();
  auto &state = data.global_state->Cast<TempCollectionGlobalState>();

  if (!state.initialized) {
    if (!state.column_ids.empty()) {
      bind_data.collection->InitializeScan(state.scan_state, state.column_ids);
    } else {
      bind_data.collection->InitializeScan(state.scan_state);
    }
    state.initialized = true;
  }

  bind_data.collection->Scan(state.scan_state, output);
}

duckdb::unique_ptr<duckdb::NodeStatistics>
DuckDBAdapter::TempCollectionCardinality(
    duckdb::ClientContext &context, const duckdb::FunctionData *bind_data) {
  auto &data = bind_data->Cast<TempCollectionFunctionData>();
  duckdb::idx_t cardinality;
  if (data.has_override_cardinality) {
    cardinality = data.override_cardinality;
  } else {
    cardinality = data.collection->Count();
  }
  return duckdb::make_uniq<duckdb::NodeStatistics>(cardinality, cardinality);
}

// Replacement scan callback
duckdb::unique_ptr<duckdb::TableRef>
DuckDBAdapter::TempCollectionReplacementScan(
    duckdb::ClientContext &context, duckdb::ReplacementScanInput &input,
    duckdb::optional_ptr<duckdb::ReplacementScanData> data) {

  auto &scan_data = data->Cast<TempCollectionScanData>();
  const auto &table_name = input.table_name;

  // Bg spec-compile threads bind through this scan while the main thread
  // may insert/replace temp entries (compensate-mode early launches and
  // zombie specs overlap Execute + store).
  std::shared_lock<std::shared_timed_mutex> guard;
  if (scan_data.temp_state_mutex)
    guard = std::shared_lock<std::shared_timed_mutex>(*scan_data.temp_state_mutex);

  // Check temp collections first
  if (scan_data.temp_collections->find(table_name) !=
      scan_data.temp_collections->end()) {
    auto table_ref = duckdb::make_uniq<duckdb::TableFunctionRef>();
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
    children.push_back(duckdb::make_uniq<duckdb::ConstantExpression>(
        duckdb::Value(table_name)));
    table_ref->function = duckdb::make_uniq<duckdb::FunctionExpression>(
        "scan_temp_collection", std::move(children));
    table_ref->alias = table_name;
    return std::move(table_ref);
  }

  // Check kernel temp tables
  if (scan_data.kernel_temps &&
      scan_data.kernel_temps->find(table_name) !=
          scan_data.kernel_temps->end()) {
    auto table_ref = duckdb::make_uniq<duckdb::TableFunctionRef>();
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
    children.push_back(duckdb::make_uniq<duckdb::ConstantExpression>(
        duckdb::Value(table_name)));
    table_ref->function = duckdb::make_uniq<duckdb::FunctionExpression>(
        "scan_kernel_temp", std::move(children));
    table_ref->alias = table_name;
    return std::move(table_ref);
  }

  // Check qjit temp tables (the map is only populated under HAVE_LLVM;
  // scan_qjit_temp is registered iff entries can exist)
  if (scan_data.qjit_meta &&
      scan_data.qjit_meta->find(table_name) != scan_data.qjit_meta->end()) {
    auto table_ref = duckdb::make_uniq<duckdb::TableFunctionRef>();
    duckdb::vector<duckdb::unique_ptr<duckdb::ParsedExpression>> children;
    children.push_back(duckdb::make_uniq<duckdb::ConstantExpression>(
        duckdb::Value(table_name)));
    table_ref->function = duckdb::make_uniq<duckdb::FunctionExpression>(
        "scan_qjit_temp", std::move(children));
    table_ref->alias = table_name;
    return std::move(table_ref);
  }

  return nullptr;
}

// Kernel temp table callbacks
duckdb::unique_ptr<duckdb::FunctionData> DuckDBAdapter::KernelTempBind(
    duckdb::ClientContext &context, duckdb::TableFunctionBindInput &input,
    duckdb::vector<duckdb::LogicalType> &return_types,
    duckdb::vector<duckdb::string> &names) {

  auto &info = input.info->Cast<KernelTempScanInfo>();
  auto table_name = input.inputs[0].GetValue<duckdb::string>();

  auto it = info.kernel_temps->find(table_name);
  if (it == info.kernel_temps->end()) {
    throw duckdb::BinderException("Kernel temp table '%s' not found",
                                  table_name);
  }

  const auto *flat = it->second;
  for (size_t i = 0; i < flat->columns.size(); i++) {
    if (flat->columns[i].type == storage::FlatColumnType::INT32)
      return_types.push_back(duckdb::LogicalType::INTEGER);
    else
      return_types.push_back(duckdb::LogicalType::VARCHAR);
    names.push_back(flat->column_names[i]);
  }

  auto result = duckdb::make_uniq<KernelTempFunctionData>();
  result->flat_table = flat;
  result->table_name = table_name;
  return std::move(result);
}

duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
DuckDBAdapter::KernelTempInitGlobal(duckdb::ClientContext &context,
                                    duckdb::TableFunctionInitInput &input) {
  auto state = duckdb::make_uniq<KernelTempGlobalState>();
  state->column_ids = input.column_ids;
  return std::move(state);
}

void DuckDBAdapter::KernelTempScanFunc(duckdb::ClientContext &context,
                                       duckdb::TableFunctionInput &data,
                                       duckdb::DataChunk &output) {
  auto &bind_data = data.bind_data->Cast<KernelTempFunctionData>();
  auto &state = data.global_state->Cast<KernelTempGlobalState>();
  const auto *flat = bind_data.flat_table;

  uint64_t remaining = flat->row_count - state.current_row;
  if (remaining == 0) {
    output.SetCardinality(0);
    return;
  }

  uint64_t count = std::min(remaining, (uint64_t)STANDARD_VECTOR_SIZE);

  for (duckdb::idx_t out_idx = 0; out_idx < state.column_ids.size();
       out_idx++) {
    auto col_idx = state.column_ids[out_idx];
    if (col_idx == duckdb::COLUMN_IDENTIFIER_ROW_ID)
      continue;

    const auto &col = flat->columns[col_idx];
    auto &vec = output.data[out_idx];

    if (col.type == storage::FlatColumnType::INT32) {
      auto *src = reinterpret_cast<const int32_t *>(col.data_raw) +
                  state.current_row;
      auto *dst = duckdb::FlatVector::GetData<int32_t>(vec);
      std::memcpy(dst, src, count * sizeof(int32_t));
    } else {
      auto *str_vec = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
      for (uint64_t r = 0; r < count; r++) {
        uint32_t len;
        const char *ptr = col.GetVarchar(state.current_row + r, len);
        str_vec[r] = duckdb::StringVector::AddString(vec, ptr, len);
      }
    }
  }

  output.SetCardinality(count);
  state.current_row += count;
}

duckdb::unique_ptr<duckdb::NodeStatistics>
DuckDBAdapter::KernelTempCardinality(duckdb::ClientContext &context,
                                     const duckdb::FunctionData *bind_data) {
  auto &data = bind_data->Cast<KernelTempFunctionData>();
  auto card = data.flat_table->row_count;
  return duckdb::make_uniq<duckdb::NodeStatistics>(card, card);
}

#ifdef HAVE_LLVM
// Qjit temp table callbacks (scan_qjit_temp)
duckdb::unique_ptr<duckdb::FunctionData> DuckDBAdapter::QjitTempBind(
    duckdb::ClientContext &context, duckdb::TableFunctionBindInput &input,
    duckdb::vector<duckdb::LogicalType> &return_types,
    duckdb::vector<duckdb::string> &names) {

  auto &info = input.info->Cast<QjitTempScanInfo>();
  auto table_name = input.inputs[0].GetValue<duckdb::string>();

  // See TempCollectionReplacementScan: bg binds vs main-thread store.
  std::shared_lock<std::shared_timed_mutex> guard;
  if (info.temp_state_mutex)
    guard = std::shared_lock<std::shared_timed_mutex>(*info.temp_state_mutex);

  auto mit = info.qjit_meta->find(table_name);
  auto tit = info.qjit_temps->find(table_name);
  if (mit == info.qjit_meta->end() || tit == info.qjit_temps->end()) {
    throw duckdb::BinderException("Qjit temp table '%s' not found",
                                  table_name);
  }

  auto &meta = mit->second;
  return_types = meta.types;
  for (auto &col_name : meta.column_names) {
    names.push_back(col_name);
  }

  auto result = duckdb::make_uniq<QjitTempFunctionData>();
  result->table = tit->second.get();
  result->table_name = table_name;
  result->has_override_cardinality = meta.has_override_cardinality;
  result->override_cardinality = meta.override_cardinality;
  return std::move(result);
}

duckdb::unique_ptr<duckdb::GlobalTableFunctionState>
DuckDBAdapter::QjitTempInitGlobal(duckdb::ClientContext &context,
                                  duckdb::TableFunctionInitInput &input) {
  auto state = duckdb::make_uniq<QjitTempGlobalState>();
  state->column_ids = input.column_ids;
  return std::move(state);
}

void DuckDBAdapter::QjitTempScanFunc(duckdb::ClientContext &context,
                                     duckdb::TableFunctionInput &data,
                                     duckdb::DataChunk &output) {
  auto &bind_data = data.bind_data->Cast<QjitTempFunctionData>();
  auto &state = data.global_state->Cast<QjitTempGlobalState>();
  const auto *t = bind_data.table;

  uint64_t remaining = t->NumRows() - state.current_row;
  if (remaining == 0) {
    output.SetCardinality(0);
    return;
  }

  uint64_t count = std::min(remaining, (uint64_t)STANDARD_VECTOR_SIZE);

  for (duckdb::idx_t out_idx = 0; out_idx < state.column_ids.size();
       out_idx++) {
    auto col_idx = state.column_ids[out_idx];
    if (col_idx == duckdb::COLUMN_IDENTIFIER_ROW_ID)
      continue;

    auto &vec = output.data[out_idx];
    switch (t->Col(col_idx).dtype) {
    case AQP_DTYPE_INT32: {
      auto *dst = duckdb::FlatVector::GetData<int32_t>(vec);
      const auto *src =
          static_cast<const int32_t *>(t->Data(col_idx)) + state.current_row;
      std::memcpy(dst, src, count * sizeof(int32_t));
      break;
    }
    case AQP_DTYPE_INT64: {
      auto *dst = duckdb::FlatVector::GetData<int64_t>(vec);
      const auto *src =
          static_cast<const int64_t *>(t->Data(col_idx)) + state.current_row;
      std::memcpy(dst, src, count * sizeof(int64_t));
      break;
    }
    case AQP_DTYPE_VARCHAR: {
      // QjitString is bit-compatible with duckdb::string_t; long strings
      // point into the QjitTable arena, which outlives the scan (temps are
      // cleared only at per-query reset) — zero-copy views, no AddString.
      static_assert(sizeof(QjitString) == sizeof(duckdb::string_t),
                    "QjitString must be bit-compatible with string_t");
      auto *dst = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
      const auto *src = static_cast<const QjitString *>(t->Data(col_idx)) +
                        state.current_row;
      std::memcpy(dst, src, count * sizeof(duckdb::string_t));
      break;
    }
    default:
      throw duckdb::InternalException("scan_qjit_temp: unsupported dtype");
    }

    if (t->HasNulls(col_idx)) {
      const uint64_t *vmask = t->Validity(col_idx);
      auto &validity = duckdb::FlatVector::Validity(vec);
      for (uint64_t r = 0; r < count; r++) {
        if (!qjit::RowValid(vmask, state.current_row + r))
          validity.SetInvalid(r);
      }
    }
  }

  output.SetCardinality(count);
  state.current_row += count;
}

duckdb::unique_ptr<duckdb::NodeStatistics>
DuckDBAdapter::QjitTempCardinality(duckdb::ClientContext &context,
                                   const duckdb::FunctionData *bind_data) {
  auto &data = bind_data->Cast<QjitTempFunctionData>();
  duckdb::idx_t cardinality;
  if (data.has_override_cardinality) {
    cardinality = data.override_cardinality;
  } else {
    cardinality = data.table->NumRows();
  }
  return duckdb::make_uniq<duckdb::NodeStatistics>(cardinality, cardinality);
}
#endif

void DuckDBAdapter::RegisterKernelTemp(const std::string &name,
                                       const storage::FlatTable *table) {
  kernel_temp_tables_[name] = table;
}

void DuckDBAdapter::ClearKernelTemps() { kernel_temp_tables_.clear(); }

const StoredTempResult *
DuckDBAdapter::GetStoredTempResult(const std::string &name) const {
  auto it = temp_collections_.find(name);
  if (it == temp_collections_.end())
    return nullptr;
  return &it->second;
}

void DuckDBAdapter::CreateTempFromFlatTable(
    const storage::FlatTable &flat, const std::string &temp_table_name) {

  auto *ctx = GetClientContext();

  // Build DuckDB types
  duckdb::vector<duckdb::LogicalType> types;
  for (const auto &col : flat.columns) {
    if (col.type == storage::FlatColumnType::INT32)
      types.push_back(duckdb::LogicalType::INTEGER);
    else
      types.push_back(duckdb::LogicalType::VARCHAR);
  }

  // Build a ColumnDataCollection from the flat arrays
  auto collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(*ctx, types);

  duckdb::DataChunk chunk;
  chunk.Initialize(*ctx, types);

  uint64_t offset = 0;
  while (offset < flat.row_count) {
    uint64_t count = std::min(flat.row_count - offset,
                              (uint64_t)STANDARD_VECTOR_SIZE);
    chunk.Reset();
    chunk.SetCardinality(count);

    for (size_t c = 0; c < flat.columns.size(); c++) {
      const auto &col = flat.columns[c];
      auto &vec = chunk.data[c];

      if (col.type == storage::FlatColumnType::INT32) {
        auto *src = reinterpret_cast<const int32_t *>(col.data_raw) + offset;
        auto *dst = duckdb::FlatVector::GetData<int32_t>(vec);
        std::memcpy(dst, src, count * sizeof(int32_t));
      } else {
        auto *str_vec = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
        for (uint64_t r = 0; r < count; r++) {
          uint32_t len;
          const char *ptr = col.GetVarchar(offset + r, len);
          str_vec[r] = duckdb::StringVector::AddString(vec, ptr, len);
        }
      }
    }

    collection->Append(chunk);
    offset += count;
  }

  // Set up DuckDB internal state
  auto data_chunk_index = planner->binder->GenerateTableIndex();
  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;
  temp_table_types = types;

  chunk_col_names_[data_chunk_index] = flat.column_names;
  for (duckdb::idx_t i = 0; i < types.size(); i++) {
    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
                                  flat.column_names[i]);
  }

  StoredTempResult stored;
  stored.collection = std::move(collection);
  stored.column_names = flat.column_names;
  temp_collections_[temp_table_name] = std::move(stored);

  temp_table_card_.emplace(temp_table_name,
                           static_cast<int64_t>(flat.row_count));
}

void DuckDBAdapter::RegisterTempMetadata(
    const storage::FlatTable &flat, const std::string &temp_table_name) {

  duckdb::vector<duckdb::LogicalType> types;
  for (const auto &col : flat.columns) {
    if (col.type == storage::FlatColumnType::INT32)
      types.push_back(duckdb::LogicalType::INTEGER);
    else
      types.push_back(duckdb::LogicalType::VARCHAR);
  }

  auto data_chunk_index = planner->binder->GenerateTableIndex();
  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;
  temp_table_types = types;

  chunk_col_names_[data_chunk_index] = flat.column_names;
  for (duckdb::idx_t i = 0; i < types.size(); i++) {
    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
                                  flat.column_names[i]);
  }

  temp_table_card_.emplace(temp_table_name,
                           static_cast<int64_t>(flat.row_count));
}

void DuckDBAdapter::MaterializeCatalogTempTable(
    const std::string &temp_table_name,
    const duckdb::vector<duckdb::LogicalType> &types,
    const std::vector<std::string> &col_names,
    duckdb::ColumnDataCollection &collection) {
  // Use a separate connection to avoid disturbing the main connection's
  // transaction/planner state. This connection's catalog changes are
  // visible to the main connection within the same DuckDB instance.
  if (!bidir_conn_)
    bidir_conn_ = std::make_unique<duckdb::Connection>(*db);

  auto ctx2 = bidir_conn_->context.get();
  ctx2->transaction.BeginTransaction();

  auto &catalog = duckdb::Catalog::GetCatalog(*ctx2, TEMP_CATALOG);

  auto info = duckdb::make_uniq<duckdb::CreateTableInfo>(
      TEMP_CATALOG, DEFAULT_SCHEMA, temp_table_name);
  info->temporary = true;
  info->on_conflict = duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
  for (duckdb::idx_t i = 0; i < types.size(); i++) {
    std::string name = i < col_names.size() ? col_names[i]
                                            : "col_" + std::to_string(i);
    info->columns.AddColumn(duckdb::ColumnDefinition(name, types[i]));
  }
  auto created_table = catalog.CreateTable(*ctx2, std::move(info));
  auto &entry = created_table->Cast<duckdb::TableCatalogEntry>();
  auto temp_binder = duckdb::Binder::CreateBinder(*ctx2);
  const duckdb::vector<duckdb::unique_ptr<duckdb::BoundConstraint>>
      bound_constraints = temp_binder->BindConstraints(entry);
  auto &storage = entry.GetStorage();
  storage.LocalAppend(entry, *ctx2, collection, bound_constraints, nullptr);

  ctx2->transaction.Commit();
}

#ifdef HAVE_LLVM
void DuckDBAdapter::MaterializeCatalogTempFromQjit(
    const std::string &temp_table_name,
    const qjit::QjitTable &qtable,
    const duckdb::vector<duckdb::LogicalType> &types,
    const std::vector<std::string> &col_names) {
  auto *ctx = GetClientContext();

  auto cdc = duckdb::make_uniq<duckdb::ColumnDataCollection>(*ctx, types);
  duckdb::DataChunk chunk;
  chunk.Initialize(*ctx, types);

  uint64_t offset = 0;
  while (offset < qtable.NumRows()) {
    uint64_t count = std::min(qtable.NumRows() - offset,
                              (uint64_t)STANDARD_VECTOR_SIZE);
    chunk.Reset();
    chunk.SetCardinality(count);
    for (duckdb::idx_t c = 0; c < types.size(); c++) {
      auto &vec = chunk.data[c];
      int32_t dt = qtable.Col(c).dtype;
      if (dt == AQP_DTYPE_INT32) {
        auto *dst = duckdb::FlatVector::GetData<int32_t>(vec);
        for (uint64_t r = 0; r < count; r++) {
          if (qtable.ValueValid(c, offset + r))
            dst[r] = qtable.GetI32(c, offset + r);
          else
            duckdb::FlatVector::SetNull(vec, r, true);
        }
      } else if (dt == AQP_DTYPE_INT64) {
        auto *dst = duckdb::FlatVector::GetData<int64_t>(vec);
        for (uint64_t r = 0; r < count; r++) {
          if (qtable.ValueValid(c, offset + r))
            dst[r] = qtable.GetI64(c, offset + r);
          else
            duckdb::FlatVector::SetNull(vec, r, true);
        }
      } else {
        auto *dst = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
        for (uint64_t r = 0; r < count; r++) {
          if (qtable.ValueValid(c, offset + r)) {
            auto qs = qtable.GetStr(c, offset + r);
            dst[r] = duckdb::StringVector::AddString(
                vec, qjit::StringData(qs), qjit::StringLen(qs));
          } else {
            duckdb::FlatVector::SetNull(vec, r, true);
          }
        }
      }
    }
    cdc->Append(chunk);
    offset += count;
  }

  MaterializeCatalogTempTable(temp_table_name, types, col_names, *cdc);
}
#endif

#endif

void DuckDBAdapter::ParseSQL(const std::string &sql) {
  auto context = GetClientContext();

  // Begin transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  duckdb::Parser parser(context->GetParserOptions());
  parser.ParseQuery(sql);

  if (parser.statements.empty()) {
    throw std::runtime_error("No statements found!");
  }

  if (duckdb::StatementType::SELECT_STATEMENT != parser.statements[0]->type) {
    throw std::runtime_error("Only SELECT queries supported!");
  }

  planner = std::make_unique<duckdb::Planner>(*context);
  planner->CreatePlan(std::move(parser.statements[0]));

  if (!planner->plan) {
    throw std::runtime_error("Failed to create logical plan");
  }

  plan = std::move(planner->plan);

  // Commit transaction if we started one
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }
}

void DuckDBAdapter::Optimize() {
  auto context = GetClientContext();

  if (!plan) {
    throw std::runtime_error("Cannot optimize null plan");
  }

  // Check if optimization is enabled and required
  if (!plan->RequireOptimizer()) {
#ifndef NDEBUG
    std::cout << "[DuckDB] Plan does not require optimization" << std::endl;
#endif
    return;
  }

  // Begin transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  if (!planner || !planner->binder) {
    throw std::runtime_error("Binder not available. Call ParseSQL first.");
  }

  duckdb::Optimizer optimizer(*planner->binder, *context);
  if (disable_engine_optimizer_) {
    plan = optimizer.FilterOptimize(std::move(plan));
  } else {
    plan = optimizer.Optimize(std::move(plan));
  }

  // Commit transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }
}

void DuckDBAdapter::FilterOptimize() {
  auto context = GetClientContext();

  if (!plan) {
    throw std::runtime_error("Cannot optimize null plan");
  }

  // Check if optimization is enabled and required
  if (!plan->RequireOptimizer()) {
#ifndef NDEBUG
    std::cout << "[DuckDB] Plan does not require optimization" << std::endl;
#endif
    return;
  }

  // Begin transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  if (!planner || !planner->binder) {
    throw std::runtime_error("Binder not available. Call ParseSQL first.");
  }

  // Create optimizer and run PreOptimize
  duckdb::Optimizer optimizer(*planner->binder, *context);
  auto optimized_plan = optimizer.FilterOptimize(std::move(plan));

  // Store the optimized plan
  plan = std::move(optimized_plan);

  // Commit transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }
}

void DuckDBAdapter::PostOptimizePlan() {
  auto context = GetClientContext();

  if (!plan) {
    throw std::runtime_error("Cannot optimize null plan");
  }

  // Check if optimization is enabled and required
  if (!plan->RequireOptimizer()) {
#ifndef NDEBUG
    std::cout << "[DuckDB] Plan does not require optimization" << std::endl;
#endif
    return;
  }

  // Begin transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  if (!planner || !planner->binder) {
    throw std::runtime_error("Binder not available. Call ParseSQL first.");
  }

  // Create optimizer and run PreOptimize
  duckdb::Optimizer optimizer(*planner->binder, *context);
  auto optimized_plan = optimizer.PostOptimize(std::move(plan));

  // Store the optimized plan
  plan = std::move(optimized_plan);

  // Commit transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }
}

// Walk a logical plan tree and update intermediate_table_map for any
// LogicalGet nodes that reference temp tables via the scan_temp_collection
// table function.  After ReOptimizeIR re-parses SQL, DuckDB assigns fresh
// table indices; this function synchronises them with the map so that
// ConvertDuckDBPlanToIR can identify temp tables correctly.
static void RebuildTempTableIndices(
    duckdb::LogicalOperator *op,
    std::unordered_map<unsigned int, std::string> &intermediate_table_map,
    std::unordered_map<unsigned int, std::vector<std::string>> &chunk_col_names,
    const std::unordered_map<std::string, StoredTempResult> &temp_collections,
    const std::unordered_map<std::string, QjitTempMeta> &qjit_temp_meta) {
  if (!op)
    return;
  if (op->type == duckdb::LogicalOperatorType::LOGICAL_GET) {
    auto &get_op = op->Cast<duckdb::LogicalGet>();
    if (get_op.function.name == "scan_temp_collection" &&
        !get_op.parameters.empty()) {
      auto temp_name = get_op.parameters[0].GetValue<std::string>();
      auto it = temp_collections.find(temp_name);
      if (it != temp_collections.end()) {
        intermediate_table_map[get_op.table_index] = temp_name;
        chunk_col_names[get_op.table_index] = it->second.column_names;
      }
    } else if (get_op.function.name == "scan_qjit_temp" &&
               !get_op.parameters.empty()) {
      auto temp_name = get_op.parameters[0].GetValue<std::string>();
      auto it = qjit_temp_meta.find(temp_name);
      if (it != qjit_temp_meta.end()) {
        intermediate_table_map[get_op.table_index] = temp_name;
        chunk_col_names[get_op.table_index] = it->second.column_names;
      }
    }
#ifdef HAVE_LLVM
    else if (get_op.function.name == "seq_scan" && get_op.bind_data) {
      auto *bd = dynamic_cast<duckdb::TableScanBindData *>(
          get_op.bind_data.get());
      if (bd && bd->table.name.find("temp") == 0) {
        const auto &tname = bd->table.name;
        auto it = temp_collections.find(tname);
        if (it != temp_collections.end()) {
          intermediate_table_map[get_op.table_index] = tname;
          chunk_col_names[get_op.table_index] = it->second.column_names;
        } else {
          auto qit = qjit_temp_meta.find(tname);
          if (qit != qjit_temp_meta.end()) {
            intermediate_table_map[get_op.table_index] = tname;
            chunk_col_names[get_op.table_index] = qit->second.column_names;
          }
        }
      }
    }
#endif
  }
  for (auto &child : op->children)
    RebuildTempTableIndices(child.get(), intermediate_table_map,
                            chunk_col_names, temp_collections, qjit_temp_meta);
}

std::unique_ptr<ir_sql_converter::AQPStmt>
DuckDBAdapter::ReOptimizeIR(std::unique_ptr<ir_sql_converter::AQPStmt> ir) {
  if (!ir)
    return ir;

  auto sql = GenerateSQL(*ir, subquery_index);

  ParseSQL(sql);

  // Run full optimization (PreOptimize + MiddleOptimize on the fork,
  // Optimize() on vanilla DuckDB) so join ordering uses actual cardinalities.
  auto context = GetClientContext();
  if (context->transaction.IsAutoCommit())
    context->transaction.BeginTransaction();

  if (planner && planner->binder && plan && plan->RequireOptimizer()) {
    duckdb::Optimizer optimizer(*planner->binder, *context);
    // Temporarily disable the split flag so Optimize() runs the full
    // pipeline including JOIN_ORDER (the fork skips it when the flag is on).
    // This makes the code forward-compatible with vanilla DuckDB where
    // Optimize() always includes JOIN_ORDER.
    auto &cfg = context->config;
    bool saved = cfg.enable_dbshaker_query_split;
    cfg.enable_dbshaker_query_split = false;
    if (disable_engine_optimizer_)
      plan = optimizer.FilterOptimize(std::move(plan));
    else
      plan = optimizer.Optimize(std::move(plan));
    cfg.enable_dbshaker_query_split = saved;
  }

  if (context->transaction.IsAutoCommit())
    context->transaction.Commit();

  // After re-parsing, DuckDB assigned fresh table indices to temp tables
  // (they come through as LogicalGet via replacement scan, not as
  // LogicalColumnDataGet).  Update intermediate_table_map so that
  // ConvertPlanToIR can identify them and create SimplestChunk nodes.
  if (plan) {
    intermediate_table_map.clear();
    chunk_col_names_.clear();
    RebuildTempTableIndices(plan.get(), intermediate_table_map,
                            chunk_col_names_, temp_collections_,
                            qjit_temp_meta_);
  }

  // Convert re-optimized plan back to IR
  return ConvertPlanToIR();
}

void *DuckDBAdapter::GetLogicalPlan() {
  return static_cast<void *>(plan.get());
}

// Collect ALL FilterNodes in DFS order.
static void
CollectAllFilterNodes(const ir_sql_converter::AQPStmt *ir,
                      std::vector<const ir_sql_converter::AQPStmt *> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::FilterNode)
    out.push_back(ir);
  for (const auto &child : ir->children)
    CollectAllFilterNodes(child.get(), out);
}

// Collect ALL ScanNodes in DFS order.
static void
CollectAllScanNodes(const ir_sql_converter::AQPStmt *ir,
                    std::vector<const ir_sql_converter::AQPStmt *> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode)
    out.push_back(ir);
  for (const auto &child : ir->children)
    CollectAllScanNodes(child.get(), out);
}

std::unique_ptr<ir_sql_converter::AQPStmt> DuckDBAdapter::ConvertPlanToIR() {
  auto context = GetClientContext();

#ifndef NDEBUG
  std::cerr << "[AQP-JIT-TRACE] === Logical plan before IR conversion ===\n";
  if (plan)
    plan->Print();
#endif

  auto ir = ir_sql_converter::ConvertDuckDBPlanToIR(
      *planner->binder, *context, plan.get(), intermediate_table_map,
      false, &chunk_col_names_);

#ifndef NDEBUG
  if (ir) {
    std::cerr << "[AQP-JIT-TRACE] === IR after conversion ===\n";
    std::vector<const ir_sql_converter::AQPStmt *> all_filters;
    CollectAllFilterNodes(ir.get(), all_filters);
    std::cerr << "[AQP-JIT-TRACE] IR has " << all_filters.size()
              << " FilterNode(s)\n";
    for (size_t fi = 0; fi < all_filters.size(); fi++) {
      std::cerr << "[AQP-JIT-TRACE]   filter[" << fi
                << "] qual_vec.size=" << all_filters[fi]->qual_vec.size()
                << "\n";
      for (size_t qi = 0; qi < all_filters[fi]->qual_vec.size(); qi++) {
        std::string qs = all_filters[fi]->qual_vec[qi]->Print(false);
        std::cerr << "[AQP-JIT-TRACE]     qual[" << qi << "]: " << qs << "\n";
      }
    }
    // Also dump all ScanNodes and their qual_vecs
    std::vector<const ir_sql_converter::AQPStmt *> all_scans;
    CollectAllScanNodes(ir.get(), all_scans);
    std::cerr << "[AQP-JIT-TRACE] IR has " << all_scans.size()
              << " ScanNode(s)\n";
    for (size_t si = 0; si < all_scans.size(); si++) {
      auto *sn = all_scans[si];
      auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(sn);
      std::cerr << "[AQP-JIT-TRACE]   scan[" << si
                << "] table_idx=" << scan->GetTableIndex()
                << " target_list.size=" << sn->target_list.size()
                << " qual_vec.size=" << sn->qual_vec.size() << "\n";
      for (size_t qi = 0; qi < sn->qual_vec.size(); qi++) {
        std::string qs = sn->qual_vec[qi]->Print(false);
        std::cerr << "[AQP-JIT-TRACE]     qual[" << qi << "]: " << qs << "\n";
      }
    }
  }
#endif

  return std::move(ir);
}

#ifdef HAVE_LLVM
namespace {
// Query-jit only analyzes SELECT statements. Non-SELECT SQL reaching
// ExecuteSQL (e.g. the FK splitters' per-table ANALYZE, or multi-statement
// combined SQL) must take the legacy path unchanged — ParseSQL would throw
// and the timing-column layout must match the baseline interpreter run.
bool IsSelectStatement(const std::string &sql) {
  size_t i = sql.find_first_not_of(" \t\r\n");
  if (i == std::string::npos)
    return false;
  auto starts_with = [&](const char *kw) {
    size_t n = std::strlen(kw);
    if (sql.size() - i < n)
      return false;
    for (size_t k = 0; k < n; k++)
      if (std::toupper(static_cast<unsigned char>(sql[i + k])) != kw[k])
        return false;
    return true;
  };
  return starts_with("SELECT") || starts_with("WITH");
}
} // namespace
#endif

QueryResult DuckDBAdapter::ExecuteSQL(const std::string &sql) {
  QueryResult result;

#ifdef HAVE_LLVM
#ifndef NDEBUG
  std::cerr << "[AQP-JIT-TRACE] ExecuteSQL: jit_pending_ir_="
            << (void *)jit_pending_ir_ << " jit_flags_=0x" << std::hex
            << jit_flags_ << std::dec << "\n";
#endif

  // When JIT is requested but no pending IR, build the IR pipeline:
  // parse -> filter optimize -> convert to IR.
  auto timer = chrono_tic();

  // Query-jit (--jit-level=query): analyze the sub-query IR, compile it if
  // it is inside the v1 coverage (scan/filter/project, inner hash joins,
  // ungrouped MIN/MAX/COUNT aggregates), execute the compiled entry over
  // the qjit runtime; otherwise the interpreter runs the prepared statement
  // from the analysis pass. Mutually exclusive with the other JIT levels
  // (enforced in ParamConfig), so jit_pending_ir_ is never set here.
  if (query_jit_ && IsSelectStatement(sql)) {
    auto prep = PrepareWithQueryJitAnalysis(sql, "result");
    auto prepared = std::move(prep.prepared);
    std::unique_ptr<QjitCompiled> compiled;
    if (prepared && !prepared->HasError() && prep.ir)
      compiled = TryCompileQueryJit(*prep.ir, prep.analysis, *prepared,
                                    "result");
    if (enable_timing_) {
      // jit_compile column: analysis + prepare + (Phase 2) codegen time,
      // plus any spec-jit wait charged by the splitter (end-of-loop drain).
      auto analyze_us =
          chrono_toc(&timer, "ExecuteSQL::qjit analyze time\n", false);
      WriteJitTimingColumn(analyze_us + ConsumeSpecWaitUs());
    }
    if (prepared->HasError())
      throw std::runtime_error("Query failed: " + prepared->GetError());
    if (compiled) {
      qjit::QjitTable qtable(compiled->out_descs,
                             qjit_executor_->NumWorkers());
      int64_t rows = qjit_executor_->Run(
          reinterpret_cast<QjitQueryFn>(compiled->fn), compiled->srcs,
          compiled->ht_tuple_sizes, compiled->agg_descs,
          compiled->agg_output_cells, qtable, compiled->ht_key0_offsets,
          compiled->params_buf);
      if (rows >= 0) {
        if (enable_timing_) {
          auto run_us =
              chrono_toc(&timer, "ExecuteSQL::qjit run time\n", false);
          WriteJitTimingColumn(run_us);
        }
#ifndef NDEBUG
        fprintf(stderr, "[AQP-QJIT] exec label=result rows=%lld\n",
                (long long)rows);
#endif
        result.num_columns = qtable.NumCols();
        for (size_t i = 0; i < qtable.NumCols(); i++)
          result.column_names.push_back(qtable.Col(i).name);
        result.num_rows = 0;
        for (uint64_t r = 0; r < (uint64_t)rows; r++) {
          std::vector<std::string> row_data;
          row_data.reserve(result.num_columns);
          for (size_t col = 0; col < result.num_columns; col++) {
            // Match duckdb::Value::ToString: NULL prints "NULL", INTEGER/
            // BIGINT print the decimal digits, VARCHAR prints the raw bytes.
            if (!qtable.ValueValid(col, r)) {
              row_data.emplace_back("NULL");
            } else if (qtable.Col(col).dtype == AQP_DTYPE_INT32) {
              uint8_t sc = qtable.Col(col).decimal_scale;
              if (sc > 0) {
                int32_t raw = qtable.GetI32(col, r);
                int32_t divisor = 1;
                for (uint8_t s = 0; s < sc; s++) divisor *= 10;
                char buf[32];
                snprintf(buf, sizeof(buf), "%s%d.%0*d",
                         raw < 0 ? "-" : "",
                         std::abs(raw) / divisor, sc,
                         std::abs(raw) % divisor);
                row_data.emplace_back(buf);
              } else {
                row_data.push_back(std::to_string(qtable.GetI32(col, r)));
              }
            } else if (qtable.Col(col).dtype == AQP_DTYPE_INT64) {
              uint8_t sc = qtable.Col(col).decimal_scale;
              if (sc > 0) {
                int64_t raw = qtable.GetI64(col, r);
                int64_t divisor = 1;
                for (uint8_t s = 0; s < sc; s++) divisor *= 10;
                char buf[48];
                snprintf(buf, sizeof(buf), "%s%lld.%0*lld",
                         raw < 0 ? "-" : "",
                         (long long)(std::abs(raw) / divisor), (int)sc,
                         (long long)(std::abs(raw) % divisor));
                row_data.emplace_back(buf);
              } else {
                row_data.push_back(std::to_string(qtable.GetI64(col, r)));
              }
            } else if (qtable.Col(col).dtype == AQP_DTYPE_DOUBLE) {
              row_data.push_back(std::to_string(qtable.GetF64(col, r)));
            } else {
              QjitString s = qtable.GetStr(col, r);
              row_data.emplace_back(qjit::StringData(s), qjit::StringLen(s));
            }
          }
          result.rows.push_back(std::move(row_data));
          result.num_rows++;
        }
        if (!compiled->order_by.empty()) {
          const auto &sort_cols = compiled->order_by;
          std::stable_sort(result.rows.begin(), result.rows.end(),
              [&](const std::vector<std::string> &a,
                  const std::vector<std::string> &b) {
            for (const auto &sc : sort_cols) {
              if ((size_t)sc.col_idx >= a.size())
                continue;
              const auto &va = a[sc.col_idx];
              const auto &vb = b[sc.col_idx];
              bool a_null = (va == "NULL"), b_null = (vb == "NULL");
              if (a_null && b_null) continue;
              if (a_null) return false; // NULLS LAST
              if (b_null) return true;
              int cmp = 0;
              if (sc.dtype == AQP_DTYPE_INT32) {
                long la = strtol(va.c_str(), nullptr, 10);
                long lb = strtol(vb.c_str(), nullptr, 10);
                cmp = (la < lb) ? -1 : (la > lb) ? 1 : 0;
              } else if (sc.dtype == AQP_DTYPE_INT64) {
                long long la = strtoll(va.c_str(), nullptr, 10);
                long long lb = strtoll(vb.c_str(), nullptr, 10);
                cmp = (la < lb) ? -1 : (la > lb) ? 1 : 0;
              } else if (sc.dtype == AQP_DTYPE_DOUBLE ||
                         sc.dtype == AQP_DTYPE_FLOAT) {
                double da = strtod(va.c_str(), nullptr);
                double db = strtod(vb.c_str(), nullptr);
                cmp = (da < db) ? -1 : (da > db) ? 1 : 0;
              } else {
                cmp = va.compare(vb);
              }
              if (cmp != 0)
                return sc.ascending ? (cmp < 0) : (cmp > 0);
            }
            return false;
          });
        }
        if (compiled->limit >= 0 &&
            (int64_t)result.rows.size() > compiled->limit) {
          result.rows.resize((size_t)compiled->limit);
          result.num_rows = (int64_t)compiled->limit;
        }
        return result;
      }
      // Entry returned an error code: fall through to the interpreter (the
      // result table is discarded; nothing was emitted).
      fprintf(stderr, "[AQP-QJIT] fallback:run-error(rc=%lld) label=result\n",
              (long long)rows);
    }
    // Drop any stale JIT context from previous sub-plan iterations.
    GetClientContext()->aqp_jit_context.reset();
    if (prepared->data && prepared->data->physical_plan)
      InjectTempTableJoinStats(prepared->data->physical_plan->Root());
    duckdb::vector<duckdb::Value> bound;
    auto duckdb_result = prepared->Execute(bound, false);
    if (enable_timing_) {
      auto run_us =
          chrono_toc(&timer, "ExecuteSQL::Execute sub-SQL time\n", false);
      WriteJitTimingColumn(run_us);
    }
    if (duckdb_result->HasError())
      throw std::runtime_error("Query failed: " + duckdb_result->GetError());
    result.num_columns = duckdb_result->ColumnCount();
    for (size_t i = 0; i < result.num_columns; i++)
      result.column_names.push_back(duckdb_result->ColumnName(i));
    result.num_rows = 0;
    while (true) {
      auto chunk = duckdb_result->Fetch();
      if (!chunk || 0 == chunk->size())
        break;
      for (size_t row = 0; row < chunk->size(); row++) {
        std::vector<std::string> row_data;
        row_data.reserve(result.num_columns);
        for (size_t col = 0; col < result.num_columns; col++)
          row_data.push_back(chunk->GetValue(col, row).ToString());
        result.rows.push_back(std::move(row_data));
        result.num_rows++;
      }
    }
    return result;
  }

  // If JIT pending IR is set, use Prepare path so we can walk the physical
  // plan and register compiled filters before execution.
  if (jit_pending_ir_) {
    std::unique_ptr<duckdb::PreparedStatement> prepared;
    if (jit_pending_plan_) {
      prepared = conn->PrepareFromPlan(std::move(jit_pending_plan_),
                                       planner->names, planner->types);
    } else {
      auto og = MakeOptGuard();
      prepared = conn->Prepare(sql);
    }
    if (!prepared->HasError() && prepared->data &&
        prepared->data->physical_plan) {
      // Reset JIT context (stale function pointers would crash if dispatched).
      // Keep jit_compiler_ alive — reusing the LLJIT instance avoids
      // re-registering 15+ runtime symbols and re-initializing the JIT on every
      // sub-plan. But we DO release all previously compiled modules so the
      // machine code / IR allocations / symbol-table entries from the
      // previous query don't accumulate in the JIT for the adapter's lifetime
      // — important for long-running servers and multi-query benchmark runs.
      //
      // Ordering: clear the function-pointer map FIRST, then ResetModules.
      // The map holds raw pointers into the JIT's code; freeing the code
      // while the map still references it would leave dangling pointers
      // dispatchable. Multi-thread note: this entire reset sequence assumes
      // no concurrent dispatch is in flight on another thread. Today the
      // executor is synchronous, but when async / parallel dispatch is
      // added a lock around (reset + ResetModules + RegisterJIT) vs each
      // dispatch site is required to avoid use-after-free.
      GetClientContext()->aqp_jit_context.reset();
      if (jit_compiler_)
        jit_compiler_->ResetModules();
#ifndef NDEBUG
      std::cerr << "[AQP-JIT-TRACE] ExecuteSQL: reset aqp_jit_context "
                   "(compiler reused)\n";
#endif
      jit_consumed_ir_filters_.clear();
      jit_consumed_ir_joins_.clear();
      RegisterJIT(prepared->data->physical_plan->Root(), *jit_pending_ir_);

      // Ensure JIT context always exists when JIT level is active, even if
      // no filters were compiled. This enables JIT-gated optimizations like
      // hash join prefetching.
      auto *ctx = GetClientContext();
      if (!ctx->aqp_jit_context) {
        ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        ctx->aqp_jit_context->flags = duckdb::AQPJIT_PIPELINE;
      }

      if (enable_timing_) {
        // Record total JIT compile time (includes Prepare + RegisterJIT)
        // plus any spec-jit wait charged by the splitter.
        auto jit_compile_time =
            chrono_toc(&timer, "ExecuteSQL::jit compile time\n", false);
        WriteJitTimingColumn(jit_compile_time + ConsumeSpecWaitUs());
      }
#ifndef NDEBUG
      if (ctx->aqp_jit_context) {
        std::cerr << "[AQP-JIT] summary: flags=0x" << std::hex
                  << ctx->aqp_jit_context->flags << std::dec
                  << " expr_fns=" << ctx->aqp_jit_context->expr_fns.size()
                  << " op_fns=" << ctx->aqp_jit_context->op_fns.size()
                  << " proj_maps=" << ctx->aqp_jit_context->proj_col_maps.size()
                  << " agg_fns=" << ctx->aqp_jit_context->agg_fns.size()
                  << " pipeline_fns="
                  << ctx->aqp_jit_context->pipeline_fns.size()
                  << " scan_filter="
                  << ctx->aqp_jit_context->scan_filter_fns.size() << "\n";
      }
#endif
    } else if (enable_timing_) {
      // Keep the timing CSV rectangular: JIT-level runs always emit the
      // jit_compile column, even when registration was skipped.
      WriteJitTimingColumn(ConsumeSpecWaitUs());
    }

    jit_pending_ir_ = nullptr;
    if (prepared->data && prepared->data->physical_plan) {
      InjectTempTableJoinStats(prepared->data->physical_plan->Root());
    }
    duckdb::vector<duckdb::Value> bound;
    auto duckdb_result = prepared->Execute(bound, false);
    if (enable_timing_) {
      auto run_us =
          chrono_toc(&timer, "ExecuteSQL::Execute sub-SQL time\n", false);
      WriteJitTimingColumn(run_us);
    }
    if (duckdb_result->HasError())
      throw std::runtime_error("Query failed: " + duckdb_result->GetError());
    result.num_columns = duckdb_result->ColumnCount();
    for (size_t i = 0; i < result.num_columns; i++)
      result.column_names.push_back(duckdb_result->ColumnName(i));
    result.num_rows = 0;
    while (true) {
      auto chunk = duckdb_result->Fetch();
      if (!chunk || 0 == chunk->size())
        break;
      for (size_t row = 0; row < chunk->size(); row++) {
        std::vector<std::string> row_data;
        for (size_t col = 0; col < result.num_columns; col++)
          row_data.push_back(chunk->GetValue(col, row).ToString());
        result.rows.push_back(row_data);
        result.num_rows++;
      }
    }
  } else
#endif
  {
#ifdef HAVE_LLVM
    // Keep the timing CSV rectangular: JIT-level runs always emit the
    // jit_compile column, even when no IR was registered (e.g. the
    // cross-product guard skipped SetJITPendingIR).
    if (enable_timing_ && !query_jit_ && jit_compiler_)
      WriteJitTimingColumn(0);
#endif
    GetClientContext()->aqp_jit_context.reset();
    std::unique_ptr<duckdb::PreparedStatement> prepared;
    {
      auto og = MakeOptGuard();
      prepared = conn->Prepare(sql);
    }
    if (prepared->HasError()) {
      throw std::runtime_error("Query failed: " + prepared->GetError());
    }

#ifdef HAVE_LLVM
    if (!pending_bloom_filters_.empty() && prepared->data &&
        prepared->data->physical_plan) {
      auto *ctx3 = GetClientContext();
      if (!ctx3->aqp_jit_context)
        ctx3->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
      else
        ctx3->aqp_jit_context->bloom_scan_filters.clear();
      RegisterBloomFilters(prepared->data->physical_plan->Root(),
                           GetClientContext(), pending_bloom_filters_);
      pending_bloom_filters_.clear();
    }
#endif
    if (prepared->data && prepared->data->physical_plan)
      InjectTempTableJoinStats(prepared->data->physical_plan->Root());

    duckdb::vector<duckdb::Value> bound;
    auto duckdb_result = prepared->Execute(bound, false);
    if (enable_timing_) {
      auto run_us =
          chrono_toc(&timer, "ExecuteSQL::Execute sub-SQL time\n", false);
      WriteJitTimingColumn(run_us);
    }
    if (duckdb_result->HasError()) {
      throw std::runtime_error("Query failed: " + duckdb_result->GetError());
    }

    // Get columns
    result.num_columns = duckdb_result->ColumnCount();
    for (size_t i = 0; i < result.num_columns; i++) {
      result.column_names.push_back(duckdb_result->ColumnName(i));
    }

    // Get rows
    result.num_rows = 0;
    while (true) {
      auto chunk = duckdb_result->Fetch();
      if (!chunk || 0 == chunk->size())
        break;

      for (size_t row = 0; row < chunk->size(); row++) {
        std::vector<std::string> row_data;
        row_data.reserve(result.num_columns);
        for (size_t col = 0; col < result.num_columns; col++) {
          row_data.push_back(chunk->GetValue(col, row).ToString());
        }
        result.rows.push_back(std::move(row_data));
        result.num_rows++;
      }
    }
  }
  return result;
}

#if IN_MEM_TMP_TABLE
void DuckDBAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name,
    bool update_temp_card) {
  std::chrono::high_resolution_clock::time_point timer;
  bool jit_active = false;
  if (enable_timing_) {
    timer = chrono_tic();
  }
  std::unique_ptr<duckdb::PreparedStatement> prepared;
#ifdef HAVE_LLVM
  // Spec-jit one-shot miss action, set by the splitter when the spec
  // for THIS sub-query was unusable: FAST_ONCE = inline TPDE compile,
  // SKIP_QJIT_ONCE = interpret (Prepare/analysis still run).
  const CompensateMissAction miss_action = compensate_miss_action_;
  compensate_miss_action_ = CompensateMissAction::NONE;
  std::unique_ptr<QjitCompiled> qjit_compiled;
  // Keepalive for a spec-jit HIT: holds the plan + IR the compiled fn's
  // metadata points into, plus the result types/names (no Prepare happens
  // on the main connection in that case).
  std::unique_ptr<QjitSpecCompiled> qjit_spec;
  bool force_interpreter = false;
  if (force_interpreter_nth_ > 0 && temp_table_name.substr(0, 4) == "temp") {
    int idx = std::atoi(temp_table_name.c_str() + 4);
    if (idx > 0 && idx % force_interpreter_nth_ == 0) {
#ifndef NDEBUG
      fprintf(stderr, "[AQP-QJIT] force-interpreter label=%s nth=%d\n",
              temp_table_name.c_str(), force_interpreter_nth_);
#endif
      force_interpreter = true;
    }
  }
  if (query_jit_ && !force_interpreter) {
    if (qjit_spec_hit_) {
      // Spec-jit HIT: the bg pool already compiled this sub-query on a spec
      // compiler. Resolve sources now (the consumed temp exists only since
      // the previous store) and skip Prepare entirely — the compiled fn
      // owns the result. Resolve failure falls through to the inline path
      // (same as a MISS).
      qjit_spec = std::move(qjit_spec_hit_);
      std::string reason;
      if (ResolveQjitSources(qjit_spec->plan, *qjit_spec->compiled, reason)) {
        qjit_compiled = std::move(qjit_spec->compiled);
        qjit_pending_ir_ = nullptr;
#ifndef NDEBUG
        fprintf(stderr, "[AQP-QJIT] spec-hit label=%s\n",
                temp_table_name.c_str());
#endif
      } else {
#ifndef NDEBUG
        fprintf(stderr, "[AQP-QJIT] spec-hit-resolve-failed:%s label=%s\n",
                reason.c_str(), temp_table_name.c_str());
#endif
        qjit_spec.reset();
      }
    }
    static const bool trace_fast =
        std::getenv("AQP_QJIT_FAST_TRACE") != nullptr;
    static const bool no_fast = std::getenv("AQP_NO_FAST_PATH") != nullptr;
    if (!qjit_compiled && qjit_pending_ir_ && !no_fast) {
      auto t_fast = trace_fast ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
      auto *fast_ir = const_cast<ir_sql_converter::AQPStmt *>(qjit_pending_ir_);
      qjit_pending_ir_ = nullptr;
      duckdb::unique_ptr<duckdb::LogicalOperator> sub_plan;
      bool ctor_plan = false;
      if (qjit_pending_use_plan_ && plan) {
        sub_plan = TakePlan();
      } else {
        // v18: TopDown Split binary subs carry no engine plan — build it from the IR
        // directly (skips the per-sub SQL parse+Optimize round trip).
        // Reject -> sub_plan null -> slow path below (v17 behavior).
        std::string ctor_reject;
        sub_plan = TryBuildBinaryPlanFromIR(*fast_ir, ctor_reject);
        ctor_plan = sub_plan != nullptr;
        if (!sub_plan && trace_fast) {
          fprintf(stderr, "[AQP-FAST-TIME] label=%s ctor-reject:%s\n",
                  temp_table_name.c_str(), ctor_reject.c_str());
        }
      }
      if (sub_plan) {
#ifndef NDEBUG
        if (trace_fast) {
          for (const auto &a : fast_ir->target_list)
            fprintf(stderr,
                    "[AQP-FAST-IR] label=%s attr t=%u c=%u name=%s type=%d "
                    "bw=%d\n",
                    temp_table_name.c_str(), a->GetTableIndex(),
                    a->GetColumnIndex(), a->GetColumnName().c_str(),
                    (int)a->GetType(), (int)a->GetBitWidth());
          if (ctor_plan)
            fprintf(stderr, "[AQP-FAST-PLAN] label=%s\n%s\n",
                    temp_table_name.c_str(), sub_plan->ToString().c_str());
        }
#endif
        sub_plan->ResolveOperatorTypes();
        auto plan_types = sub_plan->types;
        duckdb::vector<duckdb::string> plan_names;
        if (ctor_plan) {
          // The splitter's remaining SQL references temp columns by
          // ComputeColumnAlias names ({table}_{index}_{column}); the temp
          // store takes them from prepared->GetNames() = these plan_names,
          // so mirror that convention (SQL-path SELECT aliases do the same).
          // Shared with the speculative FromPlan compile so both paths name
          // the temp's columns identically.
          plan_names = CtorPlanNames(*fast_ir);
        }
        if (plan_names.size() != plan_types.size()) {
          plan_names.clear();
          for (duckdb::idx_t i = 0; i < plan_types.size(); i++)
            plan_names.push_back("col" + std::to_string(i));
        }
        prepared = conn->PrepareFromPlan(std::move(sub_plan),
                                         std::move(plan_names),
                                         std::move(plan_types));
        if (prepared && !prepared->HasError() && prepared->data &&
            prepared->data->physical_plan) {
          AnnotateBuildSides(prepared->data->physical_plan->Root(), *fast_ir,
                             /*include_chunk_scans=*/true);
          auto analysis = qjit::AnalyzeQueryJit(*fast_ir, temp_table_name);
          if (miss_action != CompensateMissAction::SKIP_QJIT_ONCE) {
            qjit_compiled = TryCompileQueryJit(
                *fast_ir, analysis, *prepared, temp_table_name,
                /*use_fast=*/miss_action == CompensateMissAction::FAST_ONCE);
          }
        }
      }
      if (!qjit_compiled) {
        prepared.reset();
      }
      if (trace_fast) {
        auto dt = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_fast).count();
        fprintf(stderr, "[AQP-FAST-TIME] label=%s fast_ms=%.3f compiled=%d\n",
                temp_table_name.c_str(), dt, !!qjit_compiled);
      }
    }
    if (!qjit_compiled && !prepared) {
      qjit_pending_ir_ = nullptr;
      auto prep = PrepareWithQueryJitAnalysis(sql, temp_table_name);
      prepared = std::move(prep.prepared);
      if (prepared && !prepared->HasError() && prep.ir &&
          miss_action != CompensateMissAction::SKIP_QJIT_ONCE) {
        qjit_compiled = TryCompileQueryJit(
            *prep.ir, prep.analysis, *prepared, temp_table_name,
            /*use_fast=*/miss_action == CompensateMissAction::FAST_ONCE);
      }
    }
    if (enable_timing_) {
      // jit_compile column: analysis + prepare + codegen time (≈0 on a spec
      // HIT — the compile happened on the bg pool), plus any spec-jit wait
      // the splitter charged (blocking on a matched-but-unfinished compile).
      auto analyze_us = chrono_toc(
          &timer, "ExecuteSQLandCreateTempTable::qjit analyze time\n", false);
      WriteJitTimingColumn(analyze_us + ConsumeSpecWaitUs());
    }
  } else if (force_interpreter) {
    qjit_pending_ir_ = nullptr;
    jit_pending_plan_.reset();
    {
      auto og = MakeOptGuard();
      prepared = conn->Prepare(sql);
    }
    if (enable_timing_)
      WriteJitTimingColumn(0);
  } else if (jit_pending_plan_) {
    jit_pending_plan_->ResolveOperatorTypes();
    auto plan_types = jit_pending_plan_->types;
    duckdb::vector<duckdb::string> plan_names;
    for (duckdb::idx_t i = 0; i < plan_types.size(); i++)
      plan_names.push_back("col" + std::to_string(i));
    prepared = conn->PrepareFromPlan(std::move(jit_pending_plan_),
                                     std::move(plan_names), std::move(plan_types));
  } else
#endif
  {
    auto og = MakeOptGuard();
    prepared = conn->Prepare(sql);
  }
  // prepared is null only on a query-jit spec HIT (no main-connection
  // Prepare; the compiled fn owns the result).
  if (prepared && prepared->HasError()) {
    throw std::runtime_error("[DuckDB] Prepare failed: " +
                             prepared->GetError());
  }

#ifdef HAVE_LLVM
  // JIT: compile filters from the pending IR and register before execution.
  if (jit_pending_ir_ && prepared && prepared->data &&
      prepared->data->physical_plan) {
    jit_active = true;
    // Reset JIT context (stale function pointers would crash if dispatched).
    // Keep jit_compiler_ alive — reusing LLJIT avoids re-init overhead.
    // Release previously compiled modules (see ExecuteSQL for the ordering
    // invariant and the multi-thread caveat).
    GetClientContext()->aqp_jit_context.reset();
    if (jit_compiler_)
      jit_compiler_->ResetModules();
#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] ExecuteSQLandCreateTempTable: reset "
                 "aqp_jit_context (compiler reused)\n";
#endif
    jit_consumed_ir_filters_.clear();
    jit_consumed_ir_joins_.clear();
    if (miss_action == CompensateMissAction::FAST_ONCE) {
      auto *comp = EnsureFastJitCompiler();
      if (comp != jit_compiler_.get())
        comp->ResetModules();
      RegisterJITImpl(prepared->data->physical_plan->Root(), *jit_pending_ir_,
                      GetClientContext(), comp, jit_consumed_ir_filters_,
                      jit_consumed_ir_joins_);
    } else {
      RegisterJIT(prepared->data->physical_plan->Root(), *jit_pending_ir_);
    }

    // Ensure JIT context always exists when JIT level is active
    auto *ctx2 = GetClientContext();
    if (!ctx2->aqp_jit_context) {
      ctx2->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
      ctx2->aqp_jit_context->flags = duckdb::AQPJIT_PIPELINE;
    }

    if (enable_timing_) {
      // Record total JIT compile time (includes Prepare + RegisterJIT)
      // plus any spec-jit wait charged by the splitter.
      auto jit_compile_time = chrono_toc(
          &timer, "ExecuteSQLandCreateTempTable::jit compile time\n", false);
      WriteJitTimingColumn(jit_compile_time + ConsumeSpecWaitUs());
    }
#ifndef NDEBUG
    if (ctx2->aqp_jit_context) {
      std::cerr << "[AQP-JIT] summary: flags=0x" << std::hex
                << ctx2->aqp_jit_context->flags << std::dec
                << " expr_fns=" << ctx2->aqp_jit_context->expr_fns.size()
                << " op_fns=" << ctx2->aqp_jit_context->op_fns.size()
                << " proj_maps=" << ctx2->aqp_jit_context->proj_col_maps.size()
                << " agg_fns=" << ctx2->aqp_jit_context->agg_fns.size()
                << " pipeline_fns="
                << ctx2->aqp_jit_context->pipeline_fns.size() << " scan_filter="
                << ctx2->aqp_jit_context->scan_filter_fns.size() << "\n";
    }
#endif
    jit_pending_ir_ = nullptr;
  } else {
    // No JIT registration for this sub-query (interpreter tune entry, or
    // cross-product guard with non-zero jit_flags). Drop the stale JIT
    // context from the previous iteration: EIDs are operator memory
    // addresses which get reused across sub-plans, so leftover compiled
    // fns would mis-dispatch on this plan and corrupt results.
    GetClientContext()->aqp_jit_context.reset();
    if (enable_timing_ && !query_jit_ && jit_compiler_) {
      WriteJitTimingColumn(0);
    }
  }
#endif

  // Always clear stale bloom filters before executing a sub-plan.
  // EIDs are derived from operator memory addresses which get reused across
  // sub-plans, so leftover BFs would match wrong scans.
#ifdef HAVE_LLVM
  {
    auto *ctx3 = GetClientContext();
    if (ctx3->aqp_jit_context) {
      ctx3->aqp_jit_context->bloom_scan_filters.clear();
    }
  }
  // Register pending bloom filters (independent of JIT).
  if (!pending_bloom_filters_.empty() && prepared && prepared->data &&
      prepared->data->physical_plan) {
    auto *ctx3 = GetClientContext();
    if (!ctx3->aqp_jit_context) {
      ctx3->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
    }
    RegisterBloomFilters(prepared->data->physical_plan->Root(),
                         GetClientContext(), pending_bloom_filters_);
    pending_bloom_filters_.clear();
  }
#endif

  // Inject temp table column stats into PhysicalHashJoin nodes to enable
  // perfect hash join (direct array lookup instead of hash table).
  if (prepared && prepared->data && prepared->data->physical_plan) {
    InjectTempTableJoinStats(prepared->data->physical_plan->Root());
  }

  // Allocate the chunk index and build deduplicated column names BEFORE
  // execution: the speculative-compile hook needs the exact identity the
  // result temp table will have (the real path below reuses the same values).
  // During plan-replay (planner is null), use a monotonic fallback index.
  duckdb::idx_t data_chunk_index;
  if (planner) {
    data_chunk_index = planner->binder->GenerateTableIndex();
  } else {
    static std::atomic<duckdb::idx_t> s_replay_idx{100000};
    data_chunk_index = s_replay_idx.fetch_add(1, std::memory_order_relaxed);
  }
  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;

#ifdef HAVE_LLVM
  // Spec HIT: no main-connection prepared statement — the payload carries
  // the spec Prepare's types/names (cross-checked against the compiled plan
  // by BuildQjitOutputDescs in the bg task).
  temp_table_types = qjit_spec ? qjit_spec->out_types : prepared->GetTypes();
  const auto &result_names =
      qjit_spec ? qjit_spec->out_names : prepared->GetNames();
#else
  temp_table_types = prepared->GetTypes();
  auto &result_names = prepared->GetNames();
#endif
  duckdb::case_insensitive_set_t used_column_names;
  std::vector<std::string> column_names;
  for (duckdb::idx_t i = 0; i < temp_table_types.size(); i++) {
    std::string column_name =
        (i < result_names.size() && !result_names[i].empty())
            ? result_names[i]
            : "col_" + std::to_string(i);

    // Handle duplicate column names
    std::string unique_column_name = column_name;
    duckdb::idx_t suffix = 1;
    while (used_column_names.count(unique_column_name) > 0) {
      unique_column_name = column_name + "_" + std::to_string(suffix);
      suffix++;
    }
    used_column_names.insert(unique_column_name);
    column_names.push_back(unique_column_name);
    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
                                  unique_column_name);
  }
  chunk_col_names_[data_chunk_index] = column_names;

#ifdef HAVE_LLVM
  if (post_prepare_hook_) {
    duckdb::idx_t est_card = 0;
    if (prepared && prepared->data && prepared->data->physical_plan)
      est_card = prepared->data->physical_plan->Root().estimated_cardinality;
    else if (qjit_spec)
      est_card = qjit_spec->est_card;
    post_prepare_hook_(temp_table_name, data_chunk_index, temp_table_types,
                       column_names, est_card, /*post_execute=*/false);
  }
#endif

  duckdb::unique_ptr<duckdb::ColumnDataCollection> subquery_result;
#ifdef HAVE_LLVM
  // §5.7: qjit results are served to later sub-queries straight from the
  // QjitTable via scan_qjit_temp — no CDC copy. Exception: a speculative
  // placeholder's bind data holds a raw pointer to the placeholder
  // collection, so that path keeps the CDC write-back (Combine below).
  bool qjit_stored = false;
  int64_t qjit_rows = 0;
  if (qjit_compiled) {
    auto qtable = std::make_unique<qjit::QjitTable>(
        qjit_compiled->out_descs, qjit_executor_->NumWorkers());
    // Progress-triggered speculation: the splitter's hook fires on this
    // thread once half of the result step's rows are done.
    if (progress_hook_) {
      qjit_executor_->EnableProgress(true, 0.5);
      qjit_executor_->SetOnProgress(
          [this, &temp_table_name, data_chunk_index,
           &column_names](const qjit::QjitExecutor::ProgressSnapshot &sn) {
            progress_hook_(temp_table_name, data_chunk_index, temp_table_types,
                           column_names, sn.rows, sn.progress, sn.ms);
          });
    }
    int64_t rows =
        qjit_executor_->Run(reinterpret_cast<QjitQueryFn>(qjit_compiled->fn),
                            qjit_compiled->srcs, qjit_compiled->ht_tuple_sizes,
                            qjit_compiled->agg_descs,
                            qjit_compiled->agg_output_cells, *qtable,
                            qjit_compiled->ht_key0_offsets,
                            qjit_compiled->params_buf);
    if (progress_hook_) {
      qjit_executor_->SetOnProgress(nullptr);
      qjit_executor_->EnableProgress(false);
    }
    if (rows >= 0) {
#ifndef NDEBUG
      fprintf(stderr, "[AQP-QJIT] exec label=%s rows=%lld\n",
              temp_table_name.c_str(), (long long)rows);
#endif
      auto pit = temp_collections_.find(temp_table_name);
      bool has_placeholder =
          pit != temp_collections_.end() && pit->second.is_placeholder;
      if (has_placeholder) {
        // temp_table_types == prepared types (or the spec payload's types on
        // a HIT, where prepared is null).
        subquery_result = QjitTableToCollection(*qtable, temp_table_types);
      } else {
        qjit_stored = true;
        qjit_rows = rows;
      }
      {
        std::unique_lock<std::shared_timed_mutex> wguard(temp_state_mutex_);
        auto &slot = qjit_temps_[temp_table_name];
        if (slot && qjit_placeholder_names_.erase(temp_table_name))
          retired_qjit_placeholders_.push_back(std::move(slot));
        slot = std::move(qtable);
      }
    } else {
      fprintf(stderr, "[AQP-QJIT] fallback:run-error(rc=%lld) label=%s\n",
              (long long)rows, temp_table_name.c_str());
    }
  }
  // A qjit placeholder whose real result is NOT a qjit table (interpreter
  // fallback / run error): drop the placeholder so the replacement scan and
  // GetOrLoadQjitTemp do not serve the empty table.
  if (!qjit_stored && qjit_placeholder_names_.erase(temp_table_name)) {
    std::unique_lock<std::shared_timed_mutex> wguard(temp_state_mutex_);
    auto qit = qjit_temps_.find(temp_table_name);
    if (qit != qjit_temps_.end()) {
      retired_qjit_placeholders_.push_back(std::move(qit->second));
      qjit_temps_.erase(qit);
    }
    qjit_temp_meta_.erase(temp_table_name);
  }
  if (!subquery_result && !qjit_stored)
#else
  if (!subquery_result)
#endif
  {
#ifdef HAVE_LLVM
    // Spec-HIT run error: no prepared statement exists — create the
    // interpreter fallback now.
    if (!prepared) {
      auto og = MakeOptGuard();
      prepared = conn->Prepare(sql);
      if (prepared->HasError())
        throw std::runtime_error("[DuckDB] Prepare failed: " +
                                 prepared->GetError());
    }
#endif
    duckdb::vector<duckdb::Value> bound_values;
    subquery_result = prepared->ExecuteRow(bound_values, false);
  }
  if (enable_timing_) {
    auto run_us = chrono_toc(
        &timer, "ExecuteSQLandCreateTempTable::Execute sub-SQL time\n", false);
    WriteJitTimingColumn(run_us);
  }

  int64_t chunk_size;
#ifdef HAVE_LLVM
  if (qjit_stored) {
    // temp_table_types already holds prepared->GetTypes() (set above); the
    // compile gate cross-checked the QjitTable layout against it.
    chunk_size = qjit_rows;
  } else
#endif
  {
    chunk_size = subquery_result->Count();
    temp_table_types = subquery_result->Types();
  }

  // Store the result. qjit-stored temps keep only binder metadata here
  // (scan_qjit_temp reads rows from qjit_temps_); everything else stores
  // the ColumnDataCollection in temp_collections_ (zero-copy). If a
  // speculative placeholder was registered for this temp, Combine the
  // result into it: a speculative Prepare's bind data holds a raw pointer to
  // the placeholder collection, so the object must survive.
#ifdef HAVE_LLVM
  if (qjit_stored) {
    QjitTempMeta meta;
    meta.column_names = std::move(column_names);
    meta.types = temp_table_types;
    std::unique_lock<std::shared_timed_mutex> wguard(temp_state_mutex_);
    qjit_temp_meta_[temp_table_name] = std::move(meta);
  } else
#endif
  {
#ifdef HAVE_LLVM
    std::unique_lock<std::shared_timed_mutex> wguard(temp_state_mutex_);
#endif
    auto existing_it = temp_collections_.find(temp_table_name);
    if (existing_it != temp_collections_.end() &&
        existing_it->second.is_placeholder) {
      existing_it->second.collection->Combine(*subquery_result);
      existing_it->second.column_names = std::move(column_names);
      existing_it->second.has_override_cardinality = false;
      existing_it->second.is_placeholder = false;
    } else {
      StoredTempResult stored;
      stored.collection = std::move(subquery_result);
      stored.column_names = std::move(column_names);
      temp_collections_[temp_table_name] = std::move(stored);
    }
  }

  temp_table_card_.emplace(temp_table_name, chunk_size);

#ifdef HAVE_LLVM
  // Zero-copy disabled: also push result into a DuckDB catalog temp
  // table so the next Prepare sees it via catalog lookup (measuring the
  // round-trip overhead that the replacement-scan path avoids).
  if (no_zero_copy_) {
    const auto &mat_names =
        qjit_stored ? qjit_temp_meta_[temp_table_name].column_names
                    : temp_collections_[temp_table_name].column_names;
    if (qjit_stored) {
      auto qit = qjit_temps_.find(temp_table_name);
      if (qit != qjit_temps_.end())
        MaterializeCatalogTempFromQjit(temp_table_name, *qit->second,
                                       temp_table_types, mat_names);
    } else {
      auto &coll = *temp_collections_[temp_table_name].collection;
      MaterializeCatalogTempTable(temp_table_name, temp_table_types,
                                  mat_names, coll);
    }
  }

  // Post-execute: the actual cardinality is known and the real temp data is
  // stored. Re-invoke the hook — this invocation launches the speculative
  // bg compile (the post-Prepare one above is trace-only).
  if (post_prepare_hook_) {
    const auto &stored_names =
        qjit_stored ? qjit_temp_meta_[temp_table_name].column_names
                    : temp_collections_[temp_table_name].column_names;
    post_prepare_hook_(temp_table_name, data_chunk_index, temp_table_types,
                       stored_names, static_cast<duckdb::idx_t>(chunk_size),
                       /*post_execute=*/true);
  }
#endif

  if (enable_timing_) {
    auto extra_materialize_time =
        chrono_toc(&timer, "Extra materialize time is\n", false);
    // save time to a file
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (extra_materialize_time / 1000.0) << ", ";
    log_file.close();

  }

}

#ifdef HAVE_LLVM
void DuckDBAdapter::ExecuteSpeculativeAndCreateTempTable(
    duckdb::PreparedStatement &prepared, duckdb::Connection &spec_conn,
    const std::string &temp_table_name, bool update_temp_card,
    const std::string &sql) {
  std::chrono::high_resolution_clock::time_point timer;
  if (enable_timing_) {
    timer = chrono_tic();
  }

  // Inject temp table stats into spec connection's physical plan
  if (prepared.data && prepared.data->physical_plan)
    InjectTempTableJoinStats(prepared.data->physical_plan->Root());

  // Allocate chunk index + dedup names BEFORE execution (see
  // ExecuteSQLandCreateTempTable) so the hook can launch the next speculation.
  // planner can be null here (TopDown Split lookahead prep skips ParseSQL): use the
  // same monotonic fallback as ExecuteSQLandCreateTempTable's replay path.
  duckdb::idx_t data_chunk_index;
  if (planner) {
    data_chunk_index = planner->binder->GenerateTableIndex();
  } else {
    static std::atomic<duckdb::idx_t> s_spec_idx{200000};
    data_chunk_index = s_spec_idx.fetch_add(1, std::memory_order_relaxed);
  }
  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;

  temp_table_types = prepared.GetTypes();
  auto &result_names = prepared.GetNames();
  duckdb::case_insensitive_set_t used_column_names;
  std::vector<std::string> column_names;
  for (duckdb::idx_t i = 0; i < temp_table_types.size(); i++) {
    std::string column_name =
        (i < result_names.size() && !result_names[i].empty())
            ? result_names[i]
            : "col_" + std::to_string(i);
    std::string unique_column_name = column_name;
    duckdb::idx_t suffix = 1;
    while (used_column_names.count(unique_column_name) > 0) {
      unique_column_name = column_name + "_" + std::to_string(suffix);
      suffix++;
    }
    used_column_names.insert(unique_column_name);
    column_names.push_back(unique_column_name);
    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
                                  unique_column_name);
  }
  chunk_col_names_[data_chunk_index] = column_names;

  if (post_prepare_hook_) {
    duckdb::idx_t est_card = 0;
    if (prepared.data && prepared.data->physical_plan)
      est_card = prepared.data->physical_plan->Root().estimated_cardinality;
    post_prepare_hook_(temp_table_name, data_chunk_index, temp_table_types,
                       column_names, est_card, /*post_execute=*/false);
  }

  duckdb::vector<duckdb::Value> bound_values;
  auto subquery_result = prepared.ExecuteRow(bound_values, false);
  if (enable_timing_) {
    auto run_us = chrono_toc(
        &timer, "ExecuteSpeculative::Execute sub-SQL time\n", false);
    WriteJitTimingColumn(run_us);
  }

  int64_t chunk_size = subquery_result->Count();
  temp_table_types = subquery_result->Types();

  auto existing_it = temp_collections_.find(temp_table_name);
  if (existing_it != temp_collections_.end() &&
      existing_it->second.is_placeholder) {
    existing_it->second.collection->Combine(*subquery_result);
    existing_it->second.column_names = std::move(column_names);
    existing_it->second.has_override_cardinality = false;
    existing_it->second.is_placeholder = false;
  } else {
    StoredTempResult stored;
    stored.collection = std::move(subquery_result);
    stored.column_names = std::move(column_names);
    temp_collections_[temp_table_name] = std::move(stored);
  }
  temp_table_card_.emplace(temp_table_name, chunk_size);

  // Post-execute hook: see ExecuteSQLandCreateTempTable.
  if (post_prepare_hook_) {
    post_prepare_hook_(temp_table_name, data_chunk_index, temp_table_types,
                       temp_collections_[temp_table_name].column_names,
                       static_cast<duckdb::idx_t>(chunk_size),
                       /*post_execute=*/true);
  }

  if (enable_timing_) {
    auto extra_materialize_time =
        chrono_toc(&timer, "Extra materialize time is\n", false);
    std::ofstream log_file;
    log_file.open(g_timing_log_name, std::ios_base::app);
    log_file << std::fixed << std::setprecision(3)
             << (extra_materialize_time / 1000.0) << ", ";
    log_file.close();
  }

}

void DuckDBAdapter::RegisterPlaceholderTemp(
    const std::string &temp_table_name,
    const duckdb::vector<duckdb::LogicalType> &types,
    const std::vector<std::string> &col_names, duckdb::idx_t est_card) {
  std::unique_lock<std::shared_timed_mutex> wguard(temp_state_mutex_);
  if (temp_collections_.count(temp_table_name) > 0)
    return; // already registered (real result or earlier placeholder)
  if (qjit_temp_meta_.count(temp_table_name) > 0)
    return; // real qjit-stored temp exists — an empty placeholder would
            // shadow it in the replacement scan (temp_collections_ is
            // checked first) and interpreter consumers would read 0 rows
  StoredTempResult stored;
  stored.collection =
      duckdb::make_uniq<duckdb::ColumnDataCollection>(*conn->context, types);
  stored.column_names = col_names;
  stored.has_override_cardinality = true;
  stored.override_cardinality = est_card == 0 ? 1 : est_card;
  stored.is_placeholder = true;
  temp_collections_[temp_table_name] = std::move(stored);
}

// qjit/qjit_annotate.cpp (the header's CollectIRJoinNodes clashes with a
// file-local helper here, so declare only what is needed).
void AnnotateUnannotatedJoinsByCard(
    ir_sql_converter::AQPStmt &ir, const storage::StoragePlan *sp,
    const std::unordered_map<std::string, int64_t> &temp_card);

std::string DuckDBAdapter::PredictQjitCacheKey(ir_sql_converter::AQPStmt &ir) {
  if (!qjit_storage_plan_ || !qjit_storage_plan_->IsLoaded())
    return "";
  EnsureJITCompiler();
  if (!jit_compiler_)
    return "";
  // Temp cardinalities: stored temps plus placeholders (estimated).
  std::unordered_map<std::string, int64_t> temp_card;
  for (const auto &kv : temp_table_card_)
    temp_card[kv.first] = static_cast<int64_t>(kv.second);
  {
    std::shared_lock<std::shared_timed_mutex> rguard(temp_state_mutex_);
    for (const auto &kv : qjit_temp_meta_)
      if (kv.second.has_override_cardinality)
        temp_card[kv.first] = static_cast<int64_t>(kv.second.override_cardinality);
  }
  AnnotateUnannotatedJoinsByCard(ir, qjit_storage_plan_, temp_card);
  qjit::JITHubIR plan;
  std::string reason;
  if (!qjit::BuildExecutionSteps(ir, plan, reason, range_guard_, block_skip_,
                                 membership_preprobe_))
    return "";
  return jit_compiler_->QueryStepsCacheKey(plan);
}

// Storage dtype of a qjit temp column for a DuckDB type; false when the qjit
// store has no representation (the temp then stays a ColumnDataCollection).
// Single source for the real store (CollectionToQjitTable) and the
// speculative placeholder, so the two can never disagree.
static bool QjitDtypeForLogicalType(const duckdb::LogicalType &type,
                                    int32_t &dt) {
  switch (type.id()) {
  case duckdb::LogicalTypeId::INTEGER:
  case duckdb::LogicalTypeId::DATE:
  case duckdb::LogicalTypeId::SMALLINT:
  case duckdb::LogicalTypeId::TINYINT:
    dt = AQP_DTYPE_INT32;
    return true;
  case duckdb::LogicalTypeId::BIGINT:
  case duckdb::LogicalTypeId::TIMESTAMP:
  case duckdb::LogicalTypeId::TIMESTAMP_TZ:
    dt = AQP_DTYPE_INT64;
    return true;
  case duckdb::LogicalTypeId::DECIMAL:
    dt = duckdb::DecimalType::GetWidth(type) <= 9 ? AQP_DTYPE_INT32
                                                  : AQP_DTYPE_INT64;
    return true;
  case duckdb::LogicalTypeId::VARCHAR:
    dt = AQP_DTYPE_VARCHAR;
    return true;
  default:
    return false;
  }
}

bool DuckDBAdapter::RegisterQjitPlaceholderTemp(
    const std::string &temp_table_name,
    const duckdb::vector<duckdb::LogicalType> &types,
    const std::vector<std::string> &col_names, duckdb::idx_t est_card) {
  std::vector<qjit::QjitTable::ColumnDesc> descs;
  descs.reserve(types.size());
  for (size_t i = 0; i < types.size(); i++) {
    int32_t dt;
    if (!QjitDtypeForLogicalType(types[i], dt))
      return false;
    descs.push_back({dt, i < col_names.size() ? col_names[i]
                                               : "col_" + std::to_string(i)});
  }
  std::unique_lock<std::shared_timed_mutex> wguard(temp_state_mutex_);
  if (temp_collections_.count(temp_table_name) > 0 ||
      qjit_temp_meta_.count(temp_table_name) > 0)
    return true;
  QjitTempMeta meta;
  meta.column_names = col_names;
  meta.types = types;
  meta.has_override_cardinality = true;
  meta.override_cardinality = est_card == 0 ? 1 : est_card;
  qjit_temp_meta_[temp_table_name] = std::move(meta);
  qjit_temps_[temp_table_name] =
      std::make_unique<qjit::QjitTable>(std::move(descs), 1u);
  qjit_placeholder_names_.insert(temp_table_name);
  return true;
}
#endif

#else
void DuckDBAdapter::ExecuteSQLandCreateTempTable(
    const std::string &sql, const std::string &temp_table_name,
    bool update_temp_card, bool enable_timing) {
  auto og = MakeOptGuard();
  auto prepared = conn->Prepare(sql);
  if (prepared->HasError()) {
    throw std::runtime_error("[DuckDB] Prepare failed: " +
                             prepared->GetError());
  }
  duckdb::vector<duckdb::Value> bound_values;
  auto subquery_result = prepared->ExecuteRow(bound_values, false);
  int64_t chunk_size = subquery_result->Count();
  auto data_chunk_index = planner->binder->GenerateTableIndex();

  intermediate_table_map[data_chunk_index] = temp_table_name;
  temp_table_index_ = data_chunk_index;

  auto context = GetClientContext();
  // create a table from data chunk
  auto &catalog = duckdb::Catalog::GetCatalog(*context, TEMP_CATALOG);
  auto &types = subquery_result->Types();
  auto info = duckdb::make_uniq<duckdb::CreateTableInfo>(
      TEMP_CATALOG, DEFAULT_SCHEMA, temp_table_name);
  info->temporary = true;
  info->on_conflict = duckdb::OnCreateConflict::ERROR_ON_CONFLICT;

  // Use actual column names from SQL result (matches alias convention)
  auto &result_names = prepared->GetNames();
  duckdb::case_insensitive_set_t used_column_names;
  for (duckdb::idx_t i = 0; i < types.size(); i++) {
    std::string column_name =
        (i < result_names.size() && !result_names[i].empty())
            ? result_names[i]
            : "col_" + std::to_string(i);

    // Handle duplicate column names
    std::string unique_column_name = column_name;
    duckdb::idx_t suffix = 1;
    while (used_column_names.count(unique_column_name) > 0) {
      unique_column_name = column_name + "_" + std::to_string(suffix);
      suffix++;
    }
    used_column_names.insert(unique_column_name);
    info->columns.AddColumn(
        duckdb::ColumnDefinition(unique_column_name, types[i]));
    chunk_col_names_[data_chunk_index].push_back(unique_column_name);
    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
                                  std::move(unique_column_name));
  }

  // Begin transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  auto created_table = catalog.CreateTable(*context, std::move(info));
  auto &created_table_entry = created_table->Cast<duckdb::TableCatalogEntry>();
  temp_table_card_.emplace(temp_table_name, chunk_size);
  const duckdb::vector<duckdb::unique_ptr<duckdb::BoundConstraint>>
      bound_constraints = planner->binder->BindConstraints(created_table_entry);

  auto &storage = created_table_entry.GetStorage();
  storage.LocalAppend(created_table_entry, *context, *subquery_result,
                      bound_constraints, nullptr);
  //  storage.LocalAppend(created_table_entry, *context, *subquery_result);

  // Commit transaction if in auto-commit mode
  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }
}
#endif

void DuckDBAdapter::CreateTempTable(const std::string &table_name,
                                    const QueryResult &result) {
  //  auto context = GetClientContext();
  //  auto &catalog = duckdb::Catalog::GetCatalog(*context, TEMP_CATALOG);
  //  auto info = duckdb::make_uniq<duckdb::CreateTableInfo>(TEMP_CATALOG,
  //  DEFAULT_SCHEMA, chunk_name); info->temporary = true; info->on_conflict =
  //  duckdb::OnCreateConflict::REPLACE_ON_CONFLICT; auto &types =
  //  result.Types(); auto data_chunk_index =
  //  planner.binder->GenerateTableIndex();
  //
  //  duckdb::case_insensitive_set_t used_column_names;
  //  for (size_t i = 0; i < result.column_names.size(); i++) {
  //    std::string column_name = result.column_names[i];
  //
  //    // Handle duplicate column names
  //    std::string unique_column_name = column_name;
  //    duckdb::idx_t suffix = 1;
  //    while (used_column_names.count(unique_column_name) > 0) {
  //      unique_column_name = column_name + "_" + std::to_string(suffix);
  //      suffix++;
  //    }
  //    used_column_names.insert(unique_column_name);
  //    info->columns.AddColumn(ColumnDefinition(unique_column_name, types[i]));
  //    table_column_mappings.emplace(std::make_pair(data_chunk_index, i),
  //    std::move(unique_column_name));
  //  }
  //
  //  auto created_table = catalog.CreateTable(*context, std::move(info));
  //  auto &created_table_entry =
  //  created_table->Cast<duckdb::TableCatalogEntry>(); int64_t
  //  created_table_size = subquery_result->Count();
  //  temp_table_card_.emplace(intermediate_table_name, created_table_size);
  //
  //  auto &storage = created_table_entry.GetStorage();
  //  storage.LocalAppend(created_table_entry, *context, *subquery_result);
}

#if IN_MEM_TMP_TABLE
void DuckDBAdapter::DropTempTable(const std::string &table_name) {
  temp_collections_.erase(table_name);
  qjit_temp_meta_.erase(table_name);
#ifdef HAVE_LLVM
  if (qjit_placeholder_names_.erase(table_name)) {
    auto qit = qjit_temps_.find(table_name);
    if (qit != qjit_temps_.end())
      retired_qjit_placeholders_.push_back(std::move(qit->second));
  }
  qjit_temps_.erase(table_name);
#endif
  if (no_zero_copy_ && bidir_conn_) {
    try {
      bidir_conn_->Query("DROP TABLE IF EXISTS temp." + table_name);
    } catch (...) {}
  }
}

bool DuckDBAdapter::TempTableExists(const std::string &table_name) {
  return temp_collections_.count(table_name) > 0 ||
         qjit_temp_meta_.count(table_name) > 0;
}
#else
void DuckDBAdapter::DropTempTable(const std::string &chunk_name) {
  ExecuteSQL("DROP TABLE IF EXISTS " + chunk_name);
}

bool DuckDBAdapter::TempTableExists(const std::string &chunk_name) {
  try {
    auto result = ExecuteSQL(
        "SELECT count(*) FROM information_schema.tables WHERE chunk_name = '" +
        chunk_name + "'");
    return result.num_rows > 0 && result.rows[0][0] != "0";
  } catch (...) {
    return false;
  }
}
#endif

uint64_t
DuckDBAdapter::GetTempTableCardinality(const std::string &temp_table_name) {
  if (temp_table_card_.count(temp_table_name)) {
    return temp_table_card_[temp_table_name];
  }
  return 0; // Default if not found
}

#if IN_MEM_TMP_TABLE
void DuckDBAdapter::SetTempTableCardinality(const std::string &temp_table_name,
                                            uint64_t cardinality) {
  // Set override cardinality on the stored collection
  auto it = temp_collections_.find(temp_table_name);
  if (it != temp_collections_.end()) {
    it->second.has_override_cardinality = true;
    it->second.override_cardinality = cardinality;
  }
  auto qit = qjit_temp_meta_.find(temp_table_name);
  if (qit != qjit_temp_meta_.end()) {
    qit->second.has_override_cardinality = true;
    qit->second.override_cardinality = cardinality;
  }

  // Update temp_table_card_ for consistency
  temp_table_card_[temp_table_name] = cardinality;

#ifndef NDEBUG
  std::cout << "[DuckDB] SetTempTableCardinality: " << temp_table_name << " = "
            << cardinality << std::endl;
#endif
}
#else
void DuckDBAdapter::SetTempTableCardinality(const std::string &temp_table_name,
                                            uint64_t cardinality) {
  auto context = GetClientContext();

  if (context->transaction.IsAutoCommit()) {
    context->transaction.BeginTransaction();
  }

  auto &catalog = duckdb::Catalog::GetCatalog(*context, TEMP_CATALOG);
  auto &table_entry = catalog.GetEntry<duckdb::TableCatalogEntry>(
      *context, DEFAULT_SCHEMA, temp_table_name);
  auto &storage = table_entry.GetStorage();
  // Inject the override cardinality into DataTableInfo so that
  // DataTable::GetTotalRows() (and thus TableScanCardinality) returns
  // this value instead of the real row count.
  storage.GetDataTableInfo()->cardinality_override.store(cardinality);

  if (context->transaction.IsAutoCommit()) {
    context->transaction.Commit();
  }

  temp_table_card_[temp_table_name] = cardinality;

#ifndef NDEBUG
  std::cout << "[DuckDB] SetTempTableCardinality: " << temp_table_name << " = "
            << cardinality << std::endl;
#endif
}
#endif

std::string DuckDBAdapter::GetColumnName(const std::string &table_name,
                                         unsigned col_idx) {
  try {
    auto result = conn->Query("SELECT column_name FROM information_schema.columns "
                              "WHERE table_name = '" + table_name + "' "
                              "ORDER BY ordinal_position");
    if (!result->HasError()) {
      unsigned row = 0;
      while (true) {
        auto chunk = result->Fetch();
        if (!chunk || chunk->size() == 0) break;
        for (duckdb::idx_t i = 0; i < chunk->size(); i++) {
          if (row == col_idx) {
            return chunk->GetValue(0, i).ToString();
          }
          row++;
        }
      }
    }
  } catch (...) {
  }
  return "";
}

std::unordered_map<size_t, std::pair<int64_t, int64_t>>
DuckDBAdapter::GetTempTableMinMax(
    const std::string &temp_table_name,
    const std::vector<std::string> & /*column_names*/,
    const std::vector<ir_sql_converter::SimplestVarType> & /*column_types*/) {
  std::unordered_map<size_t, std::pair<int64_t, int64_t>> result;
#if IN_MEM_TMP_TABLE
#ifdef HAVE_LLVM
  // qjit-stored temps (§5.7) have no CDC; compute from the QjitTable.
  if (!temp_collections_.count(temp_table_name)) {
    auto qit = qjit_temps_.find(temp_table_name);
    if (qit != qjit_temps_.end()) {
      const qjit::QjitTable &t = *qit->second;
      for (size_t col_idx = 0; col_idx < t.NumCols(); col_idx++) {
        const int32_t dt = t.Col(col_idx).dtype;
        if (dt != AQP_DTYPE_INT32 && dt != AQP_DTYPE_INT64)
          continue;
        int64_t min_val = std::numeric_limits<int64_t>::max();
        int64_t max_val = std::numeric_limits<int64_t>::min();
        bool found = false;
        for (uint64_t r = 0; r < t.NumRows(); r++) {
          if (!t.ValueValid(col_idx, r))
            continue;
          int64_t v = dt == AQP_DTYPE_INT32 ? t.GetI32(col_idx, r)
                                            : t.GetI64(col_idx, r);
          if (v < min_val) min_val = v;
          if (v > max_val) max_val = v;
          found = true;
        }
        if (found)
          result[col_idx] = {min_val, max_val};
      }
      return result;
    }
  }
#endif
  auto it = temp_collections_.find(temp_table_name);
  if (it == temp_collections_.end() || !it->second.collection)
    return result;

  auto &collection = *it->second.collection;
  auto &types = collection.Types();

  for (size_t col_idx = 0; col_idx < types.size(); col_idx++) {
    auto pt = types[col_idx].InternalType();
    if (pt != duckdb::PhysicalType::INT32 && pt != duckdb::PhysicalType::INT64)
      continue;
    // DATE/DECIMAL/TIME/TIMESTAMP are physically INT32/INT64 but their raw
    // values must not be emitted as integer literals in zone-map predicates
    if (!types[col_idx].IsIntegral())
      continue;

    int64_t min_val = std::numeric_limits<int64_t>::max();
    int64_t max_val = std::numeric_limits<int64_t>::min();
    bool found = false;

    duckdb::ColumnDataScanState scan_state;
    collection.InitializeScan(scan_state);
    duckdb::DataChunk chunk;
    chunk.Initialize(*GetClientContext(), types);
    while (collection.Scan(scan_state, chunk)) {
      if (chunk.size() == 0)
        break;
      chunk.Flatten();
      auto &vec = chunk.data[col_idx];
      auto &validity = duckdb::FlatVector::Validity(vec);
      if (pt == duckdb::PhysicalType::INT32) {
        auto *data = duckdb::FlatVector::GetData<int32_t>(vec);
        for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
          if (validity.RowIsValid(i)) {
            int64_t v = data[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            found = true;
          }
        }
      } else {
        auto *data = duckdb::FlatVector::GetData<int64_t>(vec);
        for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
          if (validity.RowIsValid(i)) {
            int64_t v = data[i];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
            found = true;
          }
        }
      }
      chunk.Reset();
    }
    if (found)
      result[col_idx] = {min_val, max_val};
  }
#endif
  return result;
}

std::vector<int64_t>
DuckDBAdapter::GetTempTableDistinctValues(const std::string &temp_table_name,
                                          size_t col_idx, size_t max_distinct) {
  std::vector<int64_t> result;
#if IN_MEM_TMP_TABLE
  auto it = temp_collections_.find(temp_table_name);
  if (it == temp_collections_.end() || !it->second.collection)
    return result;

  auto &collection = *it->second.collection;
  auto &types = collection.Types();
  if (col_idx >= types.size())
    return result;

  auto pt = types[col_idx].InternalType();
  if (pt != duckdb::PhysicalType::INT32 && pt != duckdb::PhysicalType::INT64)
    return result;

  std::unordered_set<int64_t> distinct_set;
  duckdb::ColumnDataScanState scan_state;
  collection.InitializeScan(scan_state);
  duckdb::DataChunk chunk;
  chunk.Initialize(*GetClientContext(), types);
  bool exceeded = false;
  while (collection.Scan(scan_state, chunk)) {
    if (chunk.size() == 0) break;
    chunk.Flatten();
    auto &vec = chunk.data[col_idx];
    auto &validity = duckdb::FlatVector::Validity(vec);
    if (pt == duckdb::PhysicalType::INT32) {
      auto *data = duckdb::FlatVector::GetData<int32_t>(vec);
      for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
        if (validity.RowIsValid(i)) {
          distinct_set.insert(static_cast<int64_t>(data[i]));
          if (distinct_set.size() > max_distinct) { exceeded = true; break; }
        }
      }
    } else {
      auto *data = duckdb::FlatVector::GetData<int64_t>(vec);
      for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
        if (validity.RowIsValid(i)) {
          distinct_set.insert(data[i]);
          if (distinct_set.size() > max_distinct) { exceeded = true; break; }
        }
      }
    }
    chunk.Reset();
    if (exceeded) break;
  }
  if (!exceeded) {
    result.assign(distinct_set.begin(), distinct_set.end());
  }
#endif
  return result;
}

uint64_t
DuckDBAdapter::GetBaseTableCardinality(const std::string &table_name) {
  static std::unordered_map<std::string, uint64_t> cache;
  auto it = cache.find(table_name);
  if (it != cache.end())
    return it->second;

  try {
    auto duck_result = conn->Query("SELECT COUNT(*) FROM \"" + table_name + "\"");
    if (duck_result && !duck_result->HasError()) {
      auto chunk = duck_result->Fetch();
      if (chunk && chunk->size() > 0) {
        uint64_t rows = chunk->GetValue(0, 0).GetValue<uint64_t>();
        cache[table_name] = rows;
        return rows;
      }
    }
  } catch (...) {}
  cache[table_name] = 0;
  return 0;
}

#ifdef HAVE_LLVM
// Bloom filter over a qjit-stored temp (§5.7: no CDC to scan). Same sizing
// and bit-insert scheme as the ColumnDataCollection variant below.
static DuckDBAdapter::BloomFilterInfo
BuildBloomFilterFromQjitTable(const qjit::QjitTable &table, size_t col_idx) {
  DuckDBAdapter::BloomFilterInfo info;
  if (col_idx >= table.NumCols())
    return info;
  const int32_t dt = table.Col(col_idx).dtype;
  if (dt != AQP_DTYPE_INT32 && dt != AQP_DTYPE_INT64)
    return info;

  uint64_t temp_card = table.NumRows();

  constexpr uint64_t kMinBits = 512;
  constexpr uint64_t kBitsPerKey = 12;
  constexpr uint64_t kMaxSectors = (1ULL << 26);
  uint64_t min_bits = std::max(kMinBits, temp_card * kBitsPerKey);
  uint64_t num_sectors = 1;
  while (num_sectors < (min_bits >> 6)) num_sectors <<= 1;
  if (num_sectors > kMaxSectors) num_sectors = kMaxSectors;

  info.bf_data.resize(num_sectors, 0);
  info.bitmask = num_sectors - 1;

  constexpr uint64_t kShiftMask = 0x3F3F3F3F3F3F3F3F;
  constexpr int kNBits = 4;

  auto insert_one = [&](uint64_t hash) {
    uint64_t offset = hash & info.bitmask;
    uint64_t shifts = hash & kShiftMask;
    auto shifts_8 = reinterpret_cast<const uint8_t *>(&shifts);
    uint64_t mask = 0;
    for (int i = 8 - kNBits; i < 8; i++) {
      mask |= (1ULL << shifts_8[i]);
    }
    info.bf_data[offset] |= mask;
  };

  for (uint64_t r = 0; r < table.NumRows(); r++) {
    if (!table.ValueValid(col_idx, r))
      continue;
    if (dt == AQP_DTYPE_INT32)
      insert_one(duckdb::Hash<int32_t>(table.GetI32(col_idx, r)));
    else
      insert_one(duckdb::Hash<int64_t>(table.GetI64(col_idx, r)));
  }
  return info;
}
#endif

DuckDBAdapter::BloomFilterInfo
DuckDBAdapter::BuildBloomFilter(const std::string &temp_table_name,
                                size_t col_idx,
                                uint64_t temp_card) {
  BloomFilterInfo info;
  (void)temp_card;
#if IN_MEM_TMP_TABLE
#ifdef HAVE_LLVM
  if (!temp_collections_.count(temp_table_name)) {
    auto qit = qjit_temps_.find(temp_table_name);
    if (qit != qjit_temps_.end())
      return BuildBloomFilterFromQjitTable(*qit->second, col_idx);
  }
#endif
  auto it = temp_collections_.find(temp_table_name);
  if (it == temp_collections_.end() || !it->second.collection)
    return info;
  info = BuildBloomFilter(*it->second.collection, col_idx, *GetClientContext());
#endif
  return info;
}

DuckDBAdapter::BloomFilterInfo
DuckDBAdapter::BuildBloomFilter(duckdb::ColumnDataCollection &collection,
                                size_t col_idx, duckdb::ClientContext &ctx) {
  BloomFilterInfo info;
  auto &types = collection.Types();
  if (col_idx >= types.size())
    return info;

  auto pt = types[col_idx].InternalType();
  if (pt != duckdb::PhysicalType::INT32 && pt != duckdb::PhysicalType::INT64)
    return info;

  uint64_t temp_card = collection.Count();

  // Size the Bloom filter: 12 bits per key, rounded up to power-of-2 sectors.
  // Each sector is 64 bits. This gives ~1.5% false positive rate with 4 hash bits.
  constexpr uint64_t kMinBits = 512;
  constexpr uint64_t kBitsPerKey = 12;
  constexpr uint64_t kMaxSectors = (1ULL << 26);
  uint64_t min_bits = std::max(kMinBits, temp_card * kBitsPerKey);
  uint64_t num_sectors = 1;
  while (num_sectors < (min_bits >> 6)) num_sectors <<= 1;
  if (num_sectors > kMaxSectors) num_sectors = kMaxSectors;

  info.bf_data.resize(num_sectors, 0);
  info.bitmask = num_sectors - 1;

  constexpr uint64_t kShiftMask = 0x3F3F3F3F3F3F3F3F;
  constexpr int kNBits = 4;

  auto insert_one = [&](uint64_t hash) {
    uint64_t offset = hash & info.bitmask;
    uint64_t shifts = hash & kShiftMask;
    auto shifts_8 = reinterpret_cast<const uint8_t *>(&shifts);
    uint64_t mask = 0;
    for (int i = 8 - kNBits; i < 8; i++) {
      mask |= (1ULL << shifts_8[i]);
    }
    info.bf_data[offset] |= mask;
  };

  duckdb::ColumnDataScanState scan_state;
  collection.InitializeScan(scan_state);
  duckdb::DataChunk chunk;
  chunk.Initialize(ctx, types);
  while (collection.Scan(scan_state, chunk)) {
    if (chunk.size() == 0) break;
    chunk.Flatten();
    auto &vec = chunk.data[col_idx];
    auto &validity = duckdb::FlatVector::Validity(vec);
    if (pt == duckdb::PhysicalType::INT32) {
      auto *data = duckdb::FlatVector::GetData<int32_t>(vec);
      for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
        if (validity.RowIsValid(i)) {
          insert_one(duckdb::Hash<int32_t>(data[i]));
        }
      }
    } else {
      auto *data = duckdb::FlatVector::GetData<int64_t>(vec);
      for (duckdb::idx_t i = 0; i < chunk.size(); i++) {
        if (validity.RowIsValid(i)) {
          insert_one(duckdb::Hash<int64_t>(data[i]));
        }
      }
    }
    chunk.Reset();
  }
  return info;
}

DuckDBAdapter::BloomFilterInfo
DuckDBAdapter::BuildBloomFilterFromCollection(
    duckdb::ColumnDataCollection &collection,
    size_t col_idx, duckdb::ClientContext &ctx) {
  BloomFilterInfo info;
  auto &types = collection.Types();
  if (col_idx >= types.size()) return info;

  uint64_t temp_card = collection.Count();
  if (temp_card == 0) return info;

  constexpr uint64_t kMinBits = 512;
  constexpr uint64_t kBitsPerKey = 12;
  constexpr uint64_t kMaxSectors = (1ULL << 26);
  uint64_t min_bits = std::max(kMinBits, temp_card * kBitsPerKey);
  uint64_t num_sectors = 1;
  while (num_sectors < (min_bits >> 6)) num_sectors <<= 1;
  if (num_sectors > kMaxSectors) num_sectors = kMaxSectors;

  info.bf_data.resize(num_sectors, 0);
  info.bitmask = num_sectors - 1;

  constexpr uint64_t kShiftMask = 0x3F3F3F3F3F3F3F3F;
  constexpr int kNBits = 4;

  auto insert_one = [&](uint64_t hash) {
    uint64_t offset = hash & info.bitmask;
    uint64_t shifts = hash & kShiftMask;
    auto shifts_8 = reinterpret_cast<const uint8_t *>(&shifts);
    uint64_t mask = 0;
    for (int i = 8 - kNBits; i < 8; i++)
      mask |= (1ULL << shifts_8[i]);
    info.bf_data[offset] |= mask;
  };

  duckdb::ColumnDataScanState scan_state;
  collection.InitializeScan(scan_state);
  duckdb::DataChunk chunk;
  chunk.Initialize(ctx, types);
  duckdb::Vector hashes(duckdb::LogicalType::HASH);
  while (collection.Scan(scan_state, chunk)) {
    if (chunk.size() == 0) break;
    duckdb::VectorOperations::Hash(chunk.data[col_idx], hashes, chunk.size());
    auto hash_data = duckdb::FlatVector::GetData<duckdb::hash_t>(hashes);
    for (duckdb::idx_t i = 0; i < chunk.size(); i++)
      insert_one(hash_data[i]);
    chunk.Reset();
  }
  return info;
}

// todo: if the middleware cannot access the duckdb's source code, it should run
//  `EXPLAIN` as the other engines
std::pair<double, double>
DuckDBAdapter::GetEstimatedCost(const std::string &sql) {
  // Use EXPLAIN to get estimated cost and rows
  // DuckDB's EXPLAIN output format: we'll parse the cardinality from it
  try {

    auto context = GetClientContext();

    // Begin transaction if in auto-commit mode
    if (context->transaction.IsAutoCommit()) {
      context->transaction.BeginTransaction();
    }

    auto cardest_plan = conn->ExtractPlan(sql);
    if (!cardest_plan) {
      throw std::runtime_error("couldn't extract plan!");
    }

    double estimated_rows = (double)cardest_plan->estimated_cardinality;
    double estimated_cost = estimated_rows;

    // Commit transaction if we started one
    if (context->transaction.IsAutoCommit()) {
      context->transaction.Commit();
    }

    return {estimated_cost, estimated_rows};

  } catch (const std::exception &e) {
    std::cerr << "[DuckDB] GetEstimatedCost exception: " << e.what()
              << std::endl;
    return {std::numeric_limits<double>::max(),
            std::numeric_limits<double>::max()};
  }
}

std::string DuckDBAdapter::ExplainAnalyze(const std::string &sql) {
  // DuckDB EXPLAIN ANALYZE returns (explain_key, explain_value); the rendered
  // profiling tree is the last column. Reuse ExecuteSQL and join those values.
  try {
    QueryResult r = ExecuteSQL("EXPLAIN ANALYZE " + StripSqlTerminator(sql));
    std::string plan_text;
    for (const auto &row : r.rows) {
      if (!row.empty())
        plan_text += row.back();
      plan_text += "\n";
    }

    // DuckDB prepends a "Query Profiling Information" box and echoes the whole
    // query (one very wide line) before the tree. Drop everything before the
    // "Total Time" box so the box shows just the operator tree with timings.
    // The "Total Time" content line sits two lines below its box's top border,
    // so back up two line-starts to keep the box intact.
    size_t tt = plan_text.find("Total Time");
    if (tt != std::string::npos) {
      size_t pos = plan_text.rfind('\n', tt);
      for (int k = 0; k < 2 && pos != std::string::npos && pos > 0; ++k)
        pos = plan_text.rfind('\n', pos - 1);
      plan_text = plan_text.substr(pos == std::string::npos ? 0 : pos + 1);
    }

    return plan_text;
  } catch (const std::exception &e) {
    return std::string("EXPLAIN ANALYZE failed: ") + e.what();
  }
}

void DuckDBAdapter::CleanUp() {
#if IN_MEM_TMP_TABLE
  temp_collections_.clear();
  qjit_temp_meta_.clear();
#endif
#ifdef HAVE_LLVM
  qjit_temps_.clear();
  qjit_placeholder_names_.clear();
  retired_qjit_placeholders_.clear();
  qjit_spec_hit_.reset();
#endif
  plan.reset();
  planner.reset();
  conn.reset();
  db.reset();
  table_column_mappings.clear();
  intermediate_table_map.clear();
  temp_table_card_.clear();
}

void DuckDBAdapter::ResetQueryState() {
#if IN_MEM_TMP_TABLE
  temp_collections_.clear();
  qjit_temp_meta_.clear();
#endif
#ifdef HAVE_LLVM
  qjit_temps_.clear();
  qjit_placeholder_names_.clear();
  retired_qjit_placeholders_.clear();
  qjit_spec_hit_.reset();
  spec_wait_extra_us_ = 0;
#endif
  kernel_temp_tables_.clear();
  plan.reset();
  planner.reset();
  table_column_mappings.clear();
  intermediate_table_map.clear();
  chunk_col_names_.clear();
  temp_table_card_.clear();
  subquery_index = 0;
  temp_table_index_ = 0;
#ifdef HAVE_LLVM
  jit_pending_ir_ = nullptr;
  qjit_pending_ir_ = nullptr;
  owned_jit_ir_.reset();
  jit_pending_plan_.reset();
  jit_consumed_ir_filters_.clear();
  jit_consumed_ir_joins_.clear();
  temp_col_ranges_.clear();
  pending_bloom_filters_.clear();
  if (jit_compiler_ && !JitResident())
    jit_compiler_->ResetModules();
  jit_compiler_cache_.clear();
  fast_jit_compiler_.reset();
  fast_qjit_syms_registered_ = false;
  auto ctx = GetClientContext();
  if (ctx)
    ctx->aqp_jit_context.reset();
#endif
}

void DuckDBAdapter::LoadTablesFromCSV(const std::string &schema_path,
                                       const std::string &csv_dir) {
  std::ifstream schema_file(schema_path);
  if (!schema_file.is_open())
    throw std::runtime_error("Cannot open schema file: " + schema_path);

  // Read the full schema file and extract table names + CREATE TABLE statements
  std::string schema_sql((std::istreambuf_iterator<char>(schema_file)),
                          std::istreambuf_iterator<char>());

  std::vector<std::string> table_names;
  std::string::size_type pos = 0;
  while ((pos = schema_sql.find("CREATE TABLE", pos)) != std::string::npos) {
    auto start = pos + 13;
    while (start < schema_sql.size() && schema_sql[start] == ' ')
      start++;
    auto end = start;
    while (end < schema_sql.size() && schema_sql[end] != ' ' && schema_sql[end] != '(')
      end++;
    if (end > start)
      table_names.push_back(schema_sql.substr(start, end - start));
    pos = end;
  }

#ifndef NDEBUG
  std::cerr << "[AQP] Loading " << table_names.size()
            << " tables from CSV into memory..." << std::endl;
#endif
  auto load_start = std::chrono::high_resolution_clock::now();

  // Create tables with proper schema (types, constraints)
  auto create_result = conn->Query(schema_sql);
  if (create_result->HasError())
    throw std::runtime_error("Schema creation failed: " +
                             create_result->GetError());

  // Load data from CSV into each table
  for (const auto &table_name : table_names) {
    std::string csv_path = csv_dir;
    if (!csv_path.empty() && csv_path.back() != '/')
      csv_path += '/';
    csv_path += table_name + ".csv";

    std::string sql = "COPY " + table_name + " FROM '" + csv_path +
                      "' (HEADER, DELIMITER ',', QUOTE '\"', ESCAPE '\\')";
    auto result = conn->Query(sql);
    if (result->HasError())
      throw std::runtime_error("CSV load failed for " + table_name + ": " +
                               result->GetError());
  }

  auto load_end = std::chrono::high_resolution_clock::now();
  auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                     load_end - load_start).count();
#ifndef NDEBUG
  std::cerr << "[AQP] CSV loading complete in " << load_ms << " ms"
            << std::endl;
#endif
}

duckdb::ClientContext *DuckDBAdapter::GetClientContext() {
  return conn->context.get();
}

#ifdef HAVE_LLVM
// Recursively find the first AQPStmt node of type FilterNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstFilterNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::FilterNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *f = FindFirstFilterNode(child.get()))
      return f;
  return nullptr;
}

// Recursively find the first ProjectionNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstProjectionNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ProjectionNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *p = FindFirstProjectionNode(child.get()))
      return p;
  return nullptr;
}

// Recursively find the first HashNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstHashNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::HashNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *h = FindFirstHashNode(child.get()))
      return h;
  return nullptr;
}

// Recursively find the first JoinNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstJoinNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *j = FindFirstJoinNode(child.get()))
      return j;
  return nullptr;
}

// Walk the physical plan downward to find a leaf PhysicalTableScan.
static duckdb::PhysicalTableScan *
FindBuildSideTableScan(duckdb::PhysicalOperator &op) {
  if (op.type == duckdb::PhysicalOperatorType::TABLE_SCAN)
    return &op.Cast<duckdb::PhysicalTableScan>();
  for (auto &child : op.children)
    if (auto *s = FindBuildSideTableScan(child.get()))
      return s;
  return nullptr;
}

static duckdb::PhysicalColumnDataScan *
FindBuildSideColumnDataScan(duckdb::PhysicalOperator &op) {
  if (op.type == duckdb::PhysicalOperatorType::COLUMN_DATA_SCAN)
    return &op.Cast<duckdb::PhysicalColumnDataScan>();
  for (auto &child : op.children)
    if (auto *s = FindBuildSideColumnDataScan(child.get()))
      return s;
  return nullptr;
}

// ---------------------------------------------------------------------------
// Query-jit (--jit-level=query): build-side annotation + Phase 0 analysis.
// ---------------------------------------------------------------------------

// Collect logical_table_index of every PhysicalTableScan in a subtree.
static void CollectPhysScanIndices(duckdb::PhysicalOperator &op,
                                   std::unordered_set<unsigned int> &out) {
  if (op.type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
    auto &scan = op.Cast<duckdb::PhysicalTableScan>();
    out.insert(static_cast<unsigned int>(scan.logical_table_index));
  }
  for (auto &child : op.children)
    CollectPhysScanIndices(child.get(), out);
}

static void CollectPhysScanIndicesWithChunk(duckdb::PhysicalOperator &op,
                                            std::unordered_set<unsigned int> &out) {
  if (op.type == duckdb::PhysicalOperatorType::TABLE_SCAN) {
    auto &scan = op.Cast<duckdb::PhysicalTableScan>();
    out.insert(static_cast<unsigned int>(scan.logical_table_index));
  } else if (op.type == duckdb::PhysicalOperatorType::COLUMN_DATA_SCAN) {
    auto &scan = op.Cast<duckdb::PhysicalColumnDataScan>();
    if (scan.logical_table_index != duckdb::DConstants::INVALID_INDEX)
      out.insert(static_cast<unsigned int>(scan.logical_table_index));
  }
  for (auto &child : op.children)
    CollectPhysScanIndicesWithChunk(child.get(), out);
}

// Collect table_index of every ScanNode/ChunkNode in an IR subtree.
static void CollectIRSourceIndices(const ir_sql_converter::AQPStmt *ir,
                                   std::unordered_set<unsigned int> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode)
    out.insert(
        static_cast<const ir_sql_converter::SimplestScan *>(ir)->GetTableIndex());
  else if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ChunkNode)
    out.insert(
        static_cast<const ir_sql_converter::SimplestChunk *>(ir)->GetTableIndex());
  for (const auto &child : ir->children)
    CollectIRSourceIndices(child.get(), out);
}

// Collect mutable JoinNodes in DFS order.
static void
CollectIRJoinNodes(ir_sql_converter::AQPStmt *ir,
                   std::vector<ir_sql_converter::SimplestJoin *> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode)
    out.push_back(static_cast<ir_sql_converter::SimplestJoin *>(ir));
  for (auto &child : ir->children)
    CollectIRJoinNodes(child.get(), out);
}

namespace {

bool PlanContainsOpType(const duckdb::LogicalOperator &op,
                        duckdb::LogicalOperatorType type) {
  if (op.type == type)
    return true;
  for (auto &child : op.children)
    if (PlanContainsOpType(*child, type))
      return true;
  return false;
}

// Collect IR source indices, keeping only those present in `universe` (the
// physical TABLE_SCAN indices). IR sources outside the universe — e.g. the
// ColumnDataGet chunk an IN-list rewrite joins against, which is a
// COLUMN_DATA_SCAN without a table index physically — are stripped so the
// set comparison still works.
void CollectIRSourceIndicesFiltered(
    const ir_sql_converter::AQPStmt *ir,
    const std::unordered_set<unsigned int> &universe,
    std::unordered_set<unsigned int> &out) {
  std::unordered_set<unsigned int> raw;
  CollectIRSourceIndices(ir, raw);
  for (auto v : raw)
    if (universe.count(v))
      out.insert(v);
}

typedef void (*PhysScanCollector)(duckdb::PhysicalOperator &, std::unordered_set<unsigned int> &);

void AnnotateBuildSidesRec(duckdb::PhysicalOperator &op,
                           ir_sql_converter::AQPStmt &ir,
                           const std::unordered_set<unsigned int> &universe,
                           PhysScanCollector collector) {
  for (auto &child : op.children)
    AnnotateBuildSidesRec(child.get(), ir, universe, collector);
  if (op.children.size() != 2)
    return;
  if (op.type != duckdb::PhysicalOperatorType::HASH_JOIN &&
      op.type != duckdb::PhysicalOperatorType::NESTED_LOOP_JOIN &&
      op.type != duckdb::PhysicalOperatorType::BLOCKWISE_NL_JOIN &&
      op.type != duckdb::PhysicalOperatorType::PIECEWISE_MERGE_JOIN)
    return;

  std::unordered_set<unsigned int> probe_set, build_set;
  collector(op.children[0].get(), probe_set);
  collector(op.children[1].get(), build_set);
  // Joins against table-index-less sources (e.g. IN-list chunks) cannot be
  // matched by table-index sets; they stay unannotated.
  if (probe_set.empty() || build_set.empty())
    return;

  std::vector<ir_sql_converter::SimplestJoin *> joins;
  CollectIRJoinNodes(&ir, joins);
  for (auto *join : joins) {
    if (join->GetBuildChild() != -1 || join->children.size() != 2)
      continue;
    std::unordered_set<unsigned int> c0_set, c1_set;
    CollectIRSourceIndicesFiltered(join->children[0].get(), universe, c0_set);
    CollectIRSourceIndicesFiltered(join->children[1].get(), universe, c1_set);
    if (c0_set.empty() || c1_set.empty())
      continue;
    if (c0_set == probe_set && c1_set == build_set) {
      join->SetBuildChild(1); // physical kept logical orientation
      return;
    }
    if (c0_set == build_set && c1_set == probe_set) {
      join->SetBuildChild(0); // physical planner flipped the sides
      return;
    }
  }
#ifndef NDEBUG
  auto dump_set = [](const std::unordered_set<unsigned int> &s) {
    std::string r = "{";
    for (auto v : s)
      r += std::to_string(v) + ",";
    r += "}";
    return r;
  };
  std::cerr << "[AQP-QJIT] AnnotateBuildSides: no IR JoinNode matched "
               "physical join probe="
            << dump_set(probe_set) << " build=" << dump_set(build_set) << "\n";
  for (auto *join : joins) {
    std::unordered_set<unsigned int> c0_set, c1_set;
    if (join->children.size() == 2) {
      CollectIRSourceIndices(join->children[0].get(), c0_set);
      CollectIRSourceIndices(join->children[1].get(), c1_set);
    }
    std::cerr << "[AQP-QJIT]   IR join build_child=" << join->GetBuildChild()
              << " c0=" << dump_set(c0_set) << " c1=" << dump_set(c1_set)
              << "\n";
  }
#endif
}

} // namespace

// Annotate each IR JoinNode with the build side chosen by DuckDB's physical
// planner. Requires FRESH IR: the IR and the physical plan must come from
// the same optimized logical plan (one binder), so each physical HASH_JOIN
// corresponds to exactly one IR JoinNode with identical table-index sets
// under its children — only the build/probe orientation can differ.
// Unmatched joins keep build_child = -1 (analysis then rejects).
void DuckDBAdapter::AnnotateBuildSides(duckdb::PhysicalOperator &op,
                                       ir_sql_converter::AQPStmt &ir,
                                       bool include_chunk_scans) {
  PhysScanCollector collector = include_chunk_scans
      ? CollectPhysScanIndicesWithChunk : CollectPhysScanIndices;
  std::unordered_set<unsigned int> universe;
  collector(op, universe);
  AnnotateBuildSidesRec(op, ir, universe, collector);
}

// ---------------------------------------------------------------------------
// v18 closed IR->plan constructor (fast-path enabler for TopDown Split binary subs).
// ---------------------------------------------------------------------------

namespace {

duckdb::ExpressionType
IrComparisonToDuck(ir_sql_converter::SimplestExprType t) {
  switch (t) {
  case ir_sql_converter::Equal:
    return duckdb::ExpressionType::COMPARE_EQUAL;
  case ir_sql_converter::NotEqual:
    return duckdb::ExpressionType::COMPARE_NOTEQUAL;
  case ir_sql_converter::LessThan:
    return duckdb::ExpressionType::COMPARE_LESSTHAN;
  case ir_sql_converter::GreaterThan:
    return duckdb::ExpressionType::COMPARE_GREATERTHAN;
  case ir_sql_converter::LessEqual:
    return duckdb::ExpressionType::COMPARE_LESSTHANOREQUALTO;
  case ir_sql_converter::GreaterEqual:
    return duckdb::ExpressionType::COMPARE_GREATERTHANOREQUALTO;
  default:
    return duckdb::ExpressionType::INVALID;
  }
}

struct CtorLeaf {
  unsigned int ir_index = 0;
  bool is_chunk = false;
  std::string name;
  const ir_sql_converter::AQPStmt *ir_node = nullptr;
  duckdb::unique_ptr<duckdb::LogicalOperator> op;
  duckdb::LogicalGet *get = nullptr;
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> filters;
  // physical column id -> position in get->column_ids (projection slot)
  std::unordered_map<duckdb::column_t, duckdb::idx_t> col_pos;
};

// Map an IR attr onto a leaf's LogicalGet: temps positionally (IR
// column_index == temp column position, fk_based_splitter contract), base
// tables BY NAME (IR column_index is a binding position from the ORIGINAL
// optimized plan, not a catalog id). Registers the column with the get on
// first use; the binding uses the get's (already IR-overwritten) index.
bool MapLeafAttr(CtorLeaf &leaf, const ir_sql_converter::SimplestAttr &attr,
                 duckdb::LogicalType &out_type,
                 duckdb::ColumnBinding &out_binding) {
  if (attr.GetTableIndex() != leaf.ir_index)
    return false;
  duckdb::column_t col;
  if (leaf.is_chunk) {
    col = attr.GetColumnIndex();
    if (col >= leaf.get->names.size())
      return false;
  } else {
    const auto &names = leaf.get->names;
    auto it = std::find(names.begin(), names.end(), attr.GetColumnName());
    if (it == names.end())
      return false;
    col = static_cast<duckdb::column_t>(it - names.begin());
  }
  auto pos_it = leaf.col_pos.find(col);
  duckdb::idx_t pos;
  if (pos_it == leaf.col_pos.end()) {
    pos = leaf.get->GetColumnIds().size();
    leaf.get->AddColumnId(col);
    leaf.col_pos.emplace(col, pos);
  } else {
    pos = pos_it->second;
  }
  out_type = leaf.get->returned_types[col];
  out_binding = duckdb::ColumnBinding(leaf.get->table_index, pos);
  return true;
}

duckdb::unique_ptr<duckdb::Expression>
MakeLeafColRef(CtorLeaf &leaf, const ir_sql_converter::SimplestAttr &attr) {
  duckdb::LogicalType ty;
  duckdb::ColumnBinding b;
  if (!MapLeafAttr(leaf, attr, ty, b))
    return nullptr;
  return duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
      attr.GetColumnName(), ty, b);
}

bool IrConstToValue(const ir_sql_converter::SimplestConstVar &cv,
                    const duckdb::LogicalType &target, duckdb::Value &out) {
  duckdb::Value v;
  switch (cv.GetType()) {
  case ir_sql_converter::BoolVar:
    v = duckdb::Value::BOOLEAN(cv.GetBoolValue());
    break;
  case ir_sql_converter::IntVar:
    v = duckdb::Value::BIGINT(cv.GetIntValue());
    break;
  case ir_sql_converter::FloatVar:
    v = duckdb::Value::DOUBLE(cv.GetFloatValue());
    break;
  case ir_sql_converter::StringVar:
    v = duckdb::Value(cv.GetStringValue());
    break;
  default: // Date / StringVarArr: no lossless mapping — caller rejects
    return false;
  }
  if (!v.DefaultTryCastAs(target))
    return false;
  out = std::move(v);
  return true;
}

// Convert one IR conjunct into a bound expression against `leaf`. Returns
// nullptr when the conjunct does not belong to this leaf or uses a shape
// outside the v1 whitelist (the caller tries the other leaf, then rejects
// the whole construction). Stray AddColumnId side effects from failed
// nested attempts only widen the scan — semantically harmless.
duckdb::unique_ptr<duckdb::Expression>
ConvertLeafFilter(CtorLeaf &leaf, const ir_sql_converter::AQPExpr &e,
                  duckdb::Optimizer &optimizer) {
  using namespace ir_sql_converter;
  switch (e.GetNodeType()) {
  case VarConstComparisonNode: {
    const auto &c = e.Cast<SimplestVarConstComparison>();
    if (!c.attr || !c.const_var)
      return nullptr;
    auto col = MakeLeafColRef(leaf, *c.attr);
    if (!col)
      return nullptr;
    auto cmp = c.GetSimplestExprType();
    if (cmp == TextLike || cmp == Text_Not_Like) {
      // Bind through the real "~~"/"!~~" functions so the expression
      // rewriter's LIKE->prefix/contains rules see the canonical form.
      if (col->return_type.id() != duckdb::LogicalTypeId::VARCHAR ||
          c.const_var->GetType() != StringVar)
        return nullptr;
      auto pat = duckdb::make_uniq<duckdb::BoundConstantExpression>(
          duckdb::Value(c.const_var->GetStringValue()));
      try {
        return optimizer.BindScalarFunction(cmp == TextLike ? "~~" : "!~~",
                                            std::move(col), std::move(pat));
      } catch (...) {
        return nullptr;
      }
    }
    auto duck_cmp = IrComparisonToDuck(cmp);
    if (duck_cmp == duckdb::ExpressionType::INVALID)
      return nullptr;
    duckdb::Value v;
    if (!IrConstToValue(*c.const_var, col->return_type, v))
      return nullptr;
    return duckdb::make_uniq<duckdb::BoundComparisonExpression>(
        duck_cmp, std::move(col),
        duckdb::make_uniq<duckdb::BoundConstantExpression>(std::move(v)));
  }
  case VarComparisonNode: {
    const auto &c = e.Cast<SimplestVarComparison>();
    if (!c.left_attr || !c.right_attr)
      return nullptr;
    auto duck_cmp = IrComparisonToDuck(c.GetSimplestExprType());
    if (duck_cmp == duckdb::ExpressionType::INVALID)
      return nullptr;
    auto l = MakeLeafColRef(leaf, *c.left_attr);
    auto r = MakeLeafColRef(leaf, *c.right_attr);
    if (!l || !r || l->return_type != r->return_type)
      return nullptr;
    return duckdb::make_uniq<duckdb::BoundComparisonExpression>(
        duck_cmp, std::move(l), std::move(r));
  }
  case IsNullExprNode: {
    const auto &c = e.Cast<SimplestIsNullExpr>();
    if (!c.attr)
      return nullptr;
    auto ty = c.GetSimplestExprType();
    if (ty != NullType && ty != NonNullType)
      return nullptr;
    auto col = MakeLeafColRef(leaf, *c.attr);
    if (!col)
      return nullptr;
    auto op = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
        ty == NullType ? duckdb::ExpressionType::OPERATOR_IS_NULL
                       : duckdb::ExpressionType::OPERATOR_IS_NOT_NULL,
        duckdb::LogicalType::BOOLEAN);
    op->children.push_back(std::move(col));
    return std::move(op);
  }
  case InExprNode: {
    const auto &c = e.Cast<SimplestInExpr>();
    if (!c.attr || c.values.empty())
      return nullptr;
    auto col = MakeLeafColRef(leaf, *c.attr);
    if (!col)
      return nullptr;
    auto col_type = col->return_type;
    auto op = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
        c.negated ? duckdb::ExpressionType::COMPARE_NOT_IN
                  : duckdb::ExpressionType::COMPARE_IN,
        duckdb::LogicalType::BOOLEAN);
    op->children.push_back(std::move(col));
    for (const auto &val : c.values) {
      duckdb::Value v;
      if (!val || !IrConstToValue(*val, col_type, v))
        return nullptr;
      op->children.push_back(
          duckdb::make_uniq<duckdb::BoundConstantExpression>(std::move(v)));
    }
    return std::move(op);
  }
  case LogicalExprNode: {
    const auto &c = e.Cast<SimplestLogicalExpr>();
    switch (c.GetLogicalOp()) {
    case LogicalNot: {
      if (!c.right_expr)
        return nullptr;
      auto child = ConvertLeafFilter(leaf, *c.right_expr, optimizer);
      if (!child)
        return nullptr;
      auto op = duckdb::make_uniq<duckdb::BoundOperatorExpression>(
          duckdb::ExpressionType::OPERATOR_NOT, duckdb::LogicalType::BOOLEAN);
      op->children.push_back(std::move(child));
      return std::move(op);
    }
    case LogicalAnd:
    case LogicalOr: {
      if (!c.left_expr || !c.right_expr)
        return nullptr;
      auto l = ConvertLeafFilter(leaf, *c.left_expr, optimizer);
      if (!l)
        return nullptr;
      auto r = ConvertLeafFilter(leaf, *c.right_expr, optimizer);
      if (!r)
        return nullptr;
      return duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
          c.GetLogicalOp() == LogicalAnd
              ? duckdb::ExpressionType::CONJUNCTION_AND
              : duckdb::ExpressionType::CONJUNCTION_OR,
          std::move(l), std::move(r));
    }
    default:
      return nullptr;
    }
  }
  default:
    return nullptr;
  }
}

} // namespace

duckdb::vector<duckdb::string>
DuckDBAdapter::CtorPlanNames(const ir_sql_converter::AQPStmt &ir) {
  std::map<unsigned int, std::string> leaf_names;
  std::vector<const ir_sql_converter::AQPStmt *> stack{&ir};
  while (!stack.empty()) {
    auto *n = stack.back();
    stack.pop_back();
    if (n->GetNodeType() == ir_sql_converter::ScanNode) {
      const auto &s = n->Cast<ir_sql_converter::SimplestScan>();
      leaf_names[s.GetTableIndex()] = s.GetTableName();
    } else if (n->GetNodeType() == ir_sql_converter::ChunkNode) {
      const auto &c = n->Cast<ir_sql_converter::SimplestChunk>();
      leaf_names[c.GetTableIndex()] = c.GetChunkName();
    }
    for (const auto &ch : n->children)
      if (ch)
        stack.push_back(ch.get());
  }
  auto truncate_col = [](std::string name) -> std::string {
    constexpr size_t kMaxLen = 63;
    if (name.size() > kMaxLen) {
      uint64_t h = 14695981039346656037ULL;
      for (unsigned char c : name) {
        h ^= c;
        h *= 1099511628211ULL;
      }
      char buf[18];
      snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
      return std::string("c_") + buf;
    }
    return name;
  };
  duckdb::vector<duckdb::string> names;
  for (const auto &attr : ir.target_list) {
    auto it = leaf_names.find(attr->GetTableIndex());
    if (it != leaf_names.end() && !it->second.empty())
      names.push_back(truncate_col(it->second + "_" +
                                   std::to_string(attr->GetTableIndex()) +
                                   "_" + attr->GetColumnName()));
    else
      names.push_back(truncate_col("t" + std::to_string(attr->GetTableIndex()) +
                                   "_" + attr->GetColumnName()));
  }
  return names;
}

duckdb::unique_ptr<duckdb::LogicalOperator>
DuckDBAdapter::TryBuildBinaryPlanFromIR(const ir_sql_converter::AQPStmt &ir,
                                        std::string &reject_reason,
                                        duckdb::ClientContext *ctx) {
  using namespace ir_sql_converter;
  // (1) shape: Projection -> [Filter ->] Join(Inner) -> {Scan|Chunk} x2
  if (ir.GetNodeType() != ProjectionNode || ir.children.size() != 1 ||
      ir.target_list.empty() || !ir.qual_vec.empty()) {
    reject_reason = "root-shape";
    return nullptr;
  }
  const AQPStmt *below = ir.children[0].get();
  const AQPStmt *filter_node = nullptr;
  if (below && below->GetNodeType() == FilterNode) {
    if (below->children.size() != 1) {
      reject_reason = "filter-shape";
      return nullptr;
    }
    filter_node = below;
    below = below->children[0].get();
  }
  if (!below || below->GetNodeType() != JoinNode) {
    reject_reason = "join-shape";
    return nullptr;
  }
  const auto &join_ir = below->Cast<SimplestJoin>();
  if (join_ir.GetSimplestJoinType() != Inner ||
      join_ir.join_conditions.empty() || !below->qual_vec.empty() ||
      below->children.size() != 2 || !below->children[0] ||
      !below->children[1]) {
    reject_reason = "join-shape";
    return nullptr;
  }

  // (2) leaves
  CtorLeaf leaves[2];
  for (int i = 0; i < 2; i++) {
    const AQPStmt *leaf_ir = below->children[i].get();
    auto &leaf = leaves[i];
    leaf.ir_node = leaf_ir;
    if (leaf_ir->GetNodeType() == ScanNode) {
      const auto &scan = leaf_ir->Cast<SimplestScan>();
      leaf.ir_index = scan.GetTableIndex();
      leaf.name = scan.GetTableName();
    } else if (leaf_ir->GetNodeType() == ChunkNode) {
      const auto &chunk = leaf_ir->Cast<SimplestChunk>();
      if (chunk.GetChunkName().empty() || !chunk.GetContents().empty()) {
        reject_reason = "leaf-inline-chunk";
        return nullptr;
      }
      leaf.is_chunk = true;
      leaf.ir_index = chunk.GetTableIndex();
      leaf.name = chunk.GetChunkName();
    } else {
      reject_reason = "leaf-not-scan";
      return nullptr;
    }
    if (!leaf_ir->children.empty()) {
      reject_reason = "leaf-has-children";
      return nullptr;
    }
  }
  if (leaves[0].ir_index == leaves[1].ir_index) {
    reject_reason = "duplicate-leaf-index";
    return nullptr;
  }

  // (3) bind both leaves; temps resolve through the replacement scan.
  // Binding + FilterOptimize need an active transaction (catalog lookups,
  // scan statistics) — same Begin/Commit pattern as Optimize()/ParseSQL.
  struct TxnGuard {
    duckdb::ClientContext *ctx = nullptr;
    ~TxnGuard() {
      if (ctx) {
        try {
          ctx->transaction.Commit();
        } catch (...) {
        }
      }
    }
  } txn_guard;
  // ctx: bind on another connection's context (bg speculative compile);
  // default = the adapter's main connection.
  auto *txn_ctx = ctx ? ctx : GetClientContext();
  if (txn_ctx->transaction.IsAutoCommit()) {
    txn_ctx->transaction.BeginTransaction();
    txn_guard.ctx = txn_ctx;
  }
  auto binder = duckdb::Binder::CreateBinder(*txn_ctx);
  duckdb::Optimizer optimizer(*binder, *txn_ctx);
  for (auto &leaf : leaves) {
    duckdb::BaseTableRef ref;
    ref.table_name = leaf.name;
    // unique alias so two leaves over the same catalog table stay distinct
    ref.alias = leaf.name + "__ctor" + std::to_string(leaf.ir_index);
    duckdb::BoundStatement bound;
    try {
      bound = binder->Bind(static_cast<duckdb::TableRef &>(ref));
    } catch (std::exception &ex) {
      reject_reason = std::string("bind:") + ex.what();
      return nullptr;
    }
    if (!bound.plan ||
        bound.plan->type != duckdb::LogicalOperatorType::LOGICAL_GET) {
      reject_reason = "bind-not-get";
      return nullptr;
    }
    leaf.op = std::move(bound.plan);
    leaf.get = &leaf.op->Cast<duckdb::LogicalGet>();
    if (!leaf.is_chunk && !leaf.get->GetTable()) {
      // an unwrapped temp scan (or view) — outside the v1 shape
      reject_reason = "scan-not-catalog";
      return nullptr;
    }
    // AnnotateBuildSides matches physical scans to IR nodes by table index;
    // all bindings below are created against the overwritten index.
    leaf.get->table_index = leaf.ir_index;
    auto est = leaf.ir_node->GetEstimatedCardinality();
    if (est > 0)
      leaf.get->SetEstimatedCardinality(est);
  }

  // (4) filters: leaf-level qual_vec conjuncts belong to their own leaf;
  // residual filter-node conjuncts are matched by ownership (strict
  // per-attr table-index checks make only the owner succeed).
  for (auto &leaf : leaves) {
    for (const auto &q : leaf.ir_node->qual_vec) {
      auto e = ConvertLeafFilter(leaf, *q, optimizer);
      if (!e) {
        reject_reason = "leaf-filter";
        return nullptr;
      }
      leaf.filters.push_back(std::move(e));
    }
  }
  if (filter_node) {
    for (const auto &q : filter_node->qual_vec) {
      bool converted = false;
      for (auto &leaf : leaves) {
        auto e = ConvertLeafFilter(leaf, *q, optimizer);
        if (e) {
          leaf.filters.push_back(std::move(e));
          converted = true;
          break;
        }
      }
      if (!converted) {
        reject_reason = "filter-conjunct";
        return nullptr;
      }
    }
  }
  for (auto &leaf : leaves) {
    if (leaf.filters.empty())
      continue;
    auto filt = duckdb::make_uniq<duckdb::LogicalFilter>();
    filt->expressions = std::move(leaf.filters);
    filt->children.push_back(std::move(leaf.op));
    leaf.op = std::move(filt);
  }

  // (5) join: RIGHT child = build side in the PrepareFromPlan path (no
  // BuildProbeSideOptimizer there) -> put the smaller side right.
  auto c0 = leaves[0].ir_node->GetEstimatedCardinality();
  auto c1 = leaves[1].ir_node->GetEstimatedCardinality();
  int right = (c0 > 0 && c1 > 0 && c0 < c1) ? 0 : 1;
  int left = 1 - right;
  auto join =
      duckdb::make_uniq<duckdb::LogicalComparisonJoin>(duckdb::JoinType::INNER);
  for (const auto &cond : join_ir.join_conditions) {
    if (!cond || cond->GetSimplestExprType() != Equal || !cond->left_attr ||
        !cond->right_attr) {
      reject_reason = "join-cond";
      return nullptr;
    }
    const SimplestAttr *la = cond->left_attr.get();
    const SimplestAttr *ra = cond->right_attr.get();
    if (la->GetTableIndex() == leaves[right].ir_index &&
        ra->GetTableIndex() == leaves[left].ir_index)
      std::swap(la, ra);
    auto lref = MakeLeafColRef(leaves[left], *la);
    auto rref = MakeLeafColRef(leaves[right], *ra);
    if (!lref || !rref) {
      reject_reason = "join-key-map";
      return nullptr;
    }
    if (lref->return_type != rref->return_type) {
      reject_reason = "join-key-type";
      return nullptr;
    }
    duckdb::JoinCondition jc;
    jc.left = std::move(lref);
    jc.right = std::move(rref);
    jc.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
    join->conditions.push_back(std::move(jc));
  }
  join->children.push_back(std::move(leaves[left].op));
  join->children.push_back(std::move(leaves[right].op));
  auto jest = below->GetEstimatedCardinality();
  if (jest > 0)
    join->SetEstimatedCardinality(jest);

  // (6) projection: output order = IR target order (temp-store contract)
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> proj_exprs;
  for (const auto &attr : ir.target_list) {
    duckdb::unique_ptr<duckdb::Expression> ref;
    for (auto &leaf : leaves) {
      ref = MakeLeafColRef(leaf, *attr);
      if (ref)
        break;
    }
    if (!ref) {
      reject_reason = "target-map";
      return nullptr;
    }
    proj_exprs.push_back(std::move(ref));
  }
  duckdb::idx_t proj_index =
      std::max(leaves[0].ir_index, leaves[1].ir_index) + 1;
  auto root = duckdb::make_uniq<duckdb::LogicalProjection>(
      proj_index, std::move(proj_exprs));
  root->children.push_back(std::move(join));
  auto rest = ir.GetEstimatedCardinality();
  if (rest > 0)
    root->SetEstimatedCardinality(rest);

  // (7) cheap FilterOptimize: expression rewriter (LIKE->prefix parity) +
  // filter pushdown (populates LogicalGet::table_filters -> zone maps).
  duckdb::unique_ptr<duckdb::LogicalOperator> plan_root = std::move(root);
  try {
    plan_root = optimizer.FilterOptimize(std::move(plan_root));
  } catch (std::exception &ex) {
    reject_reason = std::string("filter-optimize:") + ex.what();
    return nullptr;
  }
  return plan_root;
}

DuckDBAdapter::QueryJitPrep
DuckDBAdapter::PrepareWithQueryJitAnalysis(const std::string &sql,
                                           const std::string &label) {
  using SteadyClock = std::chrono::steady_clock;
  static const bool trace_pass2 =
      std::getenv("AQP_QJIT_PASS2_TRACE") != nullptr;
  auto t0 = trace_pass2 ? SteadyClock::now() : SteadyClock::time_point{};

  QueryJitPrep prep;
  auto saved_planner = std::move(planner);
  auto saved_plan = std::move(plan);
  auto saved_map = intermediate_table_map;
  auto saved_chunk_names = chunk_col_names_;

  std::unique_ptr<duckdb::PreparedStatement> prepared;
  auto t_before_pfp = t0;
  auto t_parse = t0, t_opt = t0, t_conv = t0;
  const char *stage = "parse";
  bool plan_empty_result = false;
  std::string fail_reason;
  try {
    ParseSQL(sql);
    if (trace_pass2) t_parse = SteadyClock::now();
    stage = "optimize";
    Optimize();
    if (trace_pass2) t_opt = SteadyClock::now();
    // Fresh binder ⇒ temp-table scans got fresh indices; rebuild the maps
    // so ConvertPlanToIR resolves ChunkNodes.
    intermediate_table_map.clear();
    chunk_col_names_.clear();
    stage = "rebuild-temp-indices";
    RebuildTempTableIndices(plan.get(), intermediate_table_map,
                            chunk_col_names_, temp_collections_,
                            qjit_temp_meta_);
    // Reject EMPTY_RESULT plans before conversion to keep stdout identical.
    plan_empty_result =
        PlanContainsOpType(*plan, duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT);
    std::unique_ptr<ir_sql_converter::AQPStmt> ir;
    if (!plan_empty_result) {
      stage = "convert-ir";
      ir = ConvertPlanToIR();
    }
    if (trace_pass2) t_conv = SteadyClock::now();
    t_before_pfp = trace_pass2 ? SteadyClock::now()
                              : SteadyClock::time_point{};
    stage = "prepare-from-plan";
    prepared = conn->PrepareFromPlan(TakePlan(), planner->names,
                                     planner->types);
    if (prepared && !prepared->HasError() && prepared->data &&
        prepared->data->physical_plan) {
      if (ir) {
        AnnotateBuildSides(prepared->data->physical_plan->Root(), *ir);
        prep.analysis = qjit::AnalyzeQueryJit(*ir, label);
        prep.ir = std::move(ir);
      } else if (!plan_empty_result) {
        fail_reason = "ir-conversion-produced-no-IR";
      }
    } else {
      fail_reason = "prepare-from-plan failed";
      if (prepared && prepared->HasError())
        fail_reason += ": " + prepared->GetError();
    }
  } catch (std::exception &e) {
    fail_reason = std::string("stage=") + stage + ": " + e.what();
    prepared.reset();
  }

  if (trace_pass2) {
    auto t_end = SteadyClock::now();
    double total = std::chrono::duration<double, std::milli>(t_end - t0).count();
    double before_pfp = std::chrono::duration<double, std::milli>(
        t_before_pfp - t0).count();
    double pfp_onward = total - before_pfp;
    auto ms = [](SteadyClock::time_point a, SteadyClock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    fprintf(stderr,
            "[AQP-PASS2] label=%s total=%.3f parse_opt_ir=%.3f pfp_onward=%.3f"
            " parse=%.3f opt=%.3f conv=%.3f\n",
            label.c_str(), total, before_pfp, pfp_onward,
            ms(t0, t_parse), ms(t_parse, t_opt), ms(t_opt, t_conv));
  }

  planner = std::move(saved_planner);
  plan = std::move(saved_plan);
  intermediate_table_map = std::move(saved_map);
  chunk_col_names_ = std::move(saved_chunk_names);

  if (!fail_reason.empty() || !prepared || prepared->HasError()) {
    auto og = MakeOptGuard();
    prepared = conn->Prepare(sql);
    prep.ir.reset();
    prep.analysis = qjit::QjitAnalysisResult{};
  }
  prep.prepared = std::move(prepared);

  return prep;
}

bool DuckDBAdapter::BuildQjitOutputDescs(const qjit::JITHubIR &plan,
                                         duckdb::PreparedStatement &prepared,
                                         QjitCompiled &compiled,
                                         std::string &reason) {
  const qjit::QjitStep &last = plan.steps.back();

  // Output cross-check against the prepared statement — under query-jit the
  // compiled code OWNS the result, so shape/type disagreement with DuckDB's
  // binding is fatal and must fall back.
  const auto &types = prepared.GetTypes();
  const auto &names = prepared.GetNames();
  auto out_name = [&](size_t i) {
    return (i < names.size() && !names[i].empty()) ? names[i]
                                                   : "col_" + std::to_string(i);
  };
  if (plan.has_agg) {
    if (types.size() != plan.agg_output_cells.size()) {
      reason = "output:count-mismatch";
      return false;
    }
    for (size_t i = 0; i < types.size(); i++) {
      const qjit::QjitAggCellPlan &cell =
          last.agg_cells[(size_t)plan.agg_output_cells[i]];
      int32_t dt;
      if (cell.fn == qjit::QjitAggFn::Count ||
          cell.fn == qjit::QjitAggFn::CountStar) {
        if (types[i].id() != duckdb::LogicalTypeId::BIGINT) {
          reason = "output:agg-type";
          return false;
        }
        dt = AQP_DTYPE_INT64;
      } else if (cell.fn == qjit::QjitAggFn::Average) {
        auto tid = types[i].id();
        if (tid != duckdb::LogicalTypeId::DOUBLE &&
            tid != duckdb::LogicalTypeId::DECIMAL) {
          reason = "output:agg-type";
          return false;
        }
        dt = AQP_DTYPE_DOUBLE;
      } else if (cell.arg.dtype == AQP_DTYPE_INT32) {
        auto tid = types[i].id();
        if (tid != duckdb::LogicalTypeId::INTEGER &&
            tid != duckdb::LogicalTypeId::DATE &&
            tid != duckdb::LogicalTypeId::SMALLINT &&
            tid != duckdb::LogicalTypeId::TINYINT &&
            !(tid == duckdb::LogicalTypeId::DECIMAL &&
              duckdb::DecimalType::GetWidth(types[i]) <= 9)) {
          reason = "output:agg-type";
          return false;
        }
        dt = AQP_DTYPE_INT32;
      } else if (cell.arg.dtype == AQP_DTYPE_INT64) {
        auto tid = types[i].id();
        if (tid != duckdb::LogicalTypeId::BIGINT &&
            tid != duckdb::LogicalTypeId::TIMESTAMP &&
            tid != duckdb::LogicalTypeId::TIMESTAMP_TZ &&
            !(tid == duckdb::LogicalTypeId::DECIMAL &&
              duckdb::DecimalType::GetWidth(types[i]) > 9)) {
          reason = "output:agg-type";
          return false;
        }
        dt = AQP_DTYPE_INT64;
      } else if (cell.arg.dtype == AQP_DTYPE_VARCHAR) {
        if (types[i].id() != duckdb::LogicalTypeId::VARCHAR) {
          reason = "output:agg-type";
          return false;
        }
        dt = AQP_DTYPE_VARCHAR;
      } else {
        reason = "output:agg-type";
        return false;
      }
      uint8_t dec_scale = 0;
      if (types[i].id() == duckdb::LogicalTypeId::DECIMAL)
        dec_scale = duckdb::DecimalType::GetScale(types[i]);
      compiled.out_descs.push_back({dt, out_name(i), dec_scale});
    }
    compiled.agg_output_cells = plan.agg_output_cells;
    compiled.agg_descs.reserve(last.agg_cells.size());
    for (const auto &cell : last.agg_cells) {
      qjit::QjitAggDType adt;
      if (cell.fn == qjit::QjitAggFn::Average)
        adt = (cell.has_arg && cell.arg.dtype == AQP_DTYPE_VARCHAR)
                  ? qjit::QjitAggDType::Str
                  : (cell.has_arg && (cell.arg.dtype == AQP_DTYPE_DOUBLE ||
                                      cell.arg.dtype == AQP_DTYPE_FLOAT))
                        ? qjit::QjitAggDType::F64
                        : qjit::QjitAggDType::I64;
      else
        adt = (cell.has_arg && cell.arg.dtype == AQP_DTYPE_VARCHAR)
                  ? qjit::QjitAggDType::Str
                  : qjit::QjitAggDType::I64;
      compiled.agg_descs.push_back({cell.fn, adt});
    }
  } else {
    if (types.size() != last.outputs.size()) {
      reason = "output:count-mismatch";
      return false;
    }
    for (size_t i = 0; i < types.size(); i++) {
      int32_t dt = last.outputs[i].dtype;
      bool ok = false;
      if (dt == AQP_DTYPE_INT32) {
        ok = types[i].id() == duckdb::LogicalTypeId::INTEGER ||
             types[i].id() == duckdb::LogicalTypeId::DATE ||
             types[i].id() == duckdb::LogicalTypeId::SMALLINT ||
             types[i].id() == duckdb::LogicalTypeId::TINYINT ||
             (types[i].id() == duckdb::LogicalTypeId::DECIMAL &&
              duckdb::DecimalType::GetWidth(types[i]) <= 9);
      } else if (dt == AQP_DTYPE_INT64) {
        ok = types[i].id() == duckdb::LogicalTypeId::BIGINT ||
             types[i].id() == duckdb::LogicalTypeId::TIMESTAMP ||
             types[i].id() == duckdb::LogicalTypeId::TIMESTAMP_TZ ||
             (types[i].id() == duckdb::LogicalTypeId::DECIMAL &&
              duckdb::DecimalType::GetWidth(types[i]) > 9);
      } else if (dt == AQP_DTYPE_VARCHAR) {
        ok = types[i].id() == duckdb::LogicalTypeId::VARCHAR;
      }
      if (!ok) {
        reason = "output:type-mismatch i=" + std::to_string(i) + " dt=" +
                 std::to_string(dt) + " plan_type=" + types[i].ToString() +
                 " name=" + out_name(i);
        return false;
      }
      compiled.out_descs.push_back({dt, out_name(i)});
    }
  }
  return true;
}

bool DuckDBAdapter::ResolveQjitSources(const qjit::JITHubIR &plan,
                                       QjitCompiled &compiled,
                                       std::string &reason) {
  if (!qjit_storage_plan_ || !qjit_storage_plan_->IsLoaded()) {
    reason = "no-storage-plan";
    return false;
  }
  if (!qjit_executor_) {
    qjit_executor_ = std::make_unique<qjit::QjitExecutor>(
        query_jit_threads_,
        query_jit_morsel_ > 0 ? (uint64_t)query_jit_morsel_ : 0);
    qjit_executor_->SetDebugRuntimeStatBehavior(debug_runtime_stat_behavior_);
  }
  compiled.srcs.resize(plan.steps.size());
  for (size_t k = 0; k < plan.steps.size(); k++) {
    const qjit::QjitStep &st = plan.steps[k];
    if (st.source_is_temp) {
      const qjit::QjitTable *tmp = GetOrLoadQjitTemp(st.source_table, reason);
      if (!tmp)
        return false;
      if (!qjit_executor_->ResolveTempSource(*tmp, st.cols, compiled.srcs[k],
                                             reason, st.block_skip_col))
        return false;
      continue;
    }
    const auto *flat = qjit_storage_plan_->GetTable(st.source_table);
    if (!flat) {
      reason = "source:table-missing:" + st.source_table;
      return false;
    }
    if (!qjit_executor_->ResolveSource(*flat, st.cols, compiled.srcs[k],
                                       reason, st.block_skip_col))
      return false;
  }
  return true;
}

void DuckDBAdapter::RegisterQjitRuntimeSymbols(
    aqp_jit::IrToLlvmCompiler *comp) {
  comp->RegisterRuntimeSymbol("qjit_parallel_for", (void *)&qjit_parallel_for);
  comp->RegisterRuntimeSymbol("qjit_buffer_grow", (void *)&qjit_buffer_grow);
  comp->RegisterRuntimeSymbol("qjit_ht_append", (void *)&qjit_ht_append);
  comp->RegisterRuntimeSymbol("qjit_ht_begin", (void *)&qjit_ht_begin);
  comp->RegisterRuntimeSymbol("qjit_ht_append_slow",
                              (void *)&qjit_ht_append_slow);
  comp->RegisterRuntimeSymbol("qjit_ht_end", (void *)&qjit_ht_end);
  comp->RegisterRuntimeSymbol("qjit_ht_finalize", (void *)&qjit_ht_finalize);
  comp->RegisterRuntimeSymbol("qjit_str_arena_copy",
                              (void *)&qjit_str_arena_copy);
  comp->RegisterRuntimeSymbol("qjit_table_append_i32",
                              (void *)&qjit_table_append_i32);
  comp->RegisterRuntimeSymbol("qjit_table_append_i64",
                              (void *)&qjit_table_append_i64);
  comp->RegisterRuntimeSymbol("qjit_table_append_str",
                              (void *)&qjit_table_append_str);
  comp->RegisterRuntimeSymbol("qjit_table_append_null",
                              (void *)&qjit_table_append_null);
  comp->RegisterRuntimeSymbol("qjit_table_finish_row",
                              (void *)&qjit_table_finish_row);
  comp->RegisterRuntimeSymbol("qjit_ht_dir", (void *)&qjit_ht_dir);
  comp->RegisterRuntimeSymbol("qjit_ht_mask", (void *)&qjit_ht_mask);
  comp->RegisterRuntimeSymbol("qjit_ht_key0_min", (void *)&qjit_ht_key0_min);
  comp->RegisterRuntimeSymbol("qjit_ht_key0_max", (void *)&qjit_ht_key0_max);
  comp->RegisterRuntimeSymbol("qjit_ht_entries", (void *)&qjit_ht_entries);
  comp->RegisterRuntimeSymbol("qjit_agg_update_i64",
                              (void *)&qjit_agg_update_i64);
  comp->RegisterRuntimeSymbol("qjit_agg_update_str",
                              (void *)&qjit_agg_update_str);
  comp->RegisterRuntimeSymbol("qjit_agg_update_count",
                              (void *)&qjit_agg_update_count);
  comp->RegisterRuntimeSymbol("qjit_gagg_lookup", (void *)&qjit_gagg_lookup);
  comp->RegisterRuntimeSymbol("qjit_hash_string", (void *)&qjit_hash_string);
  comp->RegisterRuntimeSymbol("qjit_table_begin", (void *)&qjit_table_begin);
  comp->RegisterRuntimeSymbol("qjit_table_col_slow",
                              (void *)&qjit_table_col_slow);
  comp->RegisterRuntimeSymbol("qjit_table_null_slow",
                              (void *)&qjit_table_null_slow);
  comp->RegisterRuntimeSymbol("qjit_table_end", (void *)&qjit_table_end);
  comp->RegisterRuntimeSymbol("qjit_table_str_copy",
                              (void *)&qjit_table_str_copy);
  comp->RegisterRuntimeSymbol("qjit_debug_count_row",
                              (void *)&qjit_debug_count_row);
  comp->RegisterRuntimeSymbol("qjit_debug_count_range_filtered",
                              (void *)&qjit_debug_count_range_filtered);
  comp->RegisterRuntimeSymbol("qjit_debug_count_membership_filtered",
                              (void *)&qjit_debug_count_membership_filtered);
  comp->RegisterRuntimeSymbol("qjit_debug_count_bloom_filtered",
                              (void *)&qjit_debug_count_bloom_filtered);
  comp->RegisterRuntimeSymbol("qjit_debug_count_block_checked",
                              (void *)&qjit_debug_count_block_checked);
  comp->RegisterRuntimeSymbol("qjit_debug_count_block_skipped",
                              (void *)&qjit_debug_count_block_skipped);
  comp->RegisterRuntimeSymbol("qjit_debug_flush_counters",
                              (void *)&qjit_debug_flush_counters);
}

std::unique_ptr<DuckDBAdapter::QjitCompiled>
DuckDBAdapter::TryCompileQueryJit(const ir_sql_converter::AQPStmt &ir,
                                  const qjit::QjitAnalysisResult &analysis,
                                  duckdb::PreparedStatement &prepared,
                                  const std::string &label, bool use_fast) {
  auto fallback = [&](const std::string &reason)
      -> std::unique_ptr<QjitCompiled> {
#ifndef NDEBUG
    fprintf(stderr, "[AQP-QJIT] fallback:%s label=%s\n", reason.c_str(),
            label.c_str());
#else
    (void)reason;
#endif
    return nullptr;
  };

  if (!analysis.accepted)
    return nullptr;
  {
  }
  if (!qjit_storage_plan_ || !qjit_storage_plan_->IsLoaded())
    return fallback("no-storage-plan");

  qjit::JITHubIR plan;
  std::string reason;
  if (!qjit::BuildExecutionSteps(ir, plan, reason, range_guard_,
                                   block_skip_, membership_preprobe_))
    return fallback(reason);

  auto compiled = std::make_unique<QjitCompiled>();
  if (!BuildQjitOutputDescs(plan, prepared, *compiled, reason))
    return fallback(reason);

  if (!ResolveQjitSources(plan, *compiled, reason))
    return fallback(reason);

  {
    static constexpr uint64_t kSourceRowBypass = 100000000;
    for (size_t k = 0; k < plan.steps.size(); k++) {
      if ((plan.steps[k].sink == qjit::QjitStep::Result ||
           plan.steps[k].sink == qjit::QjitStep::Agg) &&
          compiled->srcs[k].nrows > kSourceRowBypass)
        return fallback("source-rows:" +
                        std::to_string(compiled->srcs[k].nrows) + ">" +
                        std::to_string(kSourceRowBypass));
    }
  }

  compiled->ht_tuple_sizes.reserve(plan.hts.size());
  compiled->ht_key0_offsets.reserve(plan.hts.size());
  for (const auto &ht : plan.hts) {
    compiled->ht_tuple_sizes.push_back(ht.tuple_size);
    compiled->ht_key0_offsets.push_back(ht.prefix_bytes); // key 0 lives first
  }
  compiled->order_by = plan.order_by;
  compiled->limit = plan.limit;

  // use_fast (spec-jit FAST_ONCE): route through the compensate compiler.
  aqp_jit::IrToLlvmCompiler *comp;
  if (use_fast) {
    comp = EnsureFastJitCompiler();
    if (!comp)
      return fallback("no-jit-compiler");
    if (comp == jit_compiler_.get()) {
      if (!qjit_syms_registered_) {
        RegisterQjitRuntimeSymbols(comp);
        qjit_syms_registered_ = true;
      }
    } else if (!fast_qjit_syms_registered_) {
      RegisterQjitRuntimeSymbols(comp);
      fast_qjit_syms_registered_ = true;
    }
  } else {
    EnsureJITCompiler();
    if (!jit_compiler_)
      return fallback("no-jit-compiler");
    comp = jit_compiler_.get();
    if (!qjit_syms_registered_) {
      RegisterQjitRuntimeSymbols(comp);
      qjit_syms_registered_ = true;
    }
    if (!JitResident())
      comp->ResetModules();
  }

  // AQP_QJIT_TIME=1: codegen-only timing (opt + add + lookup breakdown).
  static const bool qjit_time_trace = std::getenv("AQP_QJIT_TIME") != nullptr;
  std::chrono::steady_clock::time_point cg_tic;
  if (qjit_time_trace)
    cg_tic = std::chrono::steady_clock::now();
  compiled->fn = comp->CompileQuerySteps(plan, &compiled->params_buf);
  if (qjit_time_trace) {
    auto cg_us = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - cg_tic)
                     .count();
    auto &ct = comp->LastCodegenTiming();
    fprintf(stderr, "[AQP-QJIT-TIME] codegen label=%s us=%lld opt=%ld add=%ld lookup=%ld\n",
            label.c_str(), static_cast<long long>(cg_us),
            ct.opt_us, ct.add_us, ct.lookup_us);
  }
  if (!compiled->fn)
    return fallback("compile-failed");

  return compiled;
}

std::unique_ptr<DuckDBAdapter::QjitSpecCompiled>
DuckDBAdapter::SpeculativeQueryJitCompile(const std::string &sql,
                                          const std::string &label,
                                          duckdb::Connection &spec_conn,
                                          aqp_jit::IrToLlvmCompiler *spec_comp) {
  auto reject = [&](const std::string &reason)
      -> std::unique_ptr<QjitSpecCompiled> {
#ifndef NDEBUG
    fprintf(stderr, "[AQP-QJIT] spec-reject:%s label=%s\n", reason.c_str(),
            label.c_str());
#else
    (void)reason;
#endif
    return nullptr;
  };
  if (!qjit_storage_plan_ || !qjit_storage_plan_->IsLoaded())
    return reject("no-storage-plan");

  auto *spec_ctx = spec_conn.context.get();
  try {
    // Parse + plan + optimize with bg-local state (mirrors ParseSQL/Optimize;
    // the adapter's planner/plan members belong to the main thread). The spec
    // connection's ClientConfig has the same compile-time
    // enable_dbshaker_query_split default as the main connection, so the
    // optimizer behaves identically — plan parity by construction.
    std::unique_ptr<duckdb::Planner> local_planner;
    duckdb::unique_ptr<duckdb::LogicalOperator> local_plan;
    bool auto_commit = spec_ctx->transaction.IsAutoCommit();
    if (auto_commit)
      spec_ctx->transaction.BeginTransaction();
    try {
      duckdb::Parser parser(spec_ctx->GetParserOptions());
      parser.ParseQuery(sql);
      if (parser.statements.empty() ||
          parser.statements[0]->type !=
              duckdb::StatementType::SELECT_STATEMENT)
        throw std::runtime_error("not a single SELECT statement");
      local_planner = std::make_unique<duckdb::Planner>(*spec_ctx);
      local_planner->CreatePlan(std::move(parser.statements[0]));
      local_plan = std::move(local_planner->plan);
      if (!local_plan)
        throw std::runtime_error("failed to create logical plan");
      if (local_plan->RequireOptimizer()) {
        duckdb::Optimizer optimizer(*local_planner->binder, *spec_ctx);
        if (disable_engine_optimizer_)
          local_plan = optimizer.FilterOptimize(std::move(local_plan));
        else
          local_plan = optimizer.Optimize(std::move(local_plan));
      }
    } catch (...) {
      if (auto_commit)
        spec_ctx->transaction.Commit();
      throw;
    }
    if (auto_commit)
      spec_ctx->transaction.Commit();

    if (PlanContainsOpType(*local_plan,
                           duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT))
      return reject("plan-empty-result");

    // Fresh binder ⇒ fresh temp-scan indices; bg-local maps so the converter
    // can resolve ChunkNodes. This reads the live temp maps, which the main
    // thread mutates concurrently under compensate-jit's early launch (and
    // in legacy mode when this task runs late, queued behind a zombie):
    // take temp_state_mutex_ shared, same as the replacement-scan binding
    // above (which already takes it inside TempCollectionBind/QjitTempBind).
    // RebuildTempTableIndices reads only column_names from the entries; the
    // exclusive-locked store sites cover the field replacement.
    std::unordered_map<unsigned int, std::string> local_map;
    std::unordered_map<unsigned int, std::vector<std::string>> local_names;
    {
      std::shared_lock<std::shared_timed_mutex> rguard(temp_state_mutex_);
      RebuildTempTableIndices(local_plan.get(), local_map, local_names,
                              temp_collections_, qjit_temp_meta_);
    }

    auto ir = ir_sql_converter::ConvertDuckDBPlanToIR(
        *local_planner->binder, *spec_ctx, local_plan.get(), local_map, false,
        &local_names);
    if (!ir)
      return reject("ir-conversion-failed");

    auto prepared = spec_conn.PrepareFromPlan(
        std::move(local_plan), local_planner->names, local_planner->types);
    if (!prepared || prepared->HasError() || !prepared->data ||
        !prepared->data->physical_plan)
      return reject("prepare-failed");

    AnnotateBuildSides(prepared->data->physical_plan->Root(), *ir);
    auto analysis = qjit::AnalyzeQueryJit(*ir, label);
    if (!analysis.accepted)
      return nullptr; // AnalyzeQueryJit already traced the reject reason

    auto payload = std::make_unique<QjitSpecCompiled>();
    std::string reason;
    if (!qjit::BuildExecutionSteps(*ir, payload->plan, reason,
                                   range_guard_, block_skip_,
                                   membership_preprobe_))
      return reject(reason);
    if (qjit_storage_plan_ && qjit_storage_plan_->IsLoaded()) {
      static constexpr uint64_t kSpecSourceRowBypass = 100000000;
      for (const auto &st : payload->plan.steps) {
        if ((st.sink == qjit::QjitStep::Result ||
             st.sink == qjit::QjitStep::Agg) && !st.source_is_temp) {
          const auto *flat = qjit_storage_plan_->GetTable(st.source_table);
          if (flat && flat->row_count > kSpecSourceRowBypass)
            return reject("source-rows:" +
                          std::to_string(flat->row_count) + ">" +
                          std::to_string(kSpecSourceRowBypass));
        }
      }
    }

    payload->compiled = std::make_unique<QjitCompiled>();
    if (!BuildQjitOutputDescs(payload->plan, *prepared, *payload->compiled,
                              reason))
      return reject(reason);
    payload->compiled->ht_tuple_sizes.reserve(payload->plan.hts.size());
    payload->compiled->ht_key0_offsets.reserve(payload->plan.hts.size());
    for (const auto &ht : payload->plan.hts) {
      payload->compiled->ht_tuple_sizes.push_back(ht.tuple_size);
      payload->compiled->ht_key0_offsets.push_back(ht.prefix_bytes);
    }

    payload->compiled->fn = spec_comp->CompileQuerySteps(
        payload->plan, &payload->compiled->params_buf);
    if (!payload->compiled->fn)
      return reject("compile-failed");

    payload->ir = std::move(ir);
    payload->out_types = prepared->GetTypes();
    payload->out_names = prepared->GetNames();
    payload->est_card =
        prepared->data->physical_plan->Root().estimated_cardinality;
    return payload;
  } catch (std::exception &e) {
    return reject(std::string("exception:") + e.what());
  }
}

std::unique_ptr<DuckDBAdapter::QjitSpecCompiled>
DuckDBAdapter::SpeculativeQueryJitCompileFromPlan(
    ir_sql_converter::AQPStmt &ir,
    duckdb::unique_ptr<duckdb::LogicalOperator> sub_plan,
    const duckdb::vector<duckdb::LogicalType> &plan_types,
    const std::string &label,
    duckdb::Connection &spec_conn,
    aqp_jit::IrToLlvmCompiler *spec_comp,
    const duckdb::vector<duckdb::string> *plan_names,
    bool include_chunk_scans) {
  auto reject = [&](const std::string &reason)
      -> std::unique_ptr<QjitSpecCompiled> {
#ifndef NDEBUG
    fprintf(stderr, "[AQP-QJIT] spec-reject-from-plan:%s label=%s\n",
            reason.c_str(), label.c_str());
#else
    (void)reason;
#endif
    return nullptr;
  };
  if (!qjit_storage_plan_ || !qjit_storage_plan_->IsLoaded())
    return reject("no-storage-plan");

  try {
    if (PlanContainsOpType(*sub_plan,
                           duckdb::LogicalOperatorType::LOGICAL_EMPTY_RESULT))
      return reject("plan-empty-result");

    duckdb::vector<duckdb::string> names;
    if (plan_names && plan_names->size() == plan_types.size()) {
      names = *plan_names;
    } else {
      names.reserve(plan_types.size());
      for (duckdb::idx_t i = 0; i < plan_types.size(); i++)
        names.push_back("c" + std::to_string(i));
    }

    auto prepared = spec_conn.PrepareFromPlan(
        std::move(sub_plan), std::move(names), plan_types);
    if (!prepared || prepared->HasError() || !prepared->data ||
        !prepared->data->physical_plan)
      return reject("prepare-failed");

    AnnotateBuildSides(prepared->data->physical_plan->Root(), ir,
                       include_chunk_scans);
    auto analysis = qjit::AnalyzeQueryJit(ir, label);
    if (!analysis.accepted)
      return nullptr;

    auto payload = std::make_unique<QjitSpecCompiled>();
    std::string reason;
    if (!qjit::BuildExecutionSteps(ir, payload->plan, reason,
                                   range_guard_, block_skip_,
                                   membership_preprobe_))
      return reject(reason);

    if (qjit_storage_plan_ && qjit_storage_plan_->IsLoaded()) {
      static constexpr uint64_t kSpecSourceRowBypass = 100000000;
      for (const auto &st : payload->plan.steps) {
        if ((st.sink == qjit::QjitStep::Result ||
             st.sink == qjit::QjitStep::Agg) && !st.source_is_temp) {
          const auto *flat = qjit_storage_plan_->GetTable(st.source_table);
          if (flat && flat->row_count > kSpecSourceRowBypass)
            return reject("source-rows:" +
                          std::to_string(flat->row_count) + ">" +
                          std::to_string(kSpecSourceRowBypass));
        }
      }
    }

    payload->compiled = std::make_unique<QjitCompiled>();
    if (!BuildQjitOutputDescs(payload->plan, *prepared, *payload->compiled,
                              reason))
      return reject(reason);
    payload->compiled->ht_tuple_sizes.reserve(payload->plan.hts.size());
    payload->compiled->ht_key0_offsets.reserve(payload->plan.hts.size());
    for (const auto &ht : payload->plan.hts) {
      payload->compiled->ht_tuple_sizes.push_back(ht.tuple_size);
      payload->compiled->ht_key0_offsets.push_back(ht.prefix_bytes);
    }

    payload->compiled->fn = spec_comp->CompileQuerySteps(
        payload->plan, &payload->compiled->params_buf);
    if (!payload->compiled->fn)
      return reject("compile-failed");

    payload->ir = nullptr;
    payload->out_types = prepared->GetTypes();
    payload->out_names = prepared->GetNames();
    payload->est_card =
        prepared->data->physical_plan->Root().estimated_cardinality;
    return payload;
  } catch (std::exception &e) {
    return reject(std::string("exception:") + e.what());
  }
}

const qjit::QjitTable *
DuckDBAdapter::GetOrLoadQjitTemp(const std::string &name,
                                 std::string &reason) {
  auto it = qjit_temps_.find(name);
  if (it != qjit_temps_.end())
    return it->second.get();
  if (no_mode_bridging_) {
    reason = "source:tmp-no-bridge:" + name;
    return nullptr;
  }
  auto tc = temp_collections_.find(name);
  if (tc == temp_collections_.end() || !tc->second.collection) {
    reason = "source:tmp-missing:" + name;
    return nullptr;
  }
  // Speculative placeholders are Combine()d in-place later — not final.
  if (tc->second.is_placeholder) {
    reason = "source:tmp-placeholder:" + name;
    return nullptr;
  }
  auto table = CollectionToQjitTable(tc->second, reason);
  if (!table)
    return nullptr;
  const qjit::QjitTable *raw = table.get();
  {
    std::unique_lock<std::shared_timed_mutex> wguard(temp_state_mutex_);
    qjit_temps_[name] = std::move(table);
  }
  return raw;
}

std::unique_ptr<qjit::QjitTable>
DuckDBAdapter::CollectionToQjitTable(const StoredTempResult &stored,
                                     std::string &reason) {
  auto &collection = *stored.collection;
  const auto &types = collection.Types();
  std::vector<qjit::QjitTable::ColumnDesc> descs;
  descs.reserve(types.size());
  for (size_t i = 0; i < types.size(); i++) {
    int32_t dt;
    if (!QjitDtypeForLogicalType(types[i], dt)) {
      reason = "source:tmp-cdc-dtype";
      return nullptr;
    }
    std::string name = i < stored.column_names.size()
                           ? stored.column_names[i]
                           : "col_" + std::to_string(i);
    descs.push_back({dt, std::move(name)});
  }

  uint64_t total_rows = 0;
  for (auto &chunk : collection.Chunks())
    total_rows += chunk.size();

  auto table = std::make_unique<qjit::QjitTable>(std::move(descs), 1);
  table->ReserveFlat(total_rows);

  uint64_t row_offset = 0;
  for (auto &chunk : collection.Chunks()) {
    const duckdb::idx_t n = chunk.size();
    if (n == 0) continue;
    for (duckdb::idx_t c = 0; c < chunk.ColumnCount(); c++) {
      auto &vec = chunk.data[c];
      uint32_t elem_sz = table->ElemSize(c);
      uint8_t *dst = table->FlatData(c) + row_offset * elem_sz;
      auto &mask = duckdb::FlatVector::Validity(vec);
      switch (table->Col(c).dtype) {
      case AQP_DTYPE_INT32:
        std::memcpy(dst, duckdb::FlatVector::GetData<int32_t>(vec),
                    n * sizeof(int32_t));
        break;
      case AQP_DTYPE_INT64:
        std::memcpy(dst, duckdb::FlatVector::GetData<int64_t>(vec),
                    n * sizeof(int64_t));
        break;
      default: {
        auto *src = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
        auto *out = reinterpret_cast<QjitString *>(dst);
        auto &arena = table->FlatArena();
        for (duckdb::idx_t r = 0; r < n; r++) {
          if (mask.RowIsValid(r)) {
            out[r] = arena.Copy(
                qjit::MakeString(src[r].GetData(), (uint32_t)src[r].GetSize()));
          } else {
            std::memset(&out[r], 0, sizeof(QjitString));
          }
        }
        break;
      }
      }
      if (!mask.AllValid()) {
        uint64_t *val = table->FlatValidity(c);
        const auto *mdata = mask.GetData();
        for (duckdb::idx_t r = 0; r < n; r++) {
          if (!mask.RowIsValidUnsafe(r))
            qjit::SetRowInvalid(val, row_offset + r);
        }
      }
    }
    row_offset += n;
  }
  table->MarkFinalized(total_rows);
  return table;
}

duckdb::unique_ptr<duckdb::ColumnDataCollection>
DuckDBAdapter::QjitTableToCollection(
    const qjit::QjitTable &table,
    const duckdb::vector<duckdb::LogicalType> &types) {
  auto collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(
      *GetClientContext(), types);
  duckdb::DataChunk chunk;
  chunk.Initialize(*GetClientContext(), types);
  const uint64_t nrows = table.NumRows();
  for (uint64_t base = 0; base < nrows; base += STANDARD_VECTOR_SIZE) {
    const uint64_t n =
        std::min<uint64_t>(STANDARD_VECTOR_SIZE, nrows - base);
    chunk.Reset();
    for (size_t col = 0; col < table.NumCols(); col++) {
      auto &vec = chunk.data[col];
      auto &validity = duckdb::FlatVector::Validity(vec);
      if (table.Col(col).dtype == AQP_DTYPE_INT32) {
        auto *dst = duckdb::FlatVector::GetData<int32_t>(vec);
        const auto *src = static_cast<const int32_t *>(table.Data(col));
        std::memcpy(dst, src + base, n * sizeof(int32_t));
        for (uint64_t r = 0; r < n; r++)
          if (!table.ValueValid(col, base + r))
            validity.SetInvalid(r);
      } else if (table.Col(col).dtype == AQP_DTYPE_INT64) { // COUNT outputs
        auto *dst = duckdb::FlatVector::GetData<int64_t>(vec);
        const auto *src = static_cast<const int64_t *>(table.Data(col));
        std::memcpy(dst, src + base, n * sizeof(int64_t));
        for (uint64_t r = 0; r < n; r++)
          if (!table.ValueValid(col, base + r))
            validity.SetInvalid(r);
      } else { // AQP_DTYPE_VARCHAR (gated at compile)
        auto *dst = duckdb::FlatVector::GetData<duckdb::string_t>(vec);
        for (uint64_t r = 0; r < n; r++) {
          if (!table.ValueValid(col, base + r)) {
            validity.SetInvalid(r);
            continue;
          }
          QjitString s = table.GetStr(col, base + r);
          dst[r] = duckdb::StringVector::AddString(
              vec, qjit::StringData(s), qjit::StringLen(s));
        }
      }
    }
    chunk.SetCardinality(n);
    collection->Append(chunk);
  }
  return collection;
}

// Find a ScanNode with the given table_index in an IR subtree.
static const ir_sql_converter::AQPStmt *
FindScanByTableIndex(const ir_sql_converter::AQPStmt *ir, unsigned int tidx) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan =
        static_cast<const ir_sql_converter::SimplestScan *>(ir);
    if (scan->GetTableIndex() == tidx)
      return ir;
  }
  for (const auto &child : ir->children)
    if (auto *s = FindScanByTableIndex(child.get(), tidx))
      return s;
  return nullptr;
}

// Find a ScanNode with the given table_index that is a direct descendant
// (not through any nested JoinNode).  This walks through Filter/Projection
// wrappers but stops at JoinNode boundaries.
static const ir_sql_converter::AQPStmt *
FindDirectScanByTableIndex(const ir_sql_converter::AQPStmt *ir,
                           unsigned int tidx) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(ir);
    if (scan->GetTableIndex() == tidx)
      return ir;
  }
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode)
    return nullptr;
  for (const auto &child : ir->children)
    if (auto *s = FindDirectScanByTableIndex(child.get(), tidx))
      return s;
  return nullptr;
}

// Recursive helper: find an unconsumed IR JoinNode whose conditions
// contain a pair (build_tidx, probe_tidx) in either order.
static const ir_sql_converter::AQPStmt *
FindJoinNodeByTablePair(
    const ir_sql_converter::AQPStmt *ir, unsigned int tidx_a,
    unsigned int tidx_b,
    const std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode &&
      !consumed.count(ir) && ir->children.size() > 1) {
    auto *join =
        static_cast<const ir_sql_converter::SimplestJoin *>(ir);
    for (const auto &cond : join->join_conditions) {
      unsigned lt = cond->left_attr->GetTableIndex();
      unsigned rt = cond->right_attr->GetTableIndex();
      if ((lt == tidx_a && rt == tidx_b) ||
          (lt == tidx_b && rt == tidx_a))
        return ir;
    }
  }
  for (const auto &child : ir->children)
    if (auto *j = FindJoinNodeByTablePair(child.get(), tidx_a, tidx_b,
                                          consumed))
      return j;
  return nullptr;
}

// Recursive helper: find an unconsumed IR JoinNode whose conditions
// reference build_tidx and whose direct child (not through nested
// JoinNodes) contains a ScanNode for build_tidx.
static const ir_sql_converter::AQPStmt *
FindJoinNodeByDirectScan(
    const ir_sql_converter::AQPStmt *ir, unsigned int build_tidx,
    const std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::JoinNode &&
      !consumed.count(ir) && ir->children.size() > 1) {
    bool direct_child = FindDirectScanByTableIndex(ir->children[0].get(),
                                                   build_tidx) ||
                        FindDirectScanByTableIndex(ir->children[1].get(),
                                                   build_tidx);
    if (direct_child) {
      auto *join =
          static_cast<const ir_sql_converter::SimplestJoin *>(ir);
      for (const auto &cond : join->join_conditions) {
        if (cond->left_attr->GetTableIndex() == build_tidx ||
            cond->right_attr->GetTableIndex() == build_tidx)
          return ir;
      }
    }
  }
  for (const auto &child : ir->children)
    if (auto *j = FindJoinNodeByDirectScan(child.get(), build_tidx, consumed))
      return j;
  return nullptr;
}

// Find the IR JoinNode for a physical HASH_JOIN.  Matching strategy:
//  1. If both build and probe table indices are known, find the JoinNode
//     whose conditions reference both (exact match).
//  2. Fall back to build_tidx-only with direct-child constraint.
static const ir_sql_converter::AQPStmt *
FindJoinNodeByBuildTableIndex(
    const ir_sql_converter::AQPStmt *ir, unsigned int build_tidx,
    const std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed,
    unsigned int probe_tidx = UINT_MAX) {
  // Phase 1: exact match by table pair.
  if (probe_tidx != UINT_MAX) {
    if (auto *exact = FindJoinNodeByTablePair(ir, build_tidx, probe_tidx,
                                              consumed))
      return exact;
  }
  // Phase 2: direct-child + condition match.
  if (auto *direct = FindJoinNodeByDirectScan(ir, build_tidx, consumed))
    return direct;
  // No looser fallback: a subtree-only match can pick a JoinNode whose
  // conditions reference unrelated tables (the synth_keys mismatch bug).
  return nullptr;
}

// Recursively find the first AggregateNode in the IR tree.
static const ir_sql_converter::AQPStmt *
FindFirstAggregateNode(const ir_sql_converter::AQPStmt *ir) {
  if (!ir)
    return nullptr;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::AggregateNode)
    return ir;
  for (const auto &child : ir->children)
    if (auto *a = FindFirstAggregateNode(child.get()))
      return a;
  return nullptr;
}

// Recursively find the first table index referenced by any leaf attribute
// in an AQPExpr tree.  Returns UINT_MAX if no attribute is found.
static unsigned int
FirstTableIdxFromExpr(const ir_sql_converter::AQPExpr *expr) {
  if (!expr)
    return UINT_MAX;
  using ir_sql_converter::SimplestNodeType;
  switch (expr->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
    return c->attr ? c->attr->GetTableIndex() : UINT_MAX;
  }
  case SimplestNodeType::IsNullExprNode: {
    auto *n = static_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    return n->attr ? n->attr->GetTableIndex() : UINT_MAX;
  }
  case SimplestNodeType::InExprNode: {
    auto *i = static_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    return i->attr ? i->attr->GetTableIndex() : UINT_MAX;
  }
  case SimplestNodeType::LogicalExprNode: {
    auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    unsigned int t = FirstTableIdxFromExpr(l->left_expr.get());
    if (t != UINT_MAX)
      return t;
    return FirstTableIdxFromExpr(l->right_expr.get());
  }
  default:
    return UINT_MAX;
  }
}

// Walk the IR tree and collect the set of table_index values from ScanNodes.
static void
CollectIRTableIndices(const ir_sql_converter::AQPStmt *ir,
                      std::unordered_set<unsigned int> &out) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(ir);
    out.insert(scan->GetTableIndex());
  }
  for (const auto &child : ir->children)
    CollectIRTableIndices(child.get(), out);
}

// Build table_name → IR table_index map.  If a name appears more than once
// (self-join within the sub-plan), it is moved to `ambiguous` so callers
// know not to use name-based matching for that table.
static void
CollectIRTableNameToIndex(const ir_sql_converter::AQPStmt *ir,
                          std::unordered_map<std::string, unsigned int> &out,
                          std::unordered_set<std::string> &ambiguous) {
  if (!ir)
    return;
  if (ir->GetNodeType() == ir_sql_converter::SimplestNodeType::ScanNode) {
    auto *scan = static_cast<const ir_sql_converter::SimplestScan *>(ir);
    const auto &name = scan->GetTableName();
    auto it = out.find(name);
    if (it != out.end()) {
      if (it->second != scan->GetTableIndex())
        ambiguous.insert(name);
    } else {
      out[name] = scan->GetTableIndex();
    }
  }
  for (const auto &child : ir->children)
    CollectIRTableNameToIndex(child.get(), out, ambiguous);
}

// Collect (table_idx, col_idx) pairs referenced by an IR filter's quals.
static void
CollectIRFilterCols(const ir_sql_converter::AQPStmt *filter_ir,
                    std::set<std::pair<unsigned int, unsigned int>> &out) {
  using ir_sql_converter::SimplestNodeType;
  std::function<void(const ir_sql_converter::AQPExpr *)> walk =
      [&](const ir_sql_converter::AQPExpr *expr) {
        if (!expr)
          return;
        switch (expr->GetNodeType()) {
        case SimplestNodeType::VarConstComparisonNode: {
          auto *c =
              static_cast<const ir_sql_converter::SimplestVarConstComparison *>(
                  expr);
          if (c->attr)
            out.emplace(static_cast<unsigned int>(c->attr->GetTableIndex()),
                        static_cast<unsigned int>(c->attr->GetColumnIndex()));
          break;
        }
        case SimplestNodeType::VarComparisonNode: {
          auto *v =
              static_cast<const ir_sql_converter::SimplestVarComparison *>(
                  expr);
          if (v->left_attr)
            out.emplace(
                static_cast<unsigned int>(v->left_attr->GetTableIndex()),
                static_cast<unsigned int>(v->left_attr->GetColumnIndex()));
          if (v->right_attr)
            out.emplace(
                static_cast<unsigned int>(v->right_attr->GetTableIndex()),
                static_cast<unsigned int>(v->right_attr->GetColumnIndex()));
          break;
        }
        case SimplestNodeType::IsNullExprNode: {
          auto *n =
              static_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
          if (n->attr)
            out.emplace(static_cast<unsigned int>(n->attr->GetTableIndex()),
                        static_cast<unsigned int>(n->attr->GetColumnIndex()));
          break;
        }
        case SimplestNodeType::InExprNode: {
          auto *i = static_cast<const ir_sql_converter::SimplestInExpr *>(expr);
          if (i->attr)
            out.emplace(static_cast<unsigned int>(i->attr->GetTableIndex()),
                        static_cast<unsigned int>(i->attr->GetColumnIndex()));
          break;
        }
        case SimplestNodeType::LogicalExprNode: {
          auto *l =
              static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
          walk(l->left_expr.get());
          walk(l->right_expr.get());
          break;
        }
        default:
          break;
        }
      };
  for (const auto &q : filter_ir->qual_vec)
    walk(q.get());
}

// Collect chunk column indices referenced by a DuckDB Expression tree.
static void CollectBoundRefIndices(const duckdb::Expression &expr,
                                   std::set<duckdb::idx_t> &out) {
  if (expr.expression_class == duckdb::ExpressionClass::BOUND_REF) {
    auto &ref = expr.Cast<duckdb::BoundReferenceExpression>();
    out.insert(ref.index);
    return;
  }
  duckdb::ExpressionIterator::EnumerateChildren(
      expr, [&](const duckdb::Expression &child) {
        CollectBoundRefIndices(child, out);
      });
}

// Returns true if the expr (recursively) references a (table_idx, col_idx) pair
// that exists in schema.  Used to skip JIT for filters whose predicates don't
// map to any column in the physical operator's output chunk.
static bool ExprHasColInSchema(const ir_sql_converter::AQPExpr *expr,
                               const std::vector<aqp_jit::ColSchema> &schema) {
  if (!expr)
    return false;
  using ir_sql_converter::SimplestNodeType;
  auto attrInSchema = [&](unsigned int t, unsigned int col) -> bool {
    for (auto &cs : schema)
      if (cs.table_idx == t && cs.col_idx == col)
        return true;
    return false;
  };
  switch (expr->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
    return c->attr &&
           attrInSchema(static_cast<unsigned int>(c->attr->GetTableIndex()),
                        static_cast<unsigned int>(c->attr->GetColumnIndex()));
  }
  case SimplestNodeType::IsNullExprNode: {
    auto *n = static_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    return n->attr &&
           attrInSchema(static_cast<unsigned int>(n->attr->GetTableIndex()),
                        static_cast<unsigned int>(n->attr->GetColumnIndex()));
  }
  case SimplestNodeType::InExprNode: {
    auto *i = static_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    return i->attr &&
           attrInSchema(static_cast<unsigned int>(i->attr->GetTableIndex()),
                        static_cast<unsigned int>(i->attr->GetColumnIndex()));
  }
  case SimplestNodeType::LogicalExprNode: {
    auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    return ExprHasColInSchema(l->left_expr.get(), schema) ||
           ExprHasColInSchema(l->right_expr.get(), schema);
  }
  default:
    return false;
  }
}

static bool
HasApplicablePredicate(const ir_sql_converter::AQPStmt *filter_ir,
                       const std::vector<aqp_jit::ColSchema> &schema) {
  for (const auto &q : filter_ir->qual_vec)
    if (ExprHasColInSchema(q.get(), schema))
      return true;
  return false;
}

static bool ExprHasLike(const ir_sql_converter::AQPExpr *expr) {
  if (!expr) return false;
  using ir_sql_converter::SimplestNodeType;
  switch (expr->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
    auto et = c->GetSimplestExprType();
    return et == ir_sql_converter::SimplestExprType::TextLike ||
           et == ir_sql_converter::SimplestExprType::Text_Not_Like;
  }
  case SimplestNodeType::LogicalExprNode: {
    auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    return ExprHasLike(l->left_expr.get()) ||
           ExprHasLike(l->right_expr.get());
  }
  default:
    return false;
  }
}

static bool FilterHasLike(const ir_sql_converter::AQPStmt *filter_ir) {
  for (const auto &q : filter_ir->qual_vec)
    if (ExprHasLike(q.get()))
      return true;
  return false;
}

static bool ExprIsOnlyLike(const ir_sql_converter::AQPExpr *expr) {
  if (!expr) return true;
  using ir_sql_converter::SimplestNodeType;
  switch (expr->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
    auto et = c->GetSimplestExprType();
    return et == ir_sql_converter::SimplestExprType::TextLike ||
           et == ir_sql_converter::SimplestExprType::Text_Not_Like;
  }
  case SimplestNodeType::LogicalExprNode: {
    auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    return ExprIsOnlyLike(l->left_expr.get()) &&
           ExprIsOnlyLike(l->right_expr.get());
  }
  default:
    return false;
  }
}

static bool FilterIsOnlyLike(const ir_sql_converter::AQPStmt *filter_ir) {
  if (!filter_ir || filter_ir->qual_vec.empty()) return false;
  for (const auto &q : filter_ir->qual_vec)
    if (!ExprIsOnlyLike(q.get())) return false;
  return true;
}

// Returns true only if ALL columns referenced by the expression are present in
// the schema.  If any column is missing, the compiled filter would silently
// replace that sub-expression with "pass-all" (true), corrupting the filter
// logic (e.g. "gender='m' OR (gender='f' AND name LIKE 'B%')" becomes
// "gender='m' OR gender='f'" = all rows pass when 'name' col is absent).
static bool
AllColsInExprAvailable(const ir_sql_converter::AQPExpr *expr,
                       const std::vector<aqp_jit::ColSchema> &schema) {
  if (!expr)
    return true;
  using ir_sql_converter::SimplestNodeType;
  auto inSchema = [&](unsigned int t, unsigned int col) -> bool {
    for (auto &cs : schema)
      if (cs.table_idx == t && cs.col_idx == col)
        return true;
    return false;
  };
  switch (expr->GetNodeType()) {
  case SimplestNodeType::VarConstComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarConstComparison *>(expr);
    return c->attr &&
           inSchema(static_cast<unsigned int>(c->attr->GetTableIndex()),
                    static_cast<unsigned int>(c->attr->GetColumnIndex()));
  }
  case SimplestNodeType::IsNullExprNode: {
    auto *n = static_cast<const ir_sql_converter::SimplestIsNullExpr *>(expr);
    return n->attr &&
           inSchema(static_cast<unsigned int>(n->attr->GetTableIndex()),
                    static_cast<unsigned int>(n->attr->GetColumnIndex()));
  }
  case SimplestNodeType::InExprNode: {
    auto *i = static_cast<const ir_sql_converter::SimplestInExpr *>(expr);
    return i->attr &&
           inSchema(static_cast<unsigned int>(i->attr->GetTableIndex()),
                    static_cast<unsigned int>(i->attr->GetColumnIndex()));
  }
  case SimplestNodeType::LogicalExprNode: {
    auto *l = static_cast<const ir_sql_converter::SimplestLogicalExpr *>(expr);
    return AllColsInExprAvailable(l->left_expr.get(), schema) &&
           AllColsInExprAvailable(l->right_expr.get(), schema);
  }
  case SimplestNodeType::VarComparisonNode: {
    auto *c =
        static_cast<const ir_sql_converter::SimplestVarComparison *>(expr);
    return c->left_attr &&
           inSchema(
               static_cast<unsigned int>(c->left_attr->GetTableIndex()),
               static_cast<unsigned int>(c->left_attr->GetColumnIndex())) &&
           c->right_attr &&
           inSchema(static_cast<unsigned int>(c->right_attr->GetTableIndex()),
                    static_cast<unsigned int>(c->right_attr->GetColumnIndex()));
  }
  default:
    return true;
  }
}

static bool
FilterAllColsAvailable(const ir_sql_converter::AQPStmt *filter_ir,
                       const std::vector<aqp_jit::ColSchema> &schema) {
  for (const auto &q : filter_ir->qual_vec)
    if (!AllColsInExprAvailable(q.get(), schema))
      return false;
  return true;
}

void DuckDBAdapter::EnsureJITCompiler() {
  if (compiler_precreate_future_.valid())
    compiler_precreate_future_.get();
  auto want_fast = static_cast<aqp_jit::FastCompileBackend>(compile_mode_);
  if (jit_compiler_ && jit_compiler_->GetFastMode() != want_fast) {
    auto old_key = CompilerCacheKey(
        static_cast<int>(jit_compiler_->GetFastMode()));
    jit_compiler_cache_[old_key] = {std::move(jit_compiler_),
                                    qjit_syms_registered_};
    qjit_syms_registered_ = false;
    fast_jit_compiler_.reset();
    fast_qjit_syms_registered_ = false;
  }
  if (!jit_compiler_) {
    auto new_key = CompilerCacheKey(static_cast<int>(want_fast));
    auto it = jit_compiler_cache_.find(new_key);
    if (it != jit_compiler_cache_.end()) {
      jit_compiler_ = std::move(it->second.compiler);
      qjit_syms_registered_ = it->second.qjit_syms_registered;
      jit_compiler_cache_.erase(it);
    } else {
      jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(
          jit_debug_, want_fast);
    }
  }
  jit_compiler_->SetPrefetch(jit_prefetch_, jit_prefetch_distance_);
  jit_compiler_->SetProbePrefetchDistances(jit_prefetch_entry_distance_,
                                           jit_prefetch_row_distance_);
  jit_compiler_->SetBatchProbe(jit_batch_probe_);
  jit_compiler_->SetSkipHashCmp(jit_skip_hash_cmp_);
  jit_compiler_->SetDebugRuntimeStatBehavior(debug_runtime_stat_behavior_);
  if (jit_cache_)
    jit_compiler_->SetCache(jit_cache_);
}

void DuckDBAdapter::PreCreateCompilers(const std::vector<int> &backends) {
  for (int fast : backends) {
    auto key = CompilerCacheKey(fast);
    if (jit_compiler_cache_.count(key))
      continue;
    if (jit_compiler_ &&
        static_cast<int>(jit_compiler_->GetFastMode()) == fast)
      continue;
    auto fast_be = static_cast<aqp_jit::FastCompileBackend>(fast);
    auto comp = std::make_unique<aqp_jit::IrToLlvmCompiler>(
        jit_debug_, fast_be);
    comp->SetPrefetch(jit_prefetch_, jit_prefetch_distance_);
    comp->SetProbePrefetchDistances(jit_prefetch_entry_distance_,
                                    jit_prefetch_row_distance_);
    comp->SetBatchProbe(jit_batch_probe_);
    comp->SetSkipHashCmp(jit_skip_hash_cmp_);
    comp->SetDebugRuntimeStatBehavior(debug_runtime_stat_behavior_);
    if (jit_cache_)
      comp->SetCache(jit_cache_);
    jit_compiler_cache_[key] = {std::move(comp), false};
  }
}

void DuckDBAdapter::PreCreateCompilersAsync(const std::vector<int> &backends) {
  if (compiler_precreate_future_.valid())
    compiler_precreate_future_.get();
  compiler_precreate_future_ = std::async(
      std::launch::async,
      [this, b = backends] { PreCreateCompilers(b); });
}

aqp_jit::IrToLlvmCompiler *DuckDBAdapter::GetJitCompiler() {
  EnsureJITCompiler();
  return jit_compiler_.get();
}

aqp_jit::IrToLlvmCompiler *DuckDBAdapter::EnsureFastJitCompiler() {
  // Miss recompiles use the same backend as the main compiler so that
  // compile-mode=llvm_4opt/llvm_o0 configs don't silently fall back to TPDE.
  if (compile_mode_ == static_cast<int>(aqp_jit::FastCompileBackend::TPDE))
    return GetJitCompiler();
  auto want = static_cast<aqp_jit::FastCompileBackend>(compile_mode_);
  if (!fast_jit_compiler_ || fast_jit_compiler_->GetFastMode() != want) {
    fast_jit_compiler_ = std::make_unique<aqp_jit::IrToLlvmCompiler>(
        jit_debug_, want);
  }
  fast_jit_compiler_->SetPrefetch(jit_prefetch_, jit_prefetch_distance_);
  fast_jit_compiler_->SetProbePrefetchDistances(jit_prefetch_entry_distance_,
                                                jit_prefetch_row_distance_);
  fast_jit_compiler_->SetBatchProbe(jit_batch_probe_);
  fast_jit_compiler_->SetSkipHashCmp(jit_skip_hash_cmp_);
  fast_jit_compiler_->SetDebugRuntimeStatBehavior(debug_runtime_stat_behavior_);
  if (jit_cache_)
    fast_jit_compiler_->SetCache(jit_cache_);
  return fast_jit_compiler_.get();
}

void DuckDBAdapter::RegisterJIT(duckdb::PhysicalOperator &op,
                                const ir_sql_converter::AQPStmt &ir) {
  EnsureJITCompiler();
  jit_consumed_ir_filters_.clear();
  jit_consumed_ir_joins_.clear();
  RegisterJITImpl(op, ir, GetClientContext(), jit_compiler_.get(),
                  jit_consumed_ir_filters_, jit_consumed_ir_joins_);
}

DuckDBAdapter::TempCollectionSnapshot
DuckDBAdapter::SnapshotTempCollections() const {
  TempCollectionSnapshot snap;
  for (const auto &kv : intermediate_table_map) {
    auto it = temp_collections_.find(kv.second);
    if (it != temp_collections_.end() && it->second.collection &&
        !it->second.is_placeholder)
      snap[kv.first] = it->second.collection.get();
  }
  return snap;
}

void DuckDBAdapter::RegisterJITImpl(
    duckdb::PhysicalOperator &op, const ir_sql_converter::AQPStmt &ir,
    duckdb::ClientContext *jit_ctx, aqp_jit::IrToLlvmCompiler *jit_comp,
    std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed_filters,
    std::unordered_set<const ir_sql_converter::AQPStmt *> &consumed_joins,
    const TempCollectionSnapshot *bf_temp_snapshot, bool is_build_side) {
  using duckdb::PhysicalOperatorType;

  // Diagnostic: show every node in the physical plan tree
#ifndef NDEBUG
  std::cerr << "[AQP-JIT-TRACE] visit op=" << (int)op.type
            << " children=" << op.children.size() << " addr=" << (void *)&op
            << "\n";
#endif

  if (op.type == PhysicalOperatorType::FILTER && !op.children.empty()) {
    auto &child = op.children[0].get();
    uint64_t eid = duckdb::ExpressionID(op);

#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] FILTER eid=0x" << std::hex << eid << std::dec
              << "  child_type=" << (int)child.type
              << "  child_cols=" << child.types.size() << "\n";
#endif

    // Dump all IR FilterNodes once per FILTER op (to show the full IR
    // structure)
    {
      std::vector<const ir_sql_converter::AQPStmt *> all_filters;
      CollectAllFilterNodes(&ir, all_filters);
#ifndef NDEBUG
      std::cerr << "[AQP-JIT-TRACE] IR has " << all_filters.size()
                << " FilterNode(s):\n";
      for (size_t fi = 0; fi < all_filters.size(); fi++) {
        std::cerr << "[AQP-JIT-TRACE]   IR filter[" << fi
                  << "] qual_vec.size=" << all_filters[fi]->qual_vec.size()
                  << "\n";
        for (size_t qi = 0; qi < all_filters[fi]->qual_vec.size(); qi++) {
          std::string qs = all_filters[fi]->qual_vec[qi]->Print(false);
          std::cerr << "[AQP-JIT-TRACE]     qual[" << qi << "]: " << qs << "\n";
        }
      }
#endif
    }

    // Determine the IR table_idx for this physical scan by matching the DuckDB
    // table name against IR ScanNodes.  This is reliable because each physical
    // TABLE_SCAN in DuckDB corresponds to exactly one logical table whose name
    // is stored in both the DuckDB bind_data and the IR SimplestScan node.
    const ir_sql_converter::AQPStmt *filter_ir = nullptr;
    unsigned int ir_table_idx = UINT_MAX;

    // Collect known IR table indices for debug validation
    std::unordered_set<unsigned int> ir_table_indices;
    CollectIRTableIndices(&ir, ir_table_indices);
    // Name→index fallback for node-based split where sub-SQL gets fresh
    // logical_table_index values that don't match the original IR's indices.
    std::unordered_map<std::string, unsigned int> ir_name_to_idx;
    std::unordered_set<std::string> ir_ambiguous_names;
    CollectIRTableNameToIndex(&ir, ir_name_to_idx, ir_ambiguous_names);
#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] IR table indices:";
    for (auto idx : ir_table_indices)
      std::cerr << " " << idx;
    std::cerr << "\n";
#endif

    std::vector<aqp_jit::ColSchema> schema_prelim;

    // schema/column index mapping

    // The scan's column_ids[i] directly gives the physical column index.
    // Build ColSchema from those.
    if (child.type == PhysicalOperatorType::TABLE_SCAN) {
      auto &scan = static_cast<duckdb::PhysicalTableScan &>(child);

      // Use logical_table_index propagated from the planner — unique per
      // table reference, so self-joins are handled correctly.
      ir_table_idx =
          static_cast<unsigned int>(scan.logical_table_index);
#ifndef NDEBUG
      std::cerr << "[AQP-JIT-TRACE]   scan logical_table_index="
                << ir_table_idx << "\n";
#endif
      // Fallback: node-based split re-parses sub-SQL which assigns fresh
      // table indices.  Match by table name when the index isn't in the IR.
      if (!ir_table_indices.count(ir_table_idx) && scan.bind_data) {
        auto *tsbd =
            dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
        if (tsbd) {
          const auto &tname = tsbd->table.name;
          if (!ir_ambiguous_names.count(tname)) {
            auto it = ir_name_to_idx.find(tname);
            if (it != ir_name_to_idx.end()) {
              ir_table_idx = it->second;
#ifndef NDEBUG
              std::cerr << "[AQP-JIT-TRACE]   name fallback: \"" << tname
                        << "\" → ir_table_idx=" << ir_table_idx << "\n";
#endif
            }
          }
        }
      }
      if (!ir_table_indices.count(ir_table_idx)) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT-TRACE]   WARNING: no IR table matches "
                      "logical_table_index="
                  << ir_table_idx << "\n";
#endif
      }

      {
        // child.types may be shorter than scan.column_ids when DuckDB's
        // filter-prune / projection-pushdown strips columns that are only
        // needed for table filters but not returned in the output chunk.
        //
        // IMPORTANT: child.types[i] represents the ACTUAL data type at chunk
        // position i and must be used for dtype.  However, scan.column_ids[i]
        // may be misaligned with child.types in some physical plans (the
        // column_ids and types vectors can have different orderings).  When
        // that happens, we must find the correct base column ID by matching
        // the actual type against the catalog types of the scanned columns.
        for (size_t i = 0; i < child.types.size(); i++) {
          aqp_jit::ColSchema cs;
          cs.table_idx = ir_table_idx;
          cs.dtype = duckdb::ToDtype(child.types[i].InternalType());
          cs.col_idx =
              static_cast<unsigned int>(scan.column_ids[i].GetPrimaryIndex());

          if (scan.bind_data) {
            auto *tsbd =
                dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
            if (tsbd) {
              // Check if column_ids[i] catalog type matches child.types[i].
              // If not, search for the correct column_id.
              auto n_logical = tsbd->table.GetColumns().LogicalColumnCount();
              if (cs.col_idx < n_logical) {
                auto catalog_dtype = duckdb::ToDtype(
                    tsbd->table.GetColumns()
                        .GetColumn(duckdb::LogicalIndex(cs.col_idx))
                        .Type()
                        .InternalType());
                if (catalog_dtype != cs.dtype) {
                  // Misaligned: find the column_id whose catalog type matches
                  // child.types[i].
                  for (size_t j = 0; j < scan.column_ids.size(); j++) {
                    unsigned int cid = static_cast<unsigned int>(
                        scan.column_ids[j].GetPrimaryIndex());
                    if (cid < n_logical && cid != cs.col_idx) {
                      auto c_dtype = duckdb::ToDtype(
                          tsbd->table.GetColumns()
                              .GetColumn(duckdb::LogicalIndex(cid))
                              .Type()
                              .InternalType());
                      if (c_dtype == cs.dtype) {
                        cs.col_idx = cid;
                        break;
                      }
                    }
                  }
                }
              }
            }
          }
          schema_prelim.push_back(cs);
#ifndef NDEBUG
          // Look up column name from DuckDB catalog for diagnostics
          std::string duckdb_col_name;
          if (scan.bind_data) {
            auto *tsbd =
                dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
            if (tsbd &&
                cs.col_idx < tsbd->table.GetColumns().LogicalColumnCount()) {
              duckdb_col_name = tsbd->table.GetColumns()
                                    .GetColumn(duckdb::LogicalIndex(cs.col_idx))
                                    .Name();
            }
          }
          std::cerr << "[AQP-JIT-TRACE]   scan col[" << i
                    << "] raw_col=" << cs.col_idx << " dtype=" << cs.dtype
                    << " table_idx=" << cs.table_idx << " duckdb_col_name=\""
                    << duckdb_col_name << "\"\n";
#endif
        }

        // Find the IR FilterNode that best matches this DuckDB FILTER.
        // Strategy: collect the chunk column indices referenced by the DuckDB
        // expression, map them to (table_idx, col_idx) pairs, then find the
        // IR filter whose qual columns overlap with those pairs.
        filter_ir = nullptr;
        if (ir_table_idx != UINT_MAX) {
          std::vector<const ir_sql_converter::AQPStmt *> all_filters;
          CollectAllFilterNodes(&ir, all_filters);

          // Columns referenced by DuckDB filter expression → schema col pairs
          auto &filt_op = static_cast<duckdb::PhysicalFilter &>(op);
          std::set<std::pair<unsigned int, unsigned int>> duckdb_cols;
          if (filt_op.expression) {
            std::set<duckdb::idx_t> ref_indices;
            CollectBoundRefIndices(*filt_op.expression, ref_indices);
            for (auto idx : ref_indices) {
              if (idx < schema_prelim.size())
                duckdb_cols.emplace(schema_prelim[idx].table_idx,
                                    schema_prelim[idx].col_idx);
            }
          }

          // Find best-matching IR filter: prefer the one whose cols overlap
          // with the DuckDB expression's cols, falling back to table_idx match.
          // Safety: every column referenced by the IR filter must appear in the
          // DuckDB filter's column set.  If the IR references columns that the
          // physical filter doesn't use, the IR filter is a different predicate
          // (e.g., the optimizer rebinds or transforms columns) and compiling it
          // would corrupt results.
          const ir_sql_converter::AQPStmt *best = nullptr;
          size_t best_overlap = 0;
          for (auto *f : all_filters) {
            if (consumed_filters.count(f))
              continue;
            bool table_match = false;
            for (const auto &q : f->qual_vec) {
              unsigned int t = FirstTableIdxFromExpr(q.get());
              if (t == ir_table_idx) {
                table_match = true;
                break;
              }
            }
            if (!table_match)
              continue;
            if (duckdb_cols.empty()) {
              best = f;
              break;
            }
            std::set<std::pair<unsigned int, unsigned int>> ir_cols;
            CollectIRFilterCols(f, ir_cols);
            size_t overlap = 0;
            bool all_in_duckdb = true;
            for (auto &p : ir_cols) {
              if (duckdb_cols.count(p))
                overlap++;
              else
                all_in_duckdb = false;
            }
            if (!all_in_duckdb)
              continue;
            if (overlap > best_overlap) {
              best_overlap = overlap;
              best = f;
            }
          }
          filter_ir = best;
        }
      }
    } else if (child.type == PhysicalOperatorType::PROJECTION &&
               !child.children.empty() &&
               child.children[0].get().type ==
                   PhysicalOperatorType::TABLE_SCAN) {
      // PROJECTION → TABLE_SCAN: trace projection expressions back to original
      // scan column indices.  PhysicalProjection::select_list[i] is a
      // BoundReferenceExpression with index=j meaning "output col i = scan
      // input col j", so the original table col idx = scan.column_ids[j].
      auto &proj = static_cast<duckdb::PhysicalProjection &>(child);
      auto &scan =
          static_cast<duckdb::PhysicalTableScan &>(child.children[0].get());

      // Use logical_table_index from the inner scan (same as TABLE_SCAN path)
      ir_table_idx =
          static_cast<unsigned int>(scan.logical_table_index);
#ifndef NDEBUG
      std::cerr << "[AQP-JIT-TRACE]   proj→scan logical_table_index="
                << ir_table_idx << "\n";
#endif
      // Fallback: same as TABLE_SCAN path — match by name for node-based split.
      if (!ir_table_indices.count(ir_table_idx) && scan.bind_data) {
        auto *tsbd =
            dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get());
        if (tsbd) {
          const auto &tname = tsbd->table.name;
          if (!ir_ambiguous_names.count(tname)) {
            auto it = ir_name_to_idx.find(tname);
            if (it != ir_name_to_idx.end()) {
              ir_table_idx = it->second;
#ifndef NDEBUG
              std::cerr << "[AQP-JIT-TRACE]   proj→scan name fallback: \""
                        << tname << "\" → ir_table_idx=" << ir_table_idx
                        << "\n";
#endif
            }
          }
        }
      }
      if (!ir_table_indices.count(ir_table_idx)) {
#ifndef NDEBUG
        std::cerr << "[AQP-JIT-TRACE]   WARNING: proj→scan no IR table "
                      "matches logical_table_index="
                  << ir_table_idx << "\n";
#endif
      }

      {

        // Build schema by tracing each projection expression back to scan col
        for (size_t i = 0; i < proj.select_list.size(); i++) {
          aqp_jit::ColSchema cs;
          cs.table_idx = ir_table_idx;
          cs.col_idx = UINT_MAX; // set below if traceable
          // Projection output type is correct for the actual chunk layout.
          cs.dtype = duckdb::ToDtype(child.types[i].InternalType());

          auto &expr = *proj.select_list[i];
          if (expr.GetExpressionClass() == duckdb::ExpressionClass::BOUND_REF) {
            auto &ref = expr.Cast<duckdb::BoundReferenceExpression>();
            if (ref.index < scan.column_ids.size()) {
              cs.col_idx = static_cast<unsigned int>(
                  scan.column_ids[ref.index].GetPrimaryIndex());
              // column_ids may be misaligned with the actual chunk data.
              // Verify the catalog type matches the projection output type;
              // if not, find the correct column_id.
              if (scan.bind_data) {
                auto *tsbd = dynamic_cast<duckdb::TableScanBindData *>(
                    scan.bind_data.get());
                if (tsbd) {
                  auto n_logical =
                      tsbd->table.GetColumns().LogicalColumnCount();
                  if (cs.col_idx < n_logical) {
                    auto catalog_dtype = duckdb::ToDtype(
                        tsbd->table.GetColumns()
                            .GetColumn(duckdb::LogicalIndex(cs.col_idx))
                            .Type()
                            .InternalType());
                    if (catalog_dtype != cs.dtype) {
                      for (size_t j = 0; j < scan.column_ids.size(); j++) {
                        unsigned int cid = static_cast<unsigned int>(
                            scan.column_ids[j].GetPrimaryIndex());
                        if (cid < n_logical && cid != cs.col_idx) {
                          auto c_dtype = duckdb::ToDtype(
                              tsbd->table.GetColumns()
                                  .GetColumn(duckdb::LogicalIndex(cid))
                                  .Type()
                                  .InternalType());
                          if (c_dtype == cs.dtype) {
                            cs.col_idx = cid;
                            break;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          schema_prelim.push_back(cs);
#ifndef NDEBUG
          std::cerr << "[AQP-JIT-TRACE]   proj col[" << i
                    << "] raw_col=" << cs.col_idx << " dtype=" << cs.dtype
                    << " table_idx=" << cs.table_idx << "\n";
#endif
        }

        // Find best-matching IR filter (same column-overlap strategy).
        filter_ir = nullptr;
        if (ir_table_idx != UINT_MAX) {
          std::vector<const ir_sql_converter::AQPStmt *> all_filters;
          CollectAllFilterNodes(&ir, all_filters);

          auto &filt_op = static_cast<duckdb::PhysicalFilter &>(op);
          std::set<std::pair<unsigned int, unsigned int>> duckdb_cols;
          if (filt_op.expression) {
            std::set<duckdb::idx_t> ref_indices;
            CollectBoundRefIndices(*filt_op.expression, ref_indices);
            for (auto idx : ref_indices) {
              if (idx < schema_prelim.size())
                duckdb_cols.emplace(schema_prelim[idx].table_idx,
                                    schema_prelim[idx].col_idx);
            }
          }

          const ir_sql_converter::AQPStmt *best = nullptr;
          size_t best_overlap = 0;
          for (auto *f : all_filters) {
            if (consumed_filters.count(f))
              continue;
            bool table_match = false;
            for (const auto &q : f->qual_vec) {
              unsigned int t = FirstTableIdxFromExpr(q.get());
              if (t == ir_table_idx) {
                table_match = true;
                break;
              }
            }
            if (!table_match)
              continue;
            if (duckdb_cols.empty()) {
              best = f;
              break;
            }
            std::set<std::pair<unsigned int, unsigned int>> ir_cols;
            CollectIRFilterCols(f, ir_cols);
            size_t overlap = 0;
            bool all_in_duckdb = true;
            for (auto &p : ir_cols) {
              if (duckdb_cols.count(p))
                overlap++;
              else
                all_in_duckdb = false;
            }
            if (!all_in_duckdb)
              continue;
            if (overlap > best_overlap) {
              best_overlap = overlap;
              best = f;
            }
          }
          filter_ir = best;
        }
      } // end proj path schema/filter matching
    } else {
      // Unknown child type (HASH_JOIN result, etc.): skip JIT.
      // schema_prelim stays empty → HasApplicablePredicate won't match → JIT
      // skipped.
#ifndef NDEBUG
      std::cerr << "[AQP-JIT-TRACE]   child_type=" << (int)child.type
                << " not JIT-able → interpreter\n";
#endif
    }

    if (!filter_ir) {
#ifndef NDEBUG
      std::cerr << "[AQP-JIT] FILTER eid=0x" << std::hex << eid << std::dec
                << ": no Filter IR node found in tree, using interpreter\n";
#endif
      for (auto &child_ref : op.children)
        RegisterJITImpl(child_ref.get(), ir, jit_ctx, jit_comp,
                        consumed_filters, consumed_joins, bf_temp_snapshot);
      return;
    }

#ifndef NDEBUG
    std::cerr << "[AQP-JIT-TRACE] filter_ir node_type="
              << (int)filter_ir->GetNodeType()
              << "  qual_vec.size=" << filter_ir->qual_vec.size()
              << "  ir_table_idx=" << ir_table_idx << "\n";
    for (size_t qi = 0; qi < filter_ir->qual_vec.size(); qi++) {
      std::string qs = filter_ir->qual_vec[qi]->Print(false);
      std::cerr << "[AQP-JIT-TRACE]   selected qual[" << qi << "]: " << qs
                << "\n";
    }
#endif

    // Build final ColSchema from schema_prelim.
    // For TABLE_SCAN and PROJECTION→TABLE_SCAN: schema_prelim already has
    // correct (table_idx, col_idx, dtype) from the DuckDB table-name → IR
    // table_idx lookup. For unknown child types: schema_prelim is empty → JIT
    // skipped below.
    std::vector<aqp_jit::ColSchema> schema = schema_prelim;
#ifndef NDEBUG
    for (size_t i = 0; i < schema.size(); i++)
      std::cerr << "[AQP-JIT-TRACE]   schema[" << i
                << "] table=" << schema[i].table_idx
                << " col=" << schema[i].col_idx << " dtype=" << schema[i].dtype
                << "\n";
#endif

    // Note: VARCHAR predicates (LIKE, NOT LIKE, Equal, NotEqual) are now
    // handled by the JIT via aqp_like_match / aqp_str_eq runtime helpers.

    // Only register JIT if at least one predicate references a column in the
    // schema AND every referenced column is available.  If any column is
    // missing (e.g. DuckDB pruned it from the scan output), the compiled
    // filter would silently replace that sub-expression with "pass-all",
    // corrupting the filter logic.  Fall back to interpreter in that case.
    // Keep expr-level JIT for LIKE filters: the selection-vector path benefits
    // mixed filters (LIKE + non-LIKE predicates). Scan+filter fusion and
    // pipeline filter fn are skipped for LIKE below (guards at operator/pipeline
    // level) because the per-row JIT LIKE is ~2x slower than DuckDB's vectorized
    // LIKE on large tables.
    bool skip_like = FilterIsOnlyLike(filter_ir);
    if (!schema.empty() && HasApplicablePredicate(filter_ir, schema) &&
        FilterAllColsAvailable(filter_ir, schema) &&
        !skip_like) {
      auto t_filter = chrono_tic();
      std::vector<uint8_t> filter_params;
      ::AQPExprFn raw_fn = jit_comp->CompileFilter(*filter_ir, schema, &filter_params);
      if (enable_timing_ && BREAK_DOWN_COMPILE_TIME) {
        chrono_toc(&t_filter, "RegisterJIT::CompileFilter\n", false);
      }
      if (raw_fn) {
        duckdb::AQPExprFn fn = reinterpret_cast<duckdb::AQPExprFn>(
            reinterpret_cast<void *>(raw_fn));
        if (!jit_ctx->aqp_jit_context)
          jit_ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
        consumed_filters.insert(filter_ir);

        jit_ctx->aqp_jit_context->expr_fns[eid] = fn;
        if (!filter_params.empty())
          jit_ctx->aqp_jit_context->expr_params[eid] = std::move(filter_params);
        jit_ctx->aqp_jit_context->flags |= duckdb::AQPJIT_EXPR;

#ifndef NDEBUG
        std::cerr << "[AQP-JIT] compiled filter eid=0x" << std::hex << eid
                  << std::dec << "  cols=" << schema.size()
                  << "  table_idx=" << ir_table_idx << "\n";
#endif
      } else {
        std::ostringstream msg;
        msg << "[AQP-JIT] JIT compilation failed (eid=0x" << std::hex << eid
            << std::dec << "  cols=" << schema.size()
            << "  table_idx=" << ir_table_idx
            << "). Stopping. Use --no-jit to fall back to interpreter.";
        throw std::runtime_error(msg.str());
      }
    } else if (!schema.empty()) {
#ifndef NDEBUG
      if (!FilterAllColsAvailable(filter_ir, schema)) {
        std::cerr << "[AQP-JIT] skipping filter eid=0x" << std::hex << eid
                  << std::dec
                  << ": not all columns available in chunk → interpreter\n";
      } else {
        std::cerr << "[AQP-JIT] skipping filter eid=0x" << std::hex << eid
                  << std::dec
                  << ": no predicates reference schema columns → interpreter\n";
      }
#endif
    }

  }



  // Recurse into children.
  // children[1] of a HASH_JOIN is the build side — pass is_build_side=true
  // so multi-probe fusion is disabled for chains on build pipelines.
  for (size_t ci = 0; ci < op.children.size(); ci++) {
    bool child_is_build = (op.type == PhysicalOperatorType::HASH_JOIN && ci == 1);
    RegisterJITImpl(op.children[ci].get(), ir, jit_ctx, jit_comp,
                    consumed_filters, consumed_joins, bf_temp_snapshot,
                    child_is_build);
  }
}

void DuckDBAdapter::InjectTempTableJoinStats(duckdb::PhysicalOperator &op) {
  using duckdb::PhysicalOperatorType;

  if (op.type == PhysicalOperatorType::HASH_JOIN) {
    auto &join = static_cast<duckdb::PhysicalHashJoin &>(op);

    if (join.join_type == duckdb::JoinType::INNER &&
        join.conditions.size() == 1 &&
        join.conditions[0].comparison == duckdb::ExpressionType::COMPARE_EQUAL &&
        join.join_stats.empty()) {

      auto &build_child = join.children[1].get();
      auto key_type = join.conditions[0].right->return_type;
      if (!duckdb::TypeIsIntegral(key_type.InternalType()))
        goto recurse;

      // Guard: no FILTER on build path (filters change effective key range)
      std::function<bool(duckdb::PhysicalOperator &)> has_filter =
          [&](duckdb::PhysicalOperator &node) -> bool {
        if (node.type == PhysicalOperatorType::FILTER) return true;
        if (node.type == PhysicalOperatorType::TABLE_SCAN) return false;
        for (auto &child : node.children)
          if (has_filter(child.get())) return true;
        return false;
      };
      if (has_filter(build_child))
        goto recurse;

      // Find temp table scan on build path
      std::function<std::string(duckdb::PhysicalOperator &)> find_temp_scan =
          [&](duckdb::PhysicalOperator &node) -> std::string {
        if (node.type == PhysicalOperatorType::TABLE_SCAN) {
          auto &scan = static_cast<duckdb::PhysicalTableScan &>(node);
          auto *bd = scan.bind_data
              ? dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get())
              : nullptr;
          if (bd && bd->table.name.find("temp") == 0)
            return bd->table.name;
        }
        for (auto &child : node.children) {
          auto name = find_temp_scan(child.get());
          if (!name.empty()) return name;
        }
        return "";
      };

      std::string tt_name = find_temp_scan(build_child);
      if (tt_name.empty())
        goto recurse;

      static constexpr int64_t MAX_BUILD_SIZE = 1048576;
      uint64_t tt_card = GetTempTableCardinality(tt_name);
      if (tt_card > static_cast<uint64_t>(MAX_BUILD_SIZE))
        goto recurse;

      auto min_max = GetTempTableMinMax(tt_name, {}, {});
      // Find which column in the build scan corresponds to the join key.
      // Use the right-side condition expression to find the column index.
      // For simplicity, try all integer columns and check if any has a
      // range < MAX_BUILD_SIZE.
      for (auto it = min_max.begin(); it != min_max.end(); ++it) {
        int64_t range_min = it->second.first;
        int64_t range_max = it->second.second;
        if (range_max < range_min) continue;
        int64_t range = range_max - range_min;
        if (range > MAX_BUILD_SIZE) continue;

        // Create stats and inject
        auto stats = duckdb::BaseStatistics::CreateEmpty(key_type);
        duckdb::NumericStats::SetMin(stats, duckdb::Value::Numeric(key_type, range_min));
        duckdb::NumericStats::SetMax(stats, duckdb::Value::Numeric(key_type, range_max));
        stats.SetHasNoNull();

        join.join_stats.resize(2);
        join.join_stats[1] = stats.ToUnique();
        break;
      }
    }
  }

recurse:
  for (auto &child_ref : op.children)
    InjectTempTableJoinStats(child_ref.get());
}

void DuckDBAdapter::RegisterBloomFilters(duckdb::PhysicalOperator &op,
                                         duckdb::ClientContext *ctx,
                                         std::vector<BloomFilterInfo> &bfs) {
  using duckdb::PhysicalOperatorType;
  if (bfs.empty()) return;

  if (op.type == PhysicalOperatorType::TABLE_SCAN) {
    auto &scan = static_cast<duckdb::PhysicalTableScan &>(op);
    auto *tsbd = scan.bind_data
        ? dynamic_cast<duckdb::TableScanBindData *>(scan.bind_data.get())
        : nullptr;
    if (!tsbd) return;
    const auto &tname = tsbd->table.name;

    for (auto &bf_info : bfs) {
      if (bf_info.bf_data.empty() || bf_info.base_table_name != tname)
        continue;

      // Find which chunk column index corresponds to bf_info.base_col_name.
      // With filter_prune, the chunk has projection_ids.size() columns
      // (a subset of column_ids). We must map column_ids position → chunk
      // position via projection_ids.
      auto &cols = tsbd->table.GetColumns();
      uint32_t chunk_col_idx = UINT32_MAX;
      int32_t dtype = AQP_DTYPE_INT32;
      for (size_t i = 0; i < scan.column_ids.size() && i < scan.types.size(); i++) {
        auto physical_col = scan.column_ids[i].GetPrimaryIndex();
        if (physical_col >= cols.LogicalColumnCount()) continue;
        auto &col = cols.GetColumn(duckdb::LogicalIndex(physical_col));
        if (col.Name() != bf_info.base_col_name) continue;
        auto pt = col.Type().InternalType();
        if (pt != duckdb::PhysicalType::INT32 && pt != duckdb::PhysicalType::INT64)
          break;
        dtype = (pt == duckdb::PhysicalType::INT64)
                    ? AQP_DTYPE_INT64
                    : AQP_DTYPE_INT32;
        if (!scan.projection_ids.empty()) {
          for (size_t j = 0; j < scan.projection_ids.size(); j++) {
            if (scan.projection_ids[j] == i) {
              chunk_col_idx = static_cast<uint32_t>(j);
              break;
            }
          }
        } else {
          chunk_col_idx = static_cast<uint32_t>(i);
        }
        break;
      }
      if (chunk_col_idx == UINT32_MAX) continue;

      if (!ctx->aqp_jit_context) {
        ctx->aqp_jit_context = duckdb::make_uniq<duckdb::AQPJITContext>();
      }
      uint64_t scan_eid = duckdb::ExpressionID(scan);
      auto bloom = duckdb::make_uniq<duckdb::AQPJITContext::AQPBloomScanFilter>();
      bloom->bf_data = std::move(bf_info.bf_data);
      bloom->bitmask = bf_info.bitmask;
      bloom->col_idx = chunk_col_idx;
      bloom->dtype = dtype;
      // Only one BF per (scan, column) pair. Multiple BFs on the same column
      // from different temp columns are AND'd, which is too restrictive
      // (e.g., self-join: subject_id and status_id both map to comp_cast_type.id).
      auto &bf_vec = ctx->aqp_jit_context->bloom_scan_filters[scan_eid];
      bool dup = false;
      for (auto &existing : bf_vec) {
        if (existing->col_idx == chunk_col_idx) { dup = true; break; }
      }
      if (!dup) {
        bf_vec.push_back(std::move(bloom));
      }
#ifndef NDEBUG
      std::cerr << "[AQP-BF] Registered bloom filter for " << tname
                << "." << bf_info.base_col_name
                << " col_idx=" << chunk_col_idx
                << " scan_eid=0x" << std::hex << scan_eid << std::dec
                << (dup ? " (SKIPPED dup)" : "") << "\n";
#endif
    }
    return;
  }

  for (auto &child_ref : op.children)
    RegisterBloomFilters(child_ref.get(), ctx, bfs);
}
#endif

duckdb::Binder &DuckDBAdapter::GetBinder() { return *planner->binder; }

duckdb::unique_ptr<duckdb::LogicalOperator> DuckDBAdapter::TakePlan() {
  return std::move(plan);
}

void DuckDBAdapter::SetPlan(duckdb::unique_ptr<duckdb::LogicalOperator> p) {
  plan = std::move(p);
}

void DuckDBAdapter::RegisterExternalTempTable(
    const std::string &temp_name,
    const duckdb::vector<duckdb::LogicalType> &types,
    const std::vector<std::string> &col_names) {

  auto data_chunk_index = planner->binder->GenerateTableIndex();
  intermediate_table_map[data_chunk_index] = temp_name;
  temp_table_index_ = data_chunk_index;
  temp_table_types = types;
  chunk_col_names_[data_chunk_index] = col_names;
}

void DuckDBAdapter::RegisterExternalTempTableWithIndex(
    const std::string &temp_name,
    const duckdb::vector<duckdb::LogicalType> &types,
    const std::vector<std::string> &col_names,
    duckdb::idx_t chunk_index) {
  assert(chunk_index != duckdb::DConstants::INVALID_INDEX);
  intermediate_table_map[chunk_index] = temp_name;
  temp_table_index_ = chunk_index;
  temp_table_types = types;
  chunk_col_names_[chunk_index] = col_names;
}

} // namespace middleware
