/**
 * query_jit_scheduler.h — QjitWorkerPool: morsel-driven parallel-for over
 * fixed worker threads with STABLE worker ids 0..N-1 (per-worker state in
 * the compiled query is indexed by worker_id, which middleware::ThreadPool
 * cannot provide).
 *
 * Model: lingo-db parallel-for. One job at a time; ParallelFor publishes
 * {fn, total, morsel}, workers claim morsels from a shared atomic counter,
 * the caller blocks until the last worker finishes. Workers bump
 * middleware::g_bg_active_threads while running so OpenMP paths throttle.
 *
 * --query-jit-threads=1 uses the same code path (one worker thread), not a
 * special serial mode.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace qjit {

class QjitWorkerPool {
public:
  using MorselBody =
      std::function<void(uint64_t begin, uint64_t end, uint32_t worker_id)>;

  explicit QjitWorkerPool(uint32_t num_workers);
  ~QjitWorkerPool();

  QjitWorkerPool(const QjitWorkerPool &) = delete;
  QjitWorkerPool &operator=(const QjitWorkerPool &) = delete;

  /* Run body over [0, total) in chunks of `morsel` rows. Blocks until all
   * morsels are processed; all workers are quiescent on return (safe to
   * Finalize HTs / merge states / ResetModules afterwards).
   * Must not be called re-entrantly or concurrently. */
  void ParallelFor(uint64_t total, uint64_t morsel, const MorselBody &body);

  uint32_t NumWorkers() const { return num_workers_; }
  bool Idle() const { return active_workers_.load() == 0; }

  /* Progress callback (runs on the CALLER thread, which is otherwise idle
   * inside ParallelFor): invoked once per ParallelFor call when the
   * COMPLETED fraction of [0,total) (rows whose morsel body has returned,
   * not merely claimed rows) first reaches `threshold`. Arguments: that
   * fraction and the milliseconds this ParallelFor call has been running
   * (the basis for extrapolating its remaining time). Set to nullptr to
   * disable. */
  using ProgressCb = std::function<void(double progress, double step_ms)>;
  void SetProgressCallback(ProgressCb cb, double threshold) {
    progress_cb_ = std::move(cb);
    progress_threshold_ = threshold;
  }

private:
  void WorkerLoop(uint32_t worker_id);

  uint32_t num_workers_;
  std::vector<std::thread> workers_;

  std::mutex mutex_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;

  // Job state (published under mutex_, generation bump signals new job).
  const MorselBody *body_ = nullptr;
  uint64_t total_ = 0;
  uint64_t morsel_ = 0;
  uint64_t generation_ = 0;
  std::atomic<uint64_t> next_{0};
  std::atomic<uint64_t> done_rows_{0}; // rows whose morsel body has completed
  std::atomic<uint32_t> active_workers_{0};
  uint32_t finished_workers_ = 0;
  bool shutdown_ = false;
  ProgressCb progress_cb_;
  double progress_threshold_ = 0.5;
};

} // namespace qjit
