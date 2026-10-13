#include "qjit/query_jit_scheduler.h"

#include <cassert>
#include <chrono>

#include "qjit/query_jit_abi.h"
#include "util/thread_pool.h" // middleware::g_bg_active_threads

namespace qjit {

QjitWorkerPool::QjitWorkerPool(uint32_t num_workers)
    : num_workers_(num_workers == 0 ? 1 : num_workers) {
  workers_.reserve(num_workers_);
  for (uint32_t i = 0; i < num_workers_; i++)
    workers_.emplace_back([this, i] { WorkerLoop(i); });
}

QjitWorkerPool::~QjitWorkerPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    generation_++;
  }
  cv_start_.notify_all();
  for (auto &t : workers_)
    t.join();
}

void QjitWorkerPool::ParallelFor(uint64_t total, uint64_t morsel,
                                 const MorselBody &body) {
  assert(morsel > 0);
  if (total == 0)
    return;
  std::unique_lock<std::mutex> lock(mutex_);
  assert(body_ == nullptr && "ParallelFor is not re-entrant");
  body_ = &body;
  total_ = total;
  morsel_ = morsel;
  next_.store(0, std::memory_order_relaxed);
  done_rows_.store(0, std::memory_order_relaxed);
  finished_workers_ = 0;
  generation_++;
  uint64_t gen = generation_;
  cv_start_.notify_all();
  auto done = [this, gen] {
    return generation_ == gen && finished_workers_ == num_workers_;
  };
  if (!progress_cb_) {
    cv_done_.wait(lock, done);
  } else {
    // Poll the completed fraction until the callback has fired, then wait
    // like the plain path (no further wake-ups competing with the workers).
    auto t0 = std::chrono::steady_clock::now();
    while (!done()) {
      cv_done_.wait_for(lock, std::chrono::microseconds(10), done);
      if (done())
        break;
      // completed (not merely claimed) fraction: with 12 workers all
      // morsels of a small source are claimed instantly.
      uint64_t done_rows = done_rows_.load(std::memory_order_relaxed);
      double p = done_rows >= total ? 1.0 : (double)done_rows / (double)total;
      if (p >= progress_threshold_) {
        double step_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - t0)
                             .count();
        lock.unlock();
        progress_cb_(p, step_ms);
        lock.lock();
        cv_done_.wait(lock, done);
        break;
      }
    }
  }
  body_ = nullptr;
}

void QjitWorkerPool::WorkerLoop(uint32_t worker_id) {
  uint64_t seen_generation = 0;
  for (;;) {
    const MorselBody *body;
    uint64_t total, morsel;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_start_.wait(lock, [this, seen_generation] {
        return shutdown_ || generation_ != seen_generation;
      });
      if (shutdown_)
        return;
      seen_generation = generation_;
      body = body_;
      total = total_;
      morsel = morsel_;
    }
    active_workers_.fetch_add(1, std::memory_order_relaxed);
    middleware::g_bg_active_threads.fetch_add(1, std::memory_order_relaxed);
    for (;;) {
      uint64_t begin = next_.fetch_add(morsel, std::memory_order_relaxed);
      if (begin >= total)
        break;
      uint64_t end = begin + morsel < total ? begin + morsel : total;
      (*body)(begin, end, worker_id);
      done_rows_.fetch_add(end - begin, std::memory_order_relaxed);
    }
    middleware::g_bg_active_threads.fetch_sub(1, std::memory_order_relaxed);
    active_workers_.fetch_sub(1, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_workers_++;
    }
    cv_done_.notify_one();
  }
}

} // namespace qjit

extern "C" void qjit_parallel_for(QjitQueryContext *ctx, uint64_t total,
                                  uint64_t morsel, QjitMorselFn fn) {
  auto *pool = static_cast<qjit::QjitWorkerPool *>(ctx->pool);
  pool->ParallelFor(total, morsel,
                    [ctx, fn](uint64_t begin, uint64_t end,
                              uint32_t worker_id) {
                      fn(ctx, begin, end, worker_id);
                    });
}
