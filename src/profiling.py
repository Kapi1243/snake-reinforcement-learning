"""
Performance profiling and optimization utilities.

Includes tools for:
- Profiling training loops
- Memory usage tracking
- Bottleneck identification
- Performance benchmarking

Author: Kacper Kowalski
Date: January 2026
"""

import time
import functools
from typing import Callable, Any, Dict, Optional
from pathlib import Path
import json
import cProfile
import pstats
from io import StringIO
import sys

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("psutil not available. Install with: pip install psutil")

try:
    import memory_profiler

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialize timer.

        Args:
            name: Name of the operation being timed
            verbose: Whether to print timing info
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.verbose:
            print(f"{self.name}: {self.elapsed:.4f}s")


def timeit(func: Callable) -> Callable:
    """Decorator to time function execution."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__}() took {elapsed:.4f}s")
        return result

    return wrapper


class PerformanceProfiler:
    """
    Comprehensive performance profiler.

    Tracks CPU time, memory usage, and function calls.
    """

    def __init__(self, output_dir: str = "profiling"):
        """
        Initialize profiler.

        Args:
            output_dir: Directory to save profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.profiler = cProfile.Profile()
        self.memory_samples = []
        self.timing_data: Dict[str, list] = {}

    def start(self):
        """Start profiling."""
        self.profiler.enable()
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def stop(self):
        """Stop profiling."""
        self.profiler.disable()

    def sample_memory(self, label: str = ""):
        """Sample current memory usage."""
        if PSUTIL_AVAILABLE:
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.memory_samples.append(
                {
                    "label": label,
                    "memory_mb": current_memory,
                    "delta_mb": current_memory - self.start_memory,
                }
            )

    def record_timing(self, operation: str, duration: float):
        """Record timing for a specific operation."""
        if operation not in self.timing_data:
            self.timing_data[operation] = []
        self.timing_data[operation].append(duration)

    def print_stats(self, sort_by: str = "cumulative", top_n: int = 20):
        """
        Print profiling statistics.

        Args:
            sort_by: How to sort results ('cumulative', 'time', 'calls')
            top_n: Number of top entries to show
        """
        s = StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.sort_stats(sort_by)
        ps.print_stats(top_n)
        print(s.getvalue())

    def save_stats(self, filename: str = "profile_stats.txt"):
        """Save profiling statistics to file."""
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            ps = pstats.Stats(self.profiler, stream=f)
            ps.sort_stats("cumulative")
            ps.print_stats()

        print(f"Profiling stats saved to {filepath}")

    def save_memory_profile(self, filename: str = "memory_profile.json"):
        """Save memory profiling data."""
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.memory_samples, f, indent=2)

        print(f"Memory profile saved to {filepath}")

    def save_timing_data(self, filename: str = "timing_data.json"):
        """Save timing data."""
        filepath = self.output_dir / filename

        # Compute statistics
        stats = {}
        for operation, timings in self.timing_data.items():
            import numpy as np

            stats[operation] = {
                "count": len(timings),
                "total": sum(timings),
                "mean": np.mean(timings),
                "std": np.std(timings),
                "min": min(timings),
                "max": max(timings),
            }

        with open(filepath, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Timing data saved to {filepath}")

    def generate_report(self):
        """Generate comprehensive performance report."""
        print("\n" + "=" * 60)
        print("PERFORMANCE PROFILING REPORT")
        print("=" * 60)

        # Memory summary
        if self.memory_samples and PSUTIL_AVAILABLE:
            print("\nMemory Usage:")
            print(f"  Starting: {self.start_memory:.2f} MB")
            final_memory = self.memory_samples[-1]["memory_mb"]
            print(f"  Final:    {final_memory:.2f} MB")
            print(f"  Delta:    {final_memory - self.start_memory:.2f} MB")

        # Timing summary
        if self.timing_data:
            import numpy as np

            print("\nOperation Timings:")
            for operation, timings in self.timing_data.items():
                print(f"  {operation}:")
                print(f"    Count: {len(timings)}")
                print(f"    Mean:  {np.mean(timings):.4f}s")
                print(f"    Total: {sum(timings):.4f}s")

        print("\n" + "=" * 60)


class Benchmark:
    """
    Benchmark utilities for comparing different implementations.
    """

    @staticmethod
    def compare_functions(
        functions: Dict[str, Callable], args: tuple = (), kwargs: dict = None, iterations: int = 100
    ) -> Dict[str, float]:
        """
        Compare execution time of multiple functions.

        Args:
            functions: Dictionary of name -> function
            args: Arguments to pass to functions
            kwargs: Keyword arguments to pass to functions
            iterations: Number of iterations to run

        Returns:
            Dictionary of name -> average time
        """
        kwargs = kwargs or {}
        results = {}

        print(f"\nBenchmarking {len(functions)} functions over {iterations} iterations...")

        for name, func in functions.items():
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            results[name] = avg_time
            print(f"  {name}: {avg_time*1000:.4f}ms")

        # Find fastest
        fastest = min(results, key=results.get)
        print(f"\nFastest: {fastest}")

        # Show relative performance
        print("\nRelative performance:")
        baseline = results[fastest]
        for name, time_val in sorted(results.items(), key=lambda x: x[1]):
            speedup = time_val / baseline
            print(f"  {name}: {speedup:.2f}x")

        return results


class ResourceMonitor:
    """Monitor system resources during execution."""

    def __init__(self):
        """Initialize resource monitor."""
        if not PSUTIL_AVAILABLE:
            raise ImportError("psutil required for ResourceMonitor")

        self.process = psutil.Process()
        self.samples = []

    def sample(self) -> Dict[str, float]:
        """Take a resource usage sample."""
        sample = {
            "timestamp": time.time(),
            "cpu_percent": self.process.cpu_percent(),
            "memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "num_threads": self.process.num_threads(),
        }

        self.samples.append(sample)
        return sample

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.samples:
            return {}

        import numpy as np

        cpu_values = [s["cpu_percent"] for s in self.samples]
        memory_values = [s["memory_mb"] for s in self.samples]

        return {
            "cpu": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
            },
            "memory": {
                "mean": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values),
            },
            "duration": self.samples[-1]["timestamp"] - self.samples[0]["timestamp"],
        }

    def save_samples(self, filepath: str):
        """Save samples to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.samples, f, indent=2)


def profile_training_loop(agent, env, num_episodes: int = 100, output_dir: str = "profiling"):
    """
    Profile a training loop.

    Args:
        agent: RL agent
        env: Environment
        num_episodes: Number of episodes to profile
        output_dir: Directory to save results
    """
    profiler = PerformanceProfiler(output_dir)

    print(f"Profiling training loop for {num_episodes} episodes...")

    profiler.start()

    for episode in range(num_episodes):
        episode_start = time.perf_counter()

        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_value(state, action, reward, next_state)

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_time = time.perf_counter() - episode_start
        profiler.record_timing("episode", episode_time)

        if episode % 10 == 0:
            profiler.sample_memory(f"Episode {episode}")

    profiler.stop()

    # Generate reports
    profiler.save_stats()
    profiler.save_memory_profile()
    profiler.save_timing_data()
    profiler.generate_report()


if __name__ == "__main__":
    print(f"psutil available: {PSUTIL_AVAILABLE}")
    print(f"memory_profiler available: {MEMORY_PROFILER_AVAILABLE}")

    # Demo timer
    with Timer("Example operation"):
        time.sleep(0.1)

    # Demo benchmark
    def method_a():
        return sum(range(1000))

    def method_b():
        return sum([i for i in range(1000)])

    Benchmark.compare_functions({"range": method_a, "list_comp": method_b}, iterations=1000)
