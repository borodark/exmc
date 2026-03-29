# Four Point Seven Hours

The StochTree-Ex endurance benchmark has eight tests. Seven of them were merely
slow. The eighth — extreme-p, one thousand observations with five hundred
features — took four hours and forty-two minutes. 16,854 seconds. For a dataset
that fits in four megabytes of RAM.

## The Discovery

The benchmark was designed to find walls, and it found one. BART's job is
straightforward: fit 200 small decision trees to the residuals of the other
199, iterating 200 times (100 GFR passes for exploration, 100 MH passes for
refinement). At p=10 features, the smoke test finished in 67 seconds — slow by
production standards, but functional. At p=100, the high-p test took 581
seconds. At p=500, the relationship stopped being linear and became something
closer to punitive: 16,854 seconds.

The scaling was worse than O(p). Something structural was breaking down.

## The Investigation

The Rust NIF's `grow_recursive` function decides where to split a tree node.
For each candidate split, it needs to know: if I divide the observations in
this leaf at value `cp` on feature `feat`, what are the sufficient statistics
(count, sum, sum-of-squares) on each side? The implementation:

```rust
for feat in 0..n_features {          // 500 iterations at p=500
    for &cp in &cutpoints[feat] {    // ~100 sampled cutpoints per feature
        let (left, right) = split_stats(x, y, indices, feat, cp);
        // split_stats scans ALL leaf observations: O(n_leaf)
    }
}
```

For the root node with n=1,000 observations: 500 features × 100 cutpoints ×
1,000 scans = 50 million operations. Per tree. Per iteration. With 200 trees
and 200 iterations, the theoretical worst case is 2 trillion operations. Even
with smaller leaves deeper in the tree, the aggregate is astronomical.

The code was imported from a pedagogical implementation. It was correct. It
scaled as O(C × p × n_leaf) per leaf, where C is the number of candidate
cutpoints. At p=10, nobody noticed. At p=500, the constant factors stopped
being constant.

## The Revelation

The nested loop was not just slow — it was also imprecise. The `cutpoints`
vector sampled approximately 100 candidate split points per feature from the
data range. This meant the algorithm evaluated 100 out of potentially 1,000
unique split positions. It found good splits most of the time. But "most of
the time" is not the same as "always," and at p=500 with only n=1,000
observations, the signal-to-noise ratio of individual features is razor-thin.
Missing the best cutpoint on a critical feature compounds across 200 trees.

The realization: the correct data structure would not just be faster — it would
find better splits.

## The Fix

StochTree Python's `ForestTracker` provided the template. Two new data
structures in a new Rust module, `forest_tracker.rs`:

**SortedColumnIndex**: for each feature j, sort all n observation indices by
their value of x[·, j]. Built once at initialization. Cost: O(n × p × log n),
amortized across all 40,000 tree updates. Memory: p × n × 4 bytes (u32
indices). For p=500, n=1,000: two megabytes.

**ForestTracker**: maintains `leaf_assignment[tree][obs] = leaf_id` and
`leaf_stats[tree][leaf_id] = SuffStats`. Updated incrementally: on split,
partition the leaf's observations into two children; on prune, merge them back.

The split evaluation algorithm for a single feature: scan the pre-sorted
indices. Skip observations not in the target leaf. Accumulate a running
`left_stats`. Compute `right_stats = total - left_stats` via a new
`SuffStats::sub` method — O(1). Every position in the sorted order is a
natural cutpoint. One pass, all cutpoints, no sampling.

Cost per feature: O(n). Cost for all features: O(n × p). With rayon
parallelizing across features (scoped ThreadPool, capped at 16 threads), the
wall time per leaf is O(n × p / cores).

The refactored call sites: `grow_from_root_tracked` replaces the nested loops
with `tracker.find_best_split`. `mcmc_step_tracked` calls `apply_split` on
acceptance, `apply_prune` on prune acceptance, and never modifies the tracker
speculatively. `resample_leaves` iterates observations once using
`leaf_assignment` instead of recursively partitioning through the tree.

The Elixir API did not change. All optimization is inside the Rust NIF. Fourteen
tests pass. RMSE validated against StochTree Python 0.4.0.

## The Numbers

|                    | Before       | After        | Factor   |
|--------------------|--------------|--------------|----------|
| smoke (n=1K, p=10) | 67 s         | 20 s         | **3.4×** |
| extreme-p (p=500)  | 16,854 s     | 119 s        | **142×** |
| extreme-p RMSE     | 0.49         | 0.05         | **10×**  |

The RMSE column is the one that should not go unremarked. The old brute-force
approach sampled approximately 100 cutpoints per feature. The sorted scan
evaluates every possible cutpoint in one pass. It finds better splits because
it cannot miss any. A 142× speedup that simultaneously improves prediction
quality by 10× almost never happens. When it does, it means the old approach
was leaving accuracy on the table alongside the compute.

The rayon thread pool uses a scoped builder, not the global pool. When the
BEAM runs multi-chain BART via `Task.async_stream`, each chain gets its own
scoped pool. No thread contention between chains.

## The Lesson

The right data structure does not just make things faster. It makes them more
correct. The sorted scan did not merely avoid redundant work — it found better
splits by evaluating every candidate the brute-force approach was sampling
from. Four point seven hours of compute was not exploring the split space
thoroughly. It was exploring it wastefully, and missing the best answers in the
process.

The pedagogical implementation taught us the algorithm. The production
implementation taught us that the algorithm was never the bottleneck — the
enumeration was.

## P.S.

The full endurance suite finished. All eight tests.

|                          | Before           | After            | Factor     |
|--------------------------|------------------|------------------|------------|
| smoke (n=1K, p=10)       | 67,315 ms        | 23,295 ms        | **2.9×**   |
| medium-n (n=5K, p=10)    | 893,518 ms       | 372,400 ms       | **2.4×**   |
| large-n (n=10K, p=10)    | 1,097,831 ms     | 147,706 ms       | **7.4×**   |
| high-p (n=1K, p=100)     | 581,318 ms       | 28,192 ms        | **20.6×**  |
| extreme-p (n=1K, p=500)  | 16,854,194 ms    | 127,106 ms       | **133×**   |
| smooth-sin               | 396,301 ms       | 168,563 ms       | **2.4×**   |
| california               | 1,414,483 ms     | 335,719 ms       | **4.2×**   |
| multi-chain-4            | 392,365 ms       | 90,862 ms        | **4.3×**   |
| **Total**                | **21,697 s (6h)**| **1,293 s (22m)**| **16.8×**  |

The suite that took six hours runs in twenty-two minutes. The speedup scales
with the problem the ForestTracker was built for: at p=10 it is 2.9×; at p=100
it is 20.6×; at p=500 it is 133×. The sorted scan's advantage is proportional
to the number of cutpoints the brute-force approach was enumerating per feature
— and at p=500, that number was very large.

The large-n test deserves a note. At n=10K with p=10, the speedup is 7.4× —
larger than the p=10 smoke test's 2.9×. The old approach scaled with n_leaf ×
C; the sorted scan scales with n regardless of leaf size. Bigger datasets
benefit more because the per-leaf overhead is eliminated entirely.

The multi-chain test — four independent BART chains via `Task.async_stream` —
ran in 91 seconds total, 23 seconds per chain. The scoped rayon pools did not
contend. Four chains on 88 cores, each with its own 16-thread Rust pool,
sharing nothing.
