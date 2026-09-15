Read-only review of the NEON score tile

Assembly: `target/release/deps/paged_infer-64c151c92726cdc3.s`
SHA-256: `9a9308e1cb75013bda164c15c2b46f5c27c2dd20b2c9749fd0f31c22f0aeba37`
Closure: `SharedPrefixPlan::run::closure`, lines 115810–122754.

The 2-query × 2-key primitive is fully inlined. Hot loop `LBB296_82`, lines 116590–116622, keeps all 16 accumulators in v0–v7 and v16–v23. It has 16 vector FMAs and eight paired loads (16 vector loads), with no stack accesses and no helper calls. Query vectors v26/v28, for example, feed both key0 and key1 before replacement. The compiler uses the same query/key FMA operand ordering as the existing dot_multi loop.

The vector remainder `LBB296_86` also has no stack accesses or calls. The no-scalar-tail reductions remain vector additions followed by horizontal additions. For dim64 or128, execution takes this no-scalar-tail path.

Existing four-query single-key loop `LBB296_193`, lines117476–117511, has ten paired loads for16 FMAs. Two candidate calls therefore use32 vector loads versus40 for two existing calls covering four queries and two keys. FMA count stays32 per16-coordinate slice. This verifies the intended load reduction, not a timing win.

The surrounding closure still stores/reloads integer loop metadata and saves callee-saved registers. Those are outside the vector accumulation loop; they are not accumulator spills. Bounds-check branches and result reduction/scatter overhead remain. Wider instruction footprint and pairing overhead could offset the load reduction, so use the alternating kernel measurements to decide whether to retain the candidate.

Excerpt: `score-tile-vector-loop.s`. No builds, tests, benchmark runs, disassembly generation, source edits, or documentation edits were performed during this review.
