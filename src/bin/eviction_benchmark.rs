use paged_infer::memory::kv_cache_manager::KvCacheManager;

#[derive(Clone)]
struct Seq {
    id: usize,
    remaining_tokens: usize,
    current_block_fill: usize,
    active: bool,
    long_running: bool,
}

fn run_simulation(enable_eviction: bool) -> (usize, usize, usize) {
    let block_size = 16;
    let mut manager = KvCacheManager::new(32, block_size);

    let mut seqs: Vec<Seq> = (0..4)
        .map(|i| Seq {
            id: i,
            remaining_tokens: 256,
            current_block_fill: 0,
            active: true,
            long_running: true,
        })
        .collect();

    // allocate initial blocks for long-running sequences
    for s in &seqs {
        let _ = manager.ensure_block_for_sequence(s.id, 0, enable_eviction);
    }

    let mut next_id = seqs.len();
    let mut completed = 0;
    let mut dropped = 0;

    for tick in 1..=600 {
        // new short request every 6 ticks
        if tick % 6 == 0 {
            seqs.push(Seq {
                id: next_id,
                remaining_tokens: 24,
                current_block_fill: 0,
                active: true,
                long_running: false,
            });
            next_id += 1;
        }

        for i in 0..seqs.len() {
            if !seqs[i].active {
                continue;
            }

            if seqs[i].current_block_fill == 0 {
                let ok =
                    manager.ensure_block_for_sequence(seqs[i].id, tick as u64, enable_eviction);
                if !ok {
                    seqs[i].active = false;
                    dropped += 1;
                    continue;
                }
            } else {
                manager.touch(seqs[i].id, tick as u64);
            }

            seqs[i].remaining_tokens = seqs[i].remaining_tokens.saturating_sub(1);
            seqs[i].current_block_fill = (seqs[i].current_block_fill + 1) % block_size;

            if seqs[i].remaining_tokens == 0 {
                seqs[i].active = false;
                manager.release_sequence(seqs[i].id);
                completed += 1;
                if seqs[i].long_running {
                    // restart long-running seq to keep pressure high
                    seqs.push(Seq {
                        id: next_id,
                        remaining_tokens: 256,
                        current_block_fill: 0,
                        active: true,
                        long_running: true,
                    });
                    next_id += 1;
                }
            }
        }
    }

    (completed, dropped, manager.active_sequences())
}

fn main() {
    let (c0, d0, a0) = run_simulation(false);
    let (c1, d1, a1) = run_simulation(true);

    println!("KV Eviction Benchmark (LRU)");
    println!("No eviction   => completed={c0}, dropped={d0}, active_end={a0}");
    println!("LRU eviction  => completed={c1}, dropped={d1}, active_end={a1}");

    let gain = if c0 == 0 {
        0.0
    } else {
        ((c1 as f64 - c0 as f64) / c0 as f64) * 100.0
    };
    println!("completed_gain_pct={:.2}", gain);
}
