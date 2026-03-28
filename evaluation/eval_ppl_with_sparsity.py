import json
import sys

from adasplash import enable_sparsity_stats, get_sparsity_stats, reset_sparsity_stats
import lm_eval


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/zhenyishen/Downloads/adasplash_dir/adasplash-1b-init"
    max_length = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    enable_sparsity_stats()

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path},max_length={max_length},dtype=auto,trust_remote_code=true",
        tasks=["wikitext"],
        batch_size=batch_size,
        device="cuda",
        log_samples=True,
    )

    stats = get_sparsity_stats()

    print("\n" + "=" * 80)
    print("ATTENTION SPARSITY STATISTICS (Zero Attention Score Ratio)")
    print("=" * 80)

    total_nonzero = 0
    total_elements = 0

    for layer_idx in sorted(stats.keys()):
        s = stats[layer_idx]
        zero_ratio = 1.0 - s["nonzero_elements"] / max(s["total_elements"], 1)
        print(f"  Layer {layer_idx:2d}: zero_ratio = {zero_ratio:.6f}  "
              f"(nonzero={s['nonzero_elements']:,}, total={s['total_elements']:,})")
        total_nonzero += s["nonzero_elements"]
        total_elements += s["total_elements"]

    overall_zero_ratio = 1.0 - total_nonzero / max(total_elements, 1)
    print(f"\n  Overall:  zero_ratio = {overall_zero_ratio:.6f}")
    print("=" * 80)

    output = {
        "per_layer": {str(k): {**v, "zero_ratio": 1.0 - v["nonzero_elements"] / max(v["total_elements"], 1)}
                      for k, v in stats.items()},
        "overall_zero_ratio": overall_zero_ratio,
        "lm_eval_results": results["results"],
    }

    output_path = "sparsity_stats.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
