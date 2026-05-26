"""Test fixtures for parser tests."""

SAMPLE_LOG = """\x1b[34m0.00.052.024\x1b[0m \x1b[32mI \x1b[0mlog_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
\x1b[34m0.00.052.030\x1b[0m \x1b[32mI \x1b[0m  - MTL0    : Apple M3 Pro (28753 MiB, 28753 MiB free)
\x1b[34m0.00.052.030\x1b[0m \x1b[32mI \x1b[0m  - BLAS    : Accelerate (0 MiB, 0 MiB free)
\x1b[34m0.00.052.036\x1b[0m \x1b[32mI \x1b[0m  - CPU     : Apple M3 Pro (36864 MiB, 36864 MiB free)
\x1b[34m0.00.052.047\x1b[0m \x1b[32mI \x1b[0msystem_info: n_threads = 8 (n_threads_batch = 8) / 11 | MTL : EMBED_LIBRARY = 1 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | MATMUL_INT8 = 1 | DOTPROD = 1 | ACCELERATE = 1 | REPACK = 1 | 
\x1b[34m0.00.053.497\x1b[0m \x1b[32mI \x1b[0msrv    load_model: loading model '/Users/dmitry/models/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'
\x1b[34m0.02.499.412\x1b[0m \x1b[32mI \x1b[0mcommon_speculative_impl_draft_mtp: - n_max=2, n_min=0, p_min=0.70, n_embd=2048
\x1b[34m10.54.639.911\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | prompt eval time =    8119.49 ms /  3138 tokens (    2.59 ms per token,   386.48 tokens per second)
\x1b[34m10.54.639.914\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |        eval time =   41904.01 ms /  1291 tokens (   32.46 ms per token,    30.81 tokens per second)
\x1b[34m10.54.639.915\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |       total time =   50023.50 ms /  4429 tokens
\x1b[34m10.54.639.915\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |    graphs reused =        281
\x1b[34m10.54.639.916\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | draft acceptance = 0.89809 (  705 accepted /   785 generated)
\x1b[34m10.54.640.498\x1b[0m \x1b[32mI \x1b[0mslot      release: id  0 | task 0 | stop processing: n_tokens = 4430, truncated = 0
\x1b[34m11.32.600.886\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 2 | prompt eval time =   29831.76 ms / 11199 tokens (    2.66 ms per token,   375.41 tokens per second)
\x1b[34m11.32.600.889\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 2 |        eval time =    8043.07 ms /   264 tokens (   30.47 ms per token,    32.82 tokens per second)
\x1b[34m11.32.600.889\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 2 |       total time =   37874.83 ms / 11463 tokens
\x1b[34m11.32.600.890\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 2 |    graphs reused =        349
\x1b[34m11.32.600.890\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 2 | draft acceptance = 0.94767 (  163 accepted /   172 generated)
\x1b[34m11.32.601.369\x1b[0m \x1b[32mI \x1b[0mslot      release: id  0 | task 2 | stop processing: n_tokens = 11462, truncated = 0
"""

SAMPLE_LOG_NO_ANSI = """\
0.00.052.024 I log_info: verbosity = 3
0.00.052.030 I   - MTL0    : Apple M2 Max (96096 MiB, 96096 MiB free)
0.00.052.047 I system_info: n_threads = 10 (n_threads_batch = 10) / 16
0.00.053.497 I srv    load_model: loading model '/Users/dmitry/models/Llama-3.1-8B-Q8_0.gguf'
0.02.499.412 I common_speculative_impl_draft_mtp: - n_max=4, n_min=0, p_min=0.50, n_embd=4096
10.54.639.911 I slot print_timing: id  0 | task 0 | prompt eval time =    8119.49 ms /  3138 tokens (    2.59 ms per token,   386.48 tokens per second)
10.54.639.914 I slot print_timing: id  0 | task 0 |        eval time =   41904.01 ms /  1291 tokens (   32.46 ms per token,    30.81 tokens per second)
10.54.639.915 I slot print_timing: id  0 | task 0 |       total time =   50023.50 ms /  4429 tokens
10.54.639.916 I slot print_timing: id  0 | task 0 | draft acceptance = 0.89809 (  705 accepted /   785 generated)
10.54.640.498 I slot      release: id  0 | task 0 | stop processing: n_tokens = 4430, truncated = 0
"""

SAMPLE_LOG_GARBAGE = """\
random garbage line 1
another random line
\x1b[34m0.00.052.030\x1b[0m \x1b[32mI \x1b[0m  - MTL0    : Apple M1 (16384 MiB, 16384 MiB free)
more garbage
incomplete line without proper format
\x1b[34m10.54.639.911\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | prompt eval time =    8119.49 ms /  3138 tokens (    2.59 ms per token,   386.48 tokens per second)
\x1b[34m10.54.639.914\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |        eval time =   41904.01 ms /  1291 tokens (   32.46 ms per token,    30.81 tokens per second)
\x1b[34m10.54.639.916\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | draft acceptance = 0.75000 (  500 accepted /   667 generated)
"""

SAMPLE_LOG_EMPTY = ""

SAMPLE_LOG_TRUNCATED = """\
\x1b[34m0.00.052.030\x1b[0m \x1b[32mI \x1b[0m  - MTL0    : Apple M3 (8192 MiB, 8192 MiB free)
\x1b[34m10.54.639.911\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | prompt eval time =    8119.49 ms /  3138 tokens
"""

PARSED_JSON_FIXTURE = {
    "runs": [
        {
            "id": "run_1",
            "metadata": {
                "model": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
                "quantization": "Q4_K_XL",
                "llama_cpp_version": "",
                "system": {
                    "chip": "Apple M3 Pro",
                    "chip_type": "M3",
                    "unified_memory_mb": 28753,
                    "macos_version": "",
                    "cpu_threads": 8,
                    "cpu_total_threads": 11,
                },
                "mtp_config": {"n_max": 2, "n_min": 0, "p_min": 0.7},
            },
            "measurements": [
                {
                    "n_decoded": 1291,
                    "n_tokens": 4430,
                    "prompt_eval_time_ms": 8119.49,
                    "eval_time_ms": 41904.01,
                    "total_time_ms": 50023.50,
                    "prompt_tokens_per_second": 386.48,
                    "generation_tokens_per_second": 30.81,
                    "draft_acceptance_rate": 0.89809,
                    "n_drafts_generated": 785,
                    "n_drafts_accepted": 705,
                    "truncated": 0,
                },
                {
                    "n_decoded": 264,
                    "n_tokens": 11462,
                    "prompt_eval_time_ms": 29831.76,
                    "eval_time_ms": 8043.07,
                    "total_time_ms": 37874.83,
                    "prompt_tokens_per_second": 375.41,
                    "generation_tokens_per_second": 32.82,
                    "draft_acceptance_rate": 0.94767,
                    "n_drafts_generated": 172,
                    "n_drafts_accepted": 163,
                    "truncated": 0,
                },
            ],
            "warnings": [],
        }
    ],
    "parse_warnings": [],
}

ANALYSIS_JSON_FIXTURE = {
    "run_id": "run_1",
    "metrics": {
        "avg_generation_tps": 31.82,
        "std_generation_tps": 1.41,
        "min_generation_tps": 30.81,
        "max_generation_tps": 32.82,
        "median_generation_tps": 31.815,
        "avg_prompt_tps": 380.95,
        "std_prompt_tps": 6.05,
        "avg_acceptance_rate": 0.9229,
        "std_acceptance_rate": 0.0248,
        "context_tps_correlation": -0.5,
        "context_degradation_rate": -0.1,
        "tps_variance": 1.99,
        "tps_cv": 0.044,
    },
    "mtp_setting_comparisons": [
        {
            "setting": 2,
            "count": 2,
            "avg_tps": 31.82,
            "avg_acceptance_rate": 0.9229,
            "avg_context_length": 7946.0,
            "min_tps": 30.81,
            "max_tps": 32.82,
            "tps_std": 1.41,
            "tps_cv": 0.044,
        }
    ],
    "summary": {
        "avg_generation_tps": 31.82,
        "avg_acceptance_rate": 92.29,
        "stability": "stable",
        "total_measurements": 2,
    },
}

RECOMMENDATION_JSON_FIXTURE = {
    "run_id": "run_1",
    "recommended_setting": 2,
    "recommended": {
        "mtp_setting": 2,
        "avg_throughput_uptick": None,
        "long_context_efficiency": "moderate",
        "stability": "stable",
        "memory_overhead_estimate_mb": None,
        "reasoning": [
            "Long-context efficiency: moderate",
            "Stability: stable",
        ],
    },
    "all_recommendations": [
        {
            "mtp_setting": 2,
            "avg_throughput_uptick": None,
            "long_context_efficiency": "moderate",
            "stability": "stable",
            "memory_overhead_estimate_mb": None,
            "reasoning": [
                "Long-context efficiency: moderate",
                "Stability: stable",
            ],
        }
    ],
    "summary_text": "Recommended MTP setting: 2",
}

# Multi-run log fixture (two server instances with different MTP configs)
SAMPLE_LOG_MULTI_RUN = """\x1b[34m0.00.052.024\x1b[0m \x1b[32mI \x1b[0mlog_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
\x1b[34m0.00.052.030\x1b[0m \x1b[32mI \x1b[0m  - MTL0    : Apple M3 Pro (28753 MiB, 28753 MiB free)
\x1b[34m0.00.052.047\x1b[0m \x1b[32mI \x1b[0msystem_info: n_threads = 8 (n_threads_batch = 8) / 11
\x1b[34m0.00.053.497\x1b[0m \x1b[32mI \x1b[0msrv    load_model: loading model '/Users/dmitry/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'
\x1b[34m0.02.499.412\x1b[0m \x1b[32mI \x1b[0mcommon_speculative_impl_draft_mtp: - n_max=1, n_min=0, p_min=0.70, n_embd=2048
\x1b[34m10.54.639.911\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | prompt eval time =    8119.49 ms /  3138 tokens (    2.59 ms per token,   386.48 tokens per second)
\x1b[34m10.54.639.914\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |        eval time =   41904.01 ms /  1291 tokens (   32.46 ms per token,    30.81 tokens per second)
\x1b[34m10.54.639.915\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |       total time =   50023.50 ms /  4429 tokens
\x1b[34m10.54.639.916\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | draft acceptance = 0.89809 (  705 accepted /   785 generated)
\x1b[34m10.54.640.498\x1b[0m \x1b[32mI \x1b[0mslot      release: id  0 | task 0 | stop processing: n_tokens = 4430, truncated = 0
\x1b[34m0.00.055.817\x1b[0m \x1b[32mI \x1b[0mlog_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
\x1b[34m0.00.055.822\x1b[0m \x1b[32mI \x1b[0m  - MTL0    : Apple M3 Pro (28753 MiB, 28753 MiB free)
\x1b[34m0.00.055.843\x1b[0m \x1b[32mI \x1b[0msystem_info: n_threads = 8 (n_threads_batch = 8) / 11
\x1b[34m0.00.057.268\x1b[0m \x1b[32mI \x1b[0msrv    load_model: loading model '/Users/dmitry/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf'
\x1b[34m0.02.037.728\x1b[0m \x1b[32mI \x1b[0mcommon_speculative_impl_draft_mtp: - n_max=2, n_min=0, p_min=0.70, n_embd=2048
\x1b[34m20.54.639.911\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | prompt eval time =    9119.49 ms /  3138 tokens (    2.90 ms per token,   344.48 tokens per second)
\x1b[34m20.54.639.914\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |        eval time =   42904.01 ms /  1291 tokens (   33.23 ms per token,    30.11 tokens per second)
\x1b[34m20.54.639.915\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 |       total time =   52023.50 ms /  4429 tokens
\x1b[34m20.54.639.916\x1b[0m \x1b[32mI \x1b[0mslot print_timing: id  0 | task 0 | draft acceptance = 0.92000 (  800 accepted /   870 generated)
\x1b[34m20.54.640.498\x1b[0m \x1b[32mI \x1b[0mslot      release: id  0 | task 0 | stop processing: n_tokens = 4430, truncated = 0
"""
