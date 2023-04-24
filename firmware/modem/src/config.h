#include <Arduino.h>
#include <chrono>

#pragma once

#define ONE_S_IN_NS std::chrono::nanoseconds(1'000'000'000)
#define ONE_MS_IN_NS std::chrono::nanoseconds(1'000'000)
#define ONE_US_IN_NS std::chrono::nanoseconds(1'000)

struct Config
{
    uint32_t freq_lo;
    uint32_t freq_hi;
    uint32_t freq_sample;
    uint32_t freq_bit;
    uint32_t freq_sync_bit;

    size_t sdft_N;
    float sdft_r;

    float sdft_ema_alpha;
    float sdft_emv_alpha;

    const std::chrono::nanoseconds half_period_lo_ns() { return ONE_S_IN_NS / (2 * freq_lo); };
    const std::chrono::nanoseconds half_period_hi_ns() { return ONE_S_IN_NS / (2 * freq_hi); };
    const std::chrono::nanoseconds period_sample_ns() { return ONE_S_IN_NS / freq_sample; };
    const std::chrono::nanoseconds period_bit_ns() { return ONE_S_IN_NS / freq_bit; };
    const std::chrono::nanoseconds period_sync_bit_ns() { return ONE_S_IN_NS / freq_sync_bit; };

    Config(){};
};