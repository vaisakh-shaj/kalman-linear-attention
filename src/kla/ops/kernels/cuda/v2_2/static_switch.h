/******************************************************************************
 * Static switch helper (from Mamba-1 / FlashAttention)
 ******************************************************************************/
#pragma once

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                          \
    if (COND) {                                  \
      constexpr static bool CONST_NAME = true;   \
      return __VA_ARGS__();                      \
    } else {                                     \
      constexpr static bool CONST_NAME = false;  \
      return __VA_ARGS__();                      \
    }                                            \
  }()
