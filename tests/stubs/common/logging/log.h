// Stand-in for the GPU build's logger (common/logging/log.h needs spdlog, a submodule the host
// tests do not check out): the same macros, doing nothing. Only headers that mention a log line
// inside an inline function are compiled against this.
#pragma once

#define LOG_GENERIC(log_class, log_level, format, ...) (void(0))
#define LOG_TRACE(log_class, ...) (void(0))
#define LOG_DEBUG(log_class, ...) (void(0))
#define LOG_INFO(log_class, ...) (void(0))
#define LOG_WARNING(log_class, ...) (void(0))
#define LOG_ERROR(log_class, ...) (void(0))
#define LOG_CRITICAL(log_class, ...) (void(0))
