/*** 
 * @Author: Flightmare
 * @Date: 2024-11-27 17:32:59 +0800
 * @LastEditTime: 2025-05-17 15:15:29 +0800
 * @LastEditors: Lac_Creeper
 * @Description: 
 * @FilePath: /src/flightmare/flightlib/src/common/logger.cpp
 */
#include "flightlib/common/logger.hpp"

namespace flightlib {

Logger::Logger(const std::string& name, const bool color)
  : sink_(std::cout.rdbuf()), colored_(color) {
  name_ = "[" + name + "]";

  if (name_.size() < NAME_PADDING)
    name_ = name_ + std::string(NAME_PADDING - name_.size(), ' ');
  else
    name_ = name_ + " ";

  sink_.precision(DEFAULT_PRECISION);
}

Logger::Logger(const std::string& name, const std::string& filename)
  : Logger(name, false) {
  if (!filename.empty()) {
    std::filebuf* fbuf = new std::filebuf;
    if (fbuf->open(filename, std::ios::out))
      sink_.rdbuf(fbuf);
    else
      warn("Could not open file %s. Logging to console!", filename);
  }
  sink_.precision(DEFAULT_PRECISION);
}

Logger::~Logger() {}

inline std::streamsize Logger::precision(const std::streamsize n) {
  return sink_.precision(n);
}

inline void Logger::scientific(const bool on) {
  if (on)
    sink_ << std::scientific;
  else
    sink_ << std::fixed;
}

void Logger::info(const std::string& message) const {
  if (colored_)
    sink_ << name_ << message << std::endl;
  else
    sink_ << name_ << INFO << message << std::endl;
}

void Logger::warn(const std::string& message) const {
  if (colored_)
    sink_ << YELLOW << name_ << message << RESET << std::endl;
  else
    sink_ << name_ << WARN << message << std::endl;
}

void Logger::error(const std::string& message) const {
  if (colored_)
    sink_ << RED << name_ << message << RESET << std::endl;
  else
    sink_ << name_ << ERROR << message << std::endl;
}

void Logger::fatal(const std::string& message) const {
  if (colored_)
    sink_ << RED << name_ << message << RESET << std::endl;
  else
    sink_ << name_ << FATAL << message << std::endl;
}

void Logger::debug(const std::string& message) const {
  if (colored_)
    sink_ << GREEN << name_ << message << RESET << std::endl;
  else
    sink_ << name_ << DEBUG << message << std::endl;
}

}  // namespace flightlib