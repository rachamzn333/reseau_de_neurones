#pragma once
#include "tensor.h"
#include <string>

Images load_images(const std::string& idx_path);
Labels load_labels(const std::string& idx_path);
